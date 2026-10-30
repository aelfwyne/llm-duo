//! GGUF metadata reader (header + tensor table only; never reads weights) and
//! the derived model description used by the VRAM estimate.

use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::util::{GIB, MIB};

#[derive(Clone, Debug)]
pub enum GVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<GVal>),
    /// large or string array that was skipped: only its length is known
    Skipped(u64),
}

impl GVal {
    pub fn as_str(&self) -> Option<&str> {
        if let GVal::Str(s) = self { Some(s) } else { None }
    }
    fn num(&self) -> Option<f64> {
        match self {
            GVal::Int(i) => Some(*i as f64),
            GVal::Float(f) => Some(*f),
            GVal::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }
    pub fn display(&self) -> String {
        match self {
            GVal::Int(i) => i.to_string(),
            GVal::Float(f) => super::util::py_float(*f),
            GVal::Bool(b) => if *b { "True".into() } else { "False".into() },
            GVal::Str(s) => s.clone(),
            GVal::Skipped(n) => format!("<array of {n}>"),
            GVal::Arr(a) => {
                let items: Vec<String> = a.iter().take(80).map(|x| match x {
                    GVal::Str(s) => format!("'{s}'"),
                    o => o.display(),
                }).collect();
                if a.len() > 80 {
                    format!("[{}] … ({} items)", items.join(", "), a.len())
                } else {
                    format!("[{}]", items.join(", "))
                }
            }
        }
    }
}

/// Python `scalar()`: lists collapse to their max, unknown → default.
fn scalar(v: Option<&GVal>, default: f64) -> f64 {
    match v {
        Some(GVal::Arr(a)) => {
            let nums: Vec<f64> = a.iter().filter_map(|x| x.num()).collect();
            if nums.is_empty() { default } else { nums.iter().cloned().fold(f64::MIN, f64::max) }
        }
        Some(GVal::Skipped(_)) | None => default,
        Some(x) => x.num().unwrap_or(default),
    }
}

/// `int(scalar(v, d) or 0)`-style helper
fn si(v: Option<&GVal>, d: i64) -> i64 {
    scalar(v, d as f64) as i64
}

fn scalar_size(t: u32) -> Option<usize> {
    Some(match t {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4 | 5 | 6 => 4,
        10 | 11 | 12 => 8,
        _ => return None,
    })
}

struct Rd<R: Read + Seek> {
    f: BufReader<R>,
}

impl<R: Read + Seek> Rd<R> {
    fn bytes(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.f.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.f.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.f.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        // seek_relative keeps the read buffer; a plain seek() throws it away and re-reads,
        // which made skipping a 262k-token vocabulary take ~30 s on large models
        self.f.seek_relative(n as i64)
    }
    fn str(&mut self) -> std::io::Result<String> {
        let n = self.u64()? as usize;
        if n > (1 << 30) {
            return Err(std::io::Error::other("string too long"));
        }
        Ok(String::from_utf8_lossy(&self.bytes(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> std::io::Result<GVal> {
        let size = scalar_size(t).unwrap();
        let b = self.bytes(size)?;
        Ok(match t {
            0 => GVal::Int(b[0] as i64),
            1 => GVal::Int(b[0] as i8 as i64),
            2 => GVal::Int(u16::from_le_bytes([b[0], b[1]]) as i64),
            3 => GVal::Int(i16::from_le_bytes([b[0], b[1]]) as i64),
            4 => GVal::Int(u32::from_le_bytes(b[..4].try_into().unwrap()) as i64),
            5 => GVal::Int(i32::from_le_bytes(b[..4].try_into().unwrap()) as i64),
            6 => GVal::Float(f32::from_le_bytes(b[..4].try_into().unwrap()) as f64),
            7 => GVal::Bool(b[0] != 0),
            10 => GVal::Int(u64::from_le_bytes(b[..8].try_into().unwrap()) as i64),
            11 => GVal::Int(i64::from_le_bytes(b[..8].try_into().unwrap())),
            12 => GVal::Float(f64::from_le_bytes(b[..8].try_into().unwrap())),
            _ => unreachable!(),
        })
    }
    fn val(&mut self, t: u32) -> std::io::Result<GVal> {
        if scalar_size(t).is_some() {
            return self.scalar(t);
        }
        match t {
            8 => Ok(GVal::Str(self.str()?)),
            9 => {
                let at = self.u32()?;
                let n = self.u64()?;
                if let Some(size) = scalar_size(at) {
                    if n <= 8192 {
                        let mut v = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            v.push(self.scalar(at)?);
                        }
                        return Ok(GVal::Arr(v));
                    }
                    self.skip(size as u64 * n)?;
                    return Ok(GVal::Skipped(n));
                }
                if at == 8 {
                    for _ in 0..n {
                        let ln = self.u64()?;
                        self.skip(ln)?;
                    }
                    return Ok(GVal::Skipped(n));
                }
                if n > 1_000_000 {
                    return Err(std::io::Error::other("nested array too large"));
                }
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(self.val(at)?);
                }
                Ok(GVal::Arr(v))
            }
            _ => Err(std::io::Error::other(format!("unknown GGUF value type {t}"))),
        }
    }
}

pub type Kv = HashMap<String, GVal>;

/// Return (kv dict, {tensor_name: bytes}) for one GGUF file.
pub fn read_gguf(path: &Path) -> Result<(Kv, HashMap<String, f64>), String> {
    let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let total_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut r = Rd { f: BufReader::with_capacity(1 << 20, f) };
    let e = |x: std::io::Error| format!("{}: {x}", path.display());
    if r.bytes(4).map_err(e)? != b"GGUF" {
        return Err(format!("{} is not a GGUF file", path.display()));
    }
    let version = r.u32().map_err(e)?;
    if version < 2 {
        return Err("GGUF v1 is not supported".into());
    }
    let n_tensors = r.u64().map_err(e)?;
    let n_kv = r.u64().map_err(e)?;
    let mut kv = Kv::new();
    for _ in 0..n_kv {
        let key = r.str().map_err(e)?;
        let t = r.u32().map_err(e)?;
        let v = r.val(t).map_err(e)?;
        kv.insert(key, v);
    }
    let mut tensors: Vec<(String, u64)> = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.str().map_err(e)?;
        let nd = r.u32().map_err(e)?;
        r.skip(8 * nd as u64).map_err(e)?;
        let _ttype = r.u32().map_err(e)?;
        let off = r.u64().map_err(e)?;
        tensors.push((name, off));
    }
    let align = match kv.get("general.alignment") {
        Some(v) => scalar(Some(v), 32.0) as u64,
        None => 32,
    };
    let align = if align == 0 { 32 } else { align };
    let pos = r.f.stream_position().map_err(e)?;
    let data_start = pos.div_ceil(align) * align;
    let data_len = total_len.saturating_sub(data_start);
    tensors.sort_by_key(|t| t.1);
    let mut sizes = HashMap::new();
    for i in 0..tensors.len() {
        let end = if i + 1 < tensors.len() { tensors[i + 1].1 } else { data_len };
        sizes.insert(tensors[i].0.clone(), end.saturating_sub(tensors[i].1) as f64);
    }
    Ok((kv, sizes))
}

pub fn shard_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").unwrap())
}

pub fn shard_files(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(m) = shard_re().captures(&name) else { return vec![path.to_path_buf()] };
    let total: usize = m[2].parse().unwrap_or(1);
    let mut files = Vec::new();
    for i in 1..=total {
        let nn = shard_re().replace(&name, format!("-{:05}-of-{:05}.gguf", i, total).as_str()).into_owned();
        let p = path.with_file_name(nn);
        if p.exists() {
            files.push(p);
        }
    }
    if files.is_empty() { vec![path.to_path_buf()] } else { files }
}

/// Read only general.architecture from a GGUF header (fast).
pub fn read_arch(path: &Path) -> Option<String> {
    let f = File::open(path).ok()?;
    let mut r = Rd { f: BufReader::new(f) };
    if r.bytes(4).ok()? != b"GGUF" {
        return None;
    }
    r.u32().ok()?;
    let _nt = r.u64().ok()?;
    let n_kv = r.u64().ok()?;
    for _ in 0..n_kv.min(64) {
        let key = r.str().ok()?;
        let t = r.u32().ok()?;
        if t == 8 {
            let val = r.str().ok()?;
            if key == "general.architecture" {
                return Some(val);
            }
        } else if let Some(s) = scalar_size(t) {
            r.skip(s as u64).ok()?;
        } else {
            return None; // arrays before the arch key: give up
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct MtpCand {
    pub rel: String,
    pub arch: String,
    pub ok: Option<bool>,
    pub note: String,
    pub size: f64,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub files: Vec<PathBuf>,
    pub kv: Kv,
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: f64,
    pub n_ctx_train: i64,
    pub n_head: f64,
    pub kv_heads: Vec<f64>,
    pub k_len: f64,
    pub v_len: f64,
    pub swa: f64,
    pub k_len_swa: f64,
    pub v_len_swa: f64,
    pub swa_layers: Vec<bool>,
    pub swa_guessed: bool,
    pub swa_note: String,
    pub nextn: usize,
    pub rs_bytes_per_layer: f64,
    pub rs_layers: Vec<bool>,
    pub n_vocab: f64,
    pub n_expert: i64,
    pub layer_bytes: Vec<f64>,
    pub expert_bytes: Vec<f64>,
    pub output_bytes: f64,
    pub tok_embd_bytes: f64,
    pub total_bytes: f64,
    pub tied_output: bool,
    pub mtp_candidates: Vec<MtpCand>,
}

impl ModelInfo {
    pub fn load(path: &Path) -> Result<Self, String> {
        let files = shard_files(path);
        let mut kv = Kv::new();
        let mut sizes: HashMap<String, f64> = HashMap::new();
        for (i, p) in files.iter().enumerate() {
            let (k, s) = read_gguf(p)?;
            if i == 0 {
                kv = k;
            }
            sizes.extend(s);
        }
        let arch = kv.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        let a = arch.clone();
        let g = |key: &str| kv.get(&format!("{a}.{key}"));

        let n_layer = si(g("block_count"), 0).max(0) as usize;
        let n_embd = si(g("embedding_length"), 0) as f64;
        let n_ctx_train = si(g("context_length"), 0);
        let n_head = si(g("attention.head_count"), 0) as f64;
        let mut kv_heads: Vec<f64> = match g("attention.head_count_kv") {
            Some(GVal::Arr(list)) => {
                let mut v: Vec<f64> = list.iter().map(|x| x.num().unwrap_or(0.0).trunc()).take(n_layer).collect();
                v.resize(n_layer, 0.0);
                v
            }
            other => {
                let h = if other.is_none() { n_head } else { si(other, n_head as i64) as f64 };
                vec![h; n_layer]
            }
        };
        let head_dim = if n_head != 0.0 { (n_embd / n_head).floor() } else { 0.0 };
        let mut k_len = si(g("attention.key_length"), head_dim as i64) as f64;
        let mut v_len = si(g("attention.value_length"), head_dim as i64) as f64;

        // MLA (DeepSeek-style): compressed latent cache
        if let Some(kl) = g("attention.kv_lora_rank") {
            let kl_n = scalar(Some(kl), 0.0);
            if kl_n != 0.0 {
                let rope = si(g("rope.dimension_count"), 64) as f64;
                let mut nh: Vec<f64> = kv_heads.iter().map(|h| if *h != 0.0 { 1.0 } else { 0.0 }).collect();
                if nh.is_empty() {
                    nh = vec![1.0; n_layer];
                }
                kv_heads = nh;
                k_len = kl_n.trunc() + rope;
                v_len = 0.0;
            }
        }
        // Hybrid models with periodic full-attention layers
        if let Some(f) = g("full_attention_interval") {
            let fv = scalar(Some(f), 1.0) as i64;
            if scalar(Some(f), 0.0) != 0.0 {
                let fai = if fv == 0 { 1 } else { fv };
                kv_heads = kv_heads
                    .iter()
                    .enumerate()
                    .map(|(i, h)| if ((i as i64) + 1) % fai == 0 { *h } else { 0.0 })
                    .collect();
            }
        }
        let swa = si(g("attention.sliding_window"), 0) as f64;
        let k_len_swa = si(g("attention.key_length_swa"), k_len as i64) as f64;
        let v_len_swa = si(g("attention.value_length_swa"), v_len as i64) as f64;
        let mut swa_layers = vec![false; n_layer];
        let mut swa_guessed = false;
        let mut swa_note = String::new();
        if swa != 0.0 {
            let pat = g("attention.sliding_window_pattern");
            if let Some(GVal::Arr(p)) = pat.filter(|p| matches!(p, GVal::Arr(a) if !a.is_empty())) {
                swa_layers = (0..n_layer)
                    .map(|i| p.get(i).map(|x| x.num().unwrap_or(0.0) != 0.0).unwrap_or(false))
                    .collect();
                swa_note = "per-layer pattern from GGUF".into();
            } else {
                let mut period: Option<i64> = match pat {
                    Some(GVal::Int(n)) if *n > 1 => Some(*n),
                    Some(GVal::Float(f)) if *f > 1.0 => Some(*f as i64),
                    _ => None,
                };
                if let Some(p) = period {
                    swa_note = format!("1 global layer every {p} (from GGUF)");
                } else {
                    period = match a.as_str() {
                        "gemma2" => Some(2),
                        "gemma3" => Some(6),
                        "gemma3n" => Some(5),
                        "gemma4" => Some(6),
                        "cohere2" => Some(4),
                        "gpt-oss" | "gpt_oss" => Some(2),
                        _ => None,
                    };
                    swa_guessed = true;
                    swa_note = match period {
                        Some(p) => format!("assumed 1 global layer every {p}"),
                        None => "unknown SWA layout - assuming all layers are global".into(),
                    };
                }
                if let Some(p) = period {
                    swa_layers = (0..n_layer).map(|i| (i as i64 % p) < (p - 1)).collect();
                }
            }
        }
        // Recurrent / linear-attention layers: fixed-size f32 state per sequence.
        let nextn = si(g("nextn_predict_layers"), 0).max(0) as usize;
        let d_state = si(g("ssm.state_size"), 0) as f64;
        let d_inner = si(g("ssm.inner_size"), 0) as f64;
        let mut rs_bytes_per_layer = 0.0;
        let mut rs_layers = vec![false; n_layer];
        if d_state != 0.0 && d_inner != 0.0 {
            let kc = si(g("ssm.conv_kernel"), 4);
            let k_conv = if kc == 0 { 4.0 } else { kc as f64 };
            let n_group = si(g("ssm.group_count"), 0) as f64;
            let conv = (k_conv - 1.0) * (d_inner + 2.0 * n_group * d_state);
            let ssm = d_inner * d_state;
            rs_bytes_per_layer = (conv + ssm) * 4.0;
            let last = n_layer.saturating_sub(nextn);
            rs_layers = (0..n_layer).map(|i| kv_heads[i] == 0.0 && i < last).collect();
        }
        let shared = si(g("attention.shared_kv_layers"), 0).max(0) as usize;
        if shared > 0 {
            for h in kv_heads.iter_mut().skip(n_layer.saturating_sub(shared)) {
                *h = 0.0;
            }
        }
        let n_vocab = match kv.get("tokenizer.ggml.tokens") {
            Some(GVal::Skipped(n)) => *n as f64,
            Some(GVal::Arr(a)) => a.len() as f64,
            _ => si(g("vocab_size"), 32000) as f64,
        };
        let n_expert = si(g("expert_count"), 0);
        let mut layer_bytes = vec![0.0; n_layer];
        let mut expert_bytes = vec![0.0; n_layer];
        let mut output_bytes = 0.0;
        let tok_embd_bytes = sizes.get("token_embd.weight").cloned().unwrap_or(0.0);
        let total_bytes: f64 = sizes.values().sum();
        static BLK: OnceLock<Regex> = OnceLock::new();
        let blk = BLK.get_or_init(|| Regex::new(r"^blk\.(\d+)\.").unwrap());
        for (name, sz) in &sizes {
            if let Some(m) = blk.captures(name) {
                if let Ok(i) = m[1].parse::<usize>() {
                    if i < n_layer {
                        layer_bytes[i] += sz;
                        if name.contains("_exps") {
                            expert_bytes[i] += sz;
                        }
                    }
                }
            } else if name.starts_with("output") {
                output_bytes += sz;
            }
        }
        let tied_output = !sizes.contains_key("output.weight");
        if tied_output {
            output_bytes += tok_embd_bytes;
        }
        Ok(ModelInfo {
            path: path.to_path_buf(),
            files,
            kv,
            arch,
            n_layer,
            n_embd,
            n_ctx_train,
            n_head,
            kv_heads,
            k_len,
            v_len,
            swa,
            k_len_swa,
            v_len_swa,
            swa_layers,
            swa_guessed,
            swa_note,
            nextn,
            rs_bytes_per_layer,
            rs_layers,
            n_vocab,
            n_expert,
            layer_bytes,
            expert_bytes,
            output_bytes,
            tok_embd_bytes,
            total_bytes,
            tied_output,
            mtp_candidates: Vec::new(),
        })
    }

    pub fn chat_template(&self) -> Option<String> {
        self.kv.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![self.arch.clone(), format!("{} layers", self.n_layer), format!("{:.2} GiB", self.total_bytes / GIB)];
        if self.n_expert != 0 {
            parts.push(format!("MoE {} experts", self.n_expert));
        }
        if self.n_ctx_train != 0 {
            parts.push(format!("train ctx {}", super::util::fmt_thousands(self.n_ctx_train)));
        }
        if self.rs_layers.iter().any(|x| *x) {
            let n_attn = self.kv_heads.iter().filter(|h| **h != 0.0).count();
            parts.push(format!("hybrid: {} attention / {} recurrent layers", n_attn, self.rs_layers.iter().filter(|x| **x).count()));
        }
        if self.swa != 0.0 {
            let n_glob = (0..self.n_layer).filter(|&i| self.kv_heads[i] != 0.0 && !self.swa_layers[i]).count();
            parts.push(format!("SWA {} ({} global layers)", super::util::fmt_thousands(self.swa as i64), n_glob));
        }
        if self.files.len() > 1 {
            parts.push(format!("{} shards", self.files.len()));
        }
        parts.join("  •  ")
    }

    pub fn metadata_text(&self) -> String {
        let ints = |v: &[f64]| -> String {
            format!("[{}]", v.iter().map(|x| (*x as i64).to_string()).collect::<Vec<_>>().join(", "))
        };
        let swa_idx: Vec<String> = self.swa_layers.iter().enumerate().filter(|(_, x)| **x).map(|(i, _)| i.to_string()).collect();
        let n_rs = self.rs_layers.iter().filter(|x| **x).count();
        let exp_total: f64 = self.expert_bytes.iter().sum();
        let mut lines = vec![
            format!("File(s): {}", self.files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(", ")),
            String::new(),
            "Derived:".into(),
            format!("  arch={} layers={} embd={} heads={}", self.arch, self.n_layer, self.n_embd as i64, self.n_head as i64),
            format!("  kv_heads per layer={}", ints(&self.kv_heads)),
            format!(
                "  k_len={} v_len={} k_len_swa={} v_len_swa={}",
                self.k_len as i64, self.v_len as i64, self.k_len_swa as i64, self.v_len_swa as i64
            ),
            format!(
                "  sliding_window={}  ({})",
                self.swa as i64,
                if self.swa_note.is_empty() { "no SWA" } else { &self.swa_note }
            ),
            format!("  swa layers=[{}]", swa_idx.join(", ")),
            format!(
                "  recurrent layers={}  state/layer/seq={:.2} MiB  nextn(MTP) layers={}",
                n_rs,
                self.rs_bytes_per_layer / MIB,
                self.nextn
            ),
            format!("  vocab={} experts={}", self.n_vocab as i64, self.n_expert)
                + &(if self.n_expert != 0 {
                    format!(
                        "  expert weights={:.2} GiB ({:.3} GiB per layer)",
                        exp_total / GIB,
                        exp_total / (self.n_layer.max(1) as f64) / GIB
                    )
                } else {
                    String::new()
                }),
            format!("  weights total={:.2} GiB  output={:.2} GiB", self.total_bytes / GIB, self.output_bytes / GIB)
                + if self.tied_output { "  (tied to token_embd)" } else { "" },
            String::new(),
            "Special tokens (what llama.cpp uses to stop generation):".into(),
        ];
        for k in [
            "bos_token_id", "eos_token_id", "eot_token_id", "eom_token_id", "padding_token_id", "unknown_token_id",
            "add_bos_token", "add_eos_token",
        ] {
            if let Some(v) = self.kv.get(&format!("tokenizer.ggml.{k}")) {
                lines.push(format!("  tokenizer.ggml.{k} = {}", v.display()));
            }
        }
        if let Some(tpl) = self.chat_template() {
            static RM: OnceLock<Regex> = OnceLock::new();
            let rm = RM.get_or_init(|| Regex::new(r"<\|[a-z_]+\|>|</?s>|<end_of_turn>|<eos>").unwrap());
            let mut ends: Vec<String> = rm.find_iter(&tpl).map(|m| m.as_str().to_string()).collect();
            ends.sort();
            ends.dedup();
            lines.push(format!(
                "  special markers in chat template: {}",
                if ends.is_empty() { "(none found)".into() } else { ends.join(", ") }
            ));
        }
        lines.push(String::new());
        lines.push("Raw metadata (tokenizer omitted):".into());
        let mut keys: Vec<&String> = self.kv.keys().filter(|k| !k.starts_with("tokenizer.")).collect();
        keys.sort();
        for k in keys {
            lines.push(format!("  {k} = {}", self.kv[k].display()));
        }
        lines.join("\n")
    }
}
