//! vLLM backend: model folders, estimates, planning, metrics, environment.

use regex::Regex;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use super::consts::vllm_online_quant;
use super::devices::Device;
use super::util::{expanduser, fmt_thousands, gf, gi, gs, py_round, run_capture, shlex_split, tail_chars, truthy, v_f64, v_i64, Opts, GIB, MIB};

pub const VLLM_DEFAULT_BATCHED: i64 = 2048;
pub const VLLM_NONTORCH: f64 = 1.6 * GIB;
pub const VLLM_MM_ACT: f64 = 2.0 * GIB;
pub const VLLM_HEURISTIC_VERSION: i64 = 2;

fn mm_name() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"vision|visual|image|audio|mm_proj|multi_modal|projector|siglip|perceiver|vit\.").unwrap())
}
fn layer_name() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.").unwrap())
}

/// {tensor name: bytes} from a .safetensors header (never reads the weights).
pub fn safetensors_sizes(path: &Path) -> Result<HashMap<String, f64>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    let n = u64::from_le_bytes(b) as usize;
    if n > 512 * 1024 * 1024 {
        return Err("header too large".into());
    }
    let mut hb = vec![0u8; n];
    f.read_exact(&mut hb).map_err(|e| e.to_string())?;
    let hdr: Map<String, Value> = serde_json::from_slice(&hb).map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for (name, meta) in hdr {
        if name == "__metadata__" {
            continue;
        }
        let Some(m) = meta.as_object() else { continue };
        let off = m.get("data_offsets").and_then(|v| v.as_array());
        let (a, b) = match off {
            Some(o) if o.len() >= 2 => (v_i64(o.get(0)).unwrap_or(0), v_i64(o.get(1)).unwrap_or(0)),
            _ => (0, 0),
        };
        out.insert(name, (b - a).max(0) as f64);
    }
    Ok(out)
}

fn has_glob_ext(d: &Path, ext: &str) -> bool {
    std::fs::read_dir(d)
        .map(|r| r.filter_map(|e| e.ok()).any(|e| e.path().extension().map(|x| x == ext).unwrap_or(false)))
        .unwrap_or(false)
}

fn files_ext(d: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .map(|r| {
            r.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == ext).unwrap_or(false))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub fn is_vllm_model_dir(d: &Path) -> bool {
    d.join("config.json").is_file() && (has_glob_ext(d, "safetensors") || has_glob_ext(d, "bin"))
}

/// Model folders (relative paths) under root; hidden folders (caches) are skipped.
pub fn scan_vllm_models(root: &str) -> Vec<String> {
    let root = PathBuf::from(root);
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    fn walk(d: &Path, root: &Path, out: &mut Vec<String>, depth: usize) {
        if depth > 12 {
            return;
        }
        if d.join("config.json").is_file() {
            let rel = d.strip_prefix(root).unwrap_or(d);
            if is_vllm_model_dir(d) {
                let r = rel.to_string_lossy().into_owned();
                out.push(if r.is_empty() || r == "." { d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default() } else { r });
            }
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.filter_map(|e| e.ok()) {
                let p = e.path();
                let hidden = p.file_name().map(|n| n.to_string_lossy().starts_with('.')).unwrap_or(false);
                if !hidden && e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    walk(&p, root, out, depth + 1);
                }
            }
        }
    }
    walk(&root, &root, &mut out, 0);
    out.sort_by_key(|a| a.to_lowercase());
    out.dedup();
    out
}

pub fn read_chat_template(d: &Path) -> Option<String> {
    let f = d.join("chat_template.jinja");
    if f.is_file() {
        return std::fs::read(&f).ok().map(|b| String::from_utf8_lossy(&b).into_owned());
    }
    for name in ["chat_template.json", "tokenizer_config.json"] {
        let f = d.join(name);
        if !f.is_file() {
            continue;
        }
        let Some(v) = std::fs::read(&f).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else { continue };
        match v.get("chat_template") {
            Some(Value::String(s)) => return Some(s.clone()),
            Some(Value::Array(list)) => {
                for item in list {
                    if item.get("name").and_then(|n| n.as_str()) == Some("default") {
                        return item.get("template").and_then(|t| t.as_str()).map(String::from);
                    }
                }
                if let Some(first) = list.first() {
                    return first.get("template").and_then(|t| t.as_str()).map(String::from);
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct VllmModelInfo {
    pub path: PathBuf,
    pub cfg: Value,
    pub arch: String,
    pub multimodal: bool,
    pub n_layer: usize,
    pub n_embd: f64,
    pub n_head: f64,
    pub n_kv_head: f64,
    pub head_dim: f64,
    pub n_ctx_train: i64,
    pub n_vocab: f64,
    pub n_expert: i64,
    pub tied: bool,
    pub nextn: i64,
    pub quant: String,
    pub dtype: String,
    pub mla: f64,
    pub mla_rope: f64,
    pub window: f64,
    pub kinds: Vec<&'static str>,
    pub files: Vec<PathBuf>,
    pub layer_bytes: Vec<f64>,
    pub expert_bytes: Vec<f64>,
    pub mm_bytes: f64,
    pub mtp_bytes: f64,
    pub embed_bytes: f64,
    pub head_bytes: f64,
    pub other_bytes: f64,
    pub head_shared: bool,
    pub total_bytes: f64,
    pub chat_template: Option<String>,
}

impl VllmModelInfo {
    pub fn load(path: &Path) -> Result<Self, String> {
        let txt = std::fs::read(path.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
        let cfg: Value = serde_json::from_slice(&txt).map_err(|e| format!("config.json: {e}"))?;
        let co = cfg.as_object().cloned().unwrap_or_default();
        let mut tc = co.clone();
        for k in ["text_config", "llm_config", "language_config", "thinker_config"] {
            if let Some(Value::Object(sub)) = co.get(k) {
                for (kk, vv) in sub {
                    tc.insert(kk.clone(), vv.clone());
                }
                break;
            }
        }
        // g(): tc.get(key), default if None (JSON null counts as None)
        let g = |key: &str| -> Option<&Value> { tc.get(key).filter(|v| !v.is_null()) };
        let gi_ = |key: &str, d: i64| -> i64 { v_i64(g(key)).unwrap_or(d) };
        let gor = |keys: &[&str]| -> i64 {
            for k in keys {
                if truthy(g(k)) {
                    return v_i64(g(k)).unwrap_or(0);
                }
            }
            0
        };

        let archs: Vec<String> = co
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let arch = archs.first().cloned().unwrap_or_else(|| {
            co.get("model_type").map(super::util::py_str).unwrap_or_else(|| "unknown".into())
        });
        let multimodal = ["vision_config", "audio_config", "vision_tower", "visual"]
            .iter()
            .any(|k| matches!(co.get(*k), Some(Value::Object(_))));
        let n_layer = {
            let d = gi_("n_layer", 0);
            gi_("num_hidden_layers", d).max(0) as usize
        };
        let n_embd = {
            let d = gi_("d_model", 0);
            gi_("hidden_size", d) as f64
        };
        let mut n_head = gi_("num_attention_heads", 0) as f64;
        if n_head == 0.0 {
            n_head = 1.0;
        }
        let mut n_kv_head = gi_("num_key_value_heads", n_head as i64) as f64;
        if n_kv_head == 0.0 {
            n_kv_head = n_head;
        }
        let hd = gi_("head_dim", 0);
        let head_dim = if hd != 0 { hd as f64 } else { (n_embd / n_head).floor() };
        let n_ctx_train = gi_("max_position_embeddings", 0);
        let nv = gi_("vocab_size", 32000);
        let n_vocab = if nv == 0 { 32000.0 } else { nv as f64 };
        let n_expert = gor(&["num_experts", "num_local_experts", "n_routed_experts"]);
        let tied = truthy(match tc.get("tie_word_embeddings") {
            Some(v) => Some(v),
            None => co.get("tie_word_embeddings"),
        });
        let nextn = gor(&["num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers"]);
        let q = [co.get("quantization_config"), tc.get("quantization_config")]
            .into_iter()
            .flatten()
            .find(|v| truthy(Some(v)))
            .cloned()
            .unwrap_or(Value::Null);
        let quant = q.get("quant_method").filter(|v| truthy(Some(v))).map(super::util::py_str).unwrap_or_default();
        let dtype = [g("torch_dtype"), g("dtype")]
            .into_iter()
            .flatten()
            .find(|v| truthy(Some(v)))
            .map(super::util::py_str)
            .unwrap_or_default();
        let mla = gi_("kv_lora_rank", 0) as f64;
        let mr = gi_("qk_rope_head_dim", 64);
        let mla_rope = if mr == 0 { 64.0 } else { mr as f64 };
        let mut window = gi_("sliding_window", 0) as f64;
        if tc.get("use_sliding_window") == Some(&Value::Bool(false)) {
            window = 0.0;
        }

        // attention kind per layer: full / swa / none (linear attention, mamba...)
        let mut kinds: Vec<&'static str> = vec!["full"; n_layer];
        let lt = tc.get("layer_types").and_then(|v| v.as_array()).filter(|a| !a.is_empty());
        if let Some(lt) = lt {
            kinds.clear();
            for i in 0..n_layer {
                let t = lt.get(i).map(super::util::py_str).unwrap_or_else(|| "full_attention".into()).to_lowercase();
                if ["linear", "mamba", "conv", "recurrent", "deltanet"].iter().any(|x| t.contains(x)) {
                    kinds.push("none");
                } else if ["sliding", "local", "chunk"].iter().any(|x| t.contains(x)) {
                    kinds.push(if window != 0.0 { "swa" } else { "full" });
                } else {
                    kinds.push("full");
                }
            }
        } else if gi_("full_attention_interval", 0) != 0 {
            let f = gi_("full_attention_interval", 1);
            let fai = if f == 0 { 1 } else { f };
            kinds = (0..n_layer).map(|i| if ((i as i64) + 1) % fai == 0 { "full" } else { "none" }).collect();
        } else if window != 0.0 && gi_("sliding_window_pattern", 0) != 0 {
            let per = gi_("sliding_window_pattern", 0);
            kinds = (0..n_layer).map(|i| if ((i as i64) + 1) % per == 0 { "full" } else { "swa" }).collect();
        }
        if !kinds.contains(&"swa") {
            window = 0.0;
        }

        // weights, from the safetensors headers
        let files = files_ext(path, "safetensors");
        let mut sizes: Vec<(String, f64)> = Vec::new();
        for p in &files {
            if let Ok(s) = safetensors_sizes(p) {
                sizes.extend(s);
            }
        }
        if sizes.is_empty() {
            // .bin checkpoints or unreadable headers: spread file sizes
            let sum = |v: Vec<PathBuf>| v.iter().map(|p| std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0)).sum::<f64>();
            let mut tot = sum(files_ext(path, "bin"));
            if tot == 0.0 {
                tot = sum(files.clone());
            }
            for i in 0..n_layer {
                sizes.push((format!("model.layers.{i}.x"), tot / n_layer.max(1) as f64));
            }
        }
        let n = n_layer;
        let mut layer_bytes = vec![0.0; n];
        let mut expert_bytes = vec![0.0; n];
        let (mut mm_bytes, mut mtp_bytes, mut embed_bytes, mut head_bytes, mut other_bytes) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (name, sz) in &sizes {
            let ln = name.to_lowercase();
            if mm_name().is_match(&ln) {
                mm_bytes += sz;
                continue;
            }
            if ln.starts_with("mtp.") || ln.contains(".mtp.") {
                mtp_bytes += sz;
                continue;
            }
            if let Some(m) = layer_name().captures(&ln) {
                let i: usize = m[1].parse().unwrap_or(usize::MAX);
                if i < n {
                    layer_bytes[i] += sz;
                    if ln.contains("expert") {
                        expert_bytes[i] += sz;
                    }
                } else {
                    mtp_bytes += sz; // DeepSeek/GLM style: layer n.. = MTP
                }
                continue;
            }
            if ln.contains("embed_tokens") || ln.ends_with("wte.weight") || ln.contains("word_embeddings") {
                embed_bytes += sz;
            } else if ln.contains("lm_head") {
                head_bytes += sz;
            } else {
                other_bytes += sz;
            }
        }
        let head_shared = tied && head_bytes == 0.0;
        if head_shared {
            head_bytes = embed_bytes; // a later pipeline stage needs its own copy
        }
        let total_bytes = sizes.iter().map(|x| x.1).sum();
        Ok(VllmModelInfo {
            path: path.to_path_buf(),
            cfg,
            arch,
            multimodal,
            n_layer,
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            n_ctx_train,
            n_vocab,
            n_expert,
            tied,
            nextn,
            quant,
            dtype,
            mla,
            mla_rope,
            window,
            kinds,
            files,
            layer_bytes,
            expert_bytes,
            mm_bytes,
            mtp_bytes,
            embed_bytes,
            head_bytes,
            other_bytes,
            head_shared,
            total_bytes,
            chat_template: read_chat_template(path),
        })
    }

    /// KV bytes per token for layer i.
    pub fn kv_tok(&self, i: usize, eb: f64) -> f64 {
        let k = self.kinds.get(i).copied().unwrap_or("full");
        if k == "none" {
            return 0.0;
        }
        if self.mla != 0.0 {
            return (self.mla + self.mla_rope) * eb;
        }
        2.0 * self.n_kv_head * self.head_dim * eb
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![self.arch.clone(), format!("{} layers", self.n_layer), format!("{:.2} GiB", self.total_bytes / GIB)];
        if !self.quant.is_empty() {
            parts.push(self.quant.clone());
        } else if !self.dtype.is_empty() {
            parts.push(self.dtype.replace("torch.", ""));
        }
        if self.n_expert != 0 {
            parts.push(format!("MoE {} experts", self.n_expert));
        }
        if self.n_ctx_train != 0 {
            parts.push(format!("train ctx {}", fmt_thousands(self.n_ctx_train)));
        }
        let n_none = self.kinds.iter().filter(|k| **k == "none").count();
        if n_none != 0 {
            parts.push(format!("hybrid: {} attention / {} linear layers", self.n_layer - n_none, n_none));
        }
        if self.window != 0.0 {
            parts.push(format!(
                "SWA {} ({} global layers)",
                fmt_thousands(self.window as i64),
                self.kinds.iter().filter(|k| **k == "full").count()
            ));
        }
        if self.nextn != 0 {
            parts.push(format!("MTP ×{}", self.nextn));
        }
        if self.multimodal {
            parts.push(format!("vision {:.2} GiB", self.mm_bytes / GIB));
        }
        parts.join("  •  ")
    }

    pub fn metadata_text(&self) -> String {
        let cnt = |k: &str| self.kinds.iter().filter(|x| **x == k).count();
        let kvsum: f64 = (0..self.n_layer).map(|i| self.kv_tok(i, 2.0)).sum();
        let cfg_txt = serde_json::to_string_pretty(&self.cfg).unwrap_or_default();
        let lines = [
            format!("Folder: {}", self.path.display()),
            format!(
                "Weights: {}",
                if self.files.is_empty() {
                    "(no safetensors)".to_string()
                } else {
                    self.files.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>().join(", ")
                }
            ),
            String::new(),
            "Derived:".into(),
            format!(
                "  arch={} layers={} hidden={} heads={} kv_heads={} head_dim={}",
                self.arch, self.n_layer, self.n_embd as i64, self.n_head as i64, self.n_kv_head as i64, self.head_dim as i64
            ),
            format!(
                "  layer kinds: {} full, {} sliding (window {}), {} linear/recurrent",
                cnt("full"), cnt("swa"), self.window as i64, cnt("none")
            ),
            format!("  MLA kv_lora_rank={}  MTP layers={}  experts={}", self.mla as i64, self.nextn, self.n_expert),
            format!("  KV per token (bf16, all layers) = {:.1} KiB", kvsum / 1024.0),
            format!(
                "  weights: layers {:.2} GiB, embed {:.2}, lm_head {:.2}{}, vision {:.2}, MTP {:.2}, other {:.2}",
                self.layer_bytes.iter().sum::<f64>() / GIB,
                self.embed_bytes / GIB,
                self.head_bytes / GIB,
                if self.head_shared { " (tied)" } else { "" },
                self.mm_bytes / GIB,
                self.mtp_bytes / GIB,
                self.other_bytes / GIB
            ),
            format!(
                "  quantization={} dtype={} vocab={}",
                if self.quant.is_empty() { "-" } else { &self.quant },
                if self.dtype.is_empty() { "-" } else { &self.dtype },
                self.n_vocab as i64
            ),
            format!("  chat template: {}", if self.chat_template.is_some() { "yes" } else { "none found" }),
            String::new(),
            "config.json:".into(),
            cfg_txt.chars().take(20000).collect(),
        ];
        lines.join("\n")
    }
}

pub fn vllm_batched(s: &Opts) -> i64 {
    let b = gi(s, "max_batched", 0);
    if b > 0 { b } else { VLLM_DEFAULT_BATCHED }
}

/// Runtime + peak activation memory for one GPU.
pub fn vllm_heuristic_extra(info: &VllmModelInfo, s: &Opts, last: bool, first: bool, tp: f64, version: i64) -> f64 {
    let t = vllm_batched(s) as f64;
    let mut act;
    if version == 1 {
        act = t * info.n_embd * 350.0 + if last { t * info.n_vocab * 8.0 } else { 0.0 };
    } else {
        let seqs = (gi(s, "parallel", 1).max(1) as f64).min(t).max(1.0);
        act = t * info.n_embd * 180.0;
        if last {
            act += seqs * info.n_vocab * 107.0; // logits + sampler buffers per sequence
        }
    }
    if first && info.multimodal && !super::util::gb(s, "text_only", false) {
        act += VLLM_MM_ACT;
    }
    VLLM_NONTORCH + act / tp
}

#[derive(Clone, Debug)]
pub struct VCard {
    pub weights: f64,
    pub offload: f64,
    pub fixed: f64,
    pub budget: f64,
    pub total: f64,
    pub kv: f64,
    pub tok_full: f64,
    pub tok_swa: f64,
    pub layers: Option<(usize, usize)>,
}

/// Estimate for one GPU: weights, fixed memory, KV pool and KV bytes per token.
#[allow(clippy::too_many_arguments)]
pub fn vllm_card(info: &VllmModelInfo, s: &Opts, total: f64, layers: Vec<usize>, first: bool, last: bool, tp: f64, delta: f64, factor: f64) -> VCard {
    let eb = if gs(s, "kv_dtype", "auto") == "fp8" { 1.0 } else { 2.0 };
    let qf = if !info.quant.is_empty() {
        1.0
    } else {
        let q = gs(s, "quant", "(none)");
        vllm_online_quant(if q.is_empty() { "(none)" } else { &q }).unwrap_or(1.0)
    };
    let mut w: f64 = layers.iter().map(|&i| info.layer_bytes[i]).sum::<f64>() * qf;
    if first {
        w += info.embed_bytes + info.other_bytes;
        if info.multimodal && !super::util::gb(s, "text_only", false) {
            w += info.mm_bytes;
        }
    }
    if last && !(first && info.head_shared) {
        w += info.head_bytes;
    }
    let mtp = gs(s, "mtp", "") == "(built-in)" && info.nextn > 0;
    if last && mtp {
        w += info.mtp_bytes * qf;
    }
    w /= tp;
    let off = w.min(gf(s, "cpu_offload", 0.0).max(0.0) * GIB);
    w -= off;
    let fixed = (0.5 * GIB).max(vllm_heuristic_extra(info, s, last, first, tp, VLLM_HEURISTIC_VERSION) + delta);
    let budget = gf(s, "gpu_util", 0.9) * total;
    let mut tok_full: f64 = layers.iter().filter(|&&i| info.kinds[i] == "full").map(|&i| info.kv_tok(i, eb)).sum();
    let tok_swa: f64 = layers.iter().filter(|&&i| info.kinds[i] == "swa").map(|&i| info.kv_tok(i, eb)).sum();
    if last && mtp {
        tok_full += info.nextn as f64
            * if info.mla == 0.0 { 2.0 * info.n_kv_head * info.head_dim * eb } else { (info.mla + info.mla_rope) * eb };
    }
    VCard {
        weights: w,
        offload: off,
        fixed,
        budget,
        total,
        kv: budget - w - fixed,
        tok_full: tok_full * factor / tp,
        tok_swa: tok_swa * factor / tp,
        layers: if layers.is_empty() { None } else { Some((layers[0], *layers.last().unwrap())) },
    }
}

/// KV bytes one request of ctx tokens needs on this card.
pub fn vllm_kv_need(card: &VCard, info: &VllmModelInfo, ctx: f64) -> f64 {
    let w = if info.window != 0.0 { info.window } else { ctx };
    card.tok_full * ctx + card.tok_swa * w.min(ctx)
}

/// Largest context whose KV fits in this card's pool (one request).
pub fn vllm_max_ctx(card: &VCard, info: &VllmModelInfo) -> i64 {
    let (kv, a, b, w) = (card.kv, card.tok_full, card.tok_swa, info.window);
    if kv <= 0.0 {
        return 0;
    }
    if a + b <= 0.0 {
        return 1_000_000_000;
    }
    if b == 0.0 || w == 0.0 {
        return (kv / (a + b)) as i64;
    }
    let lo = kv / (a + b);
    if lo <= w {
        return lo as i64;
    }
    if a > 0.0 { ((kv - b * w) / a) as i64 } else { 1_000_000_000 }
}

/// Tokens of KV the pool holds, counted the way vLLM reports it (at this context).
pub fn vllm_capacity(card: &VCard, info: &VllmModelInfo, ctx: f64) -> i64 {
    let per = vllm_kv_need(card, info, ctx) / ctx.max(1.0);
    if per > 0.0 && card.kv > 0.0 { (card.kv / per) as i64 } else { 0 }
}

#[derive(Clone, Debug)]
pub struct VPlan {
    pub mode: String,
    pub order: Vec<Device>,
    pub cards: Vec<VCard>,
    pub k: usize,
    pub max_ctx: i64,
    pub fits: bool,
    pub note: String,
}

#[derive(Clone, Debug, Default)]
pub struct VAdjust {
    pub delta: HashMap<String, f64>,
    pub factor: f64,
    pub present: bool,
}

/// Place a model on one or two GPUs for vLLM.
pub fn plan_vllm(
    info: &VllmModelInfo,
    s: &Opts,
    devs: &[Device],
    totals: &HashMap<String, f64>,
    strategy: &str,
    manual: Option<&super::estimate::Manual>,
    adjust: &VAdjust,
) -> VPlan {
    let factor = if adjust.present { adjust.factor } else { 1.0 };
    let deltas = &adjust.delta;
    let n = info.n_layer;
    let ctx = gi(s, "ctx", 0);
    let tot = |d: &Device| totals.get(&d.id).cloned().unwrap_or(0.0);
    let dl = |d: &Device| deltas.get(&d.id).cloned().unwrap_or(0.0);

    let finish = |mode: &str, order: Vec<Device>, cards: Vec<VCard>, k: usize| -> VPlan {
        let max_ctx = cards.iter().map(|c| vllm_max_ctx(c, info)).min().unwrap_or(0);
        let fits = cards.iter().all(|c| c.kv > 0.0) && max_ctx >= ctx.max(1);
        VPlan { mode: mode.into(), order, cards, k, max_ctx, fits, note: String::new() }
    };
    let single = |d: &Device| -> VPlan {
        let c = vllm_card(info, s, tot(d), (0..n).collect(), true, true, 1.0, dl(d), factor);
        finish("single", vec![d.clone()], vec![c], n)
    };
    let pipe = |order: &[Device], k: usize| -> VPlan {
        let a = vllm_card(info, s, tot(&order[0]), (0..k).collect(), true, false, 1.0, dl(&order[0]), factor);
        let b = vllm_card(info, s, tot(&order[1]), (k..n).collect(), false, true, 1.0, dl(&order[1]), factor);
        finish("pipeline", order.to_vec(), vec![a, b], k)
    };
    let tensor = |order: &[Device]| -> VPlan {
        let cs: Vec<VCard> =
            order.iter().map(|d| vllm_card(info, s, tot(d), (0..n).collect(), true, true, 2.0, dl(d), factor)).collect();
        let mut p = finish("tensor", order.to_vec(), cs, 0);
        if (info.n_kv_head as i64) % 2 != 0 {
            p.fits = false;
            p.note = format!("tensor split needs an even KV head count (this model has {})", info.n_kv_head as i64);
        }
        p
    };

    if devs.len() == 1 {
        return single(&devs[0]);
    }
    let mut sorted = devs.to_vec();
    sorted.sort_by_key(|d| (d.total_mib, d.index));
    let (small, big) = (sorted[0].clone(), sorted[1].clone());
    if let (Some(m), "Manual") = (manual, strategy) {
        let mode = if m.mode.is_empty() { "pipeline" } else { m.mode.as_str() };
        let order = if m.first == big.id { vec![big.clone(), small.clone()] } else { vec![small.clone(), big.clone()] };
        if mode == "tensor" {
            return tensor(&order);
        }
        let parts: Vec<&str> = m.ts.split(',').take(2).collect();
        let parsed = if parts.len() == 2 {
            match (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                (Ok(a), Ok(b)) => Some((a, b)),
                _ => None,
            }
        } else {
            None
        };
        let (ra, rb) = parsed.unwrap_or((tot(&order[0]), tot(&order[1])));
        let k = if ra + rb > 0.0 {
            py_round(n as f64 * ra / (ra + rb)).min(n as i64 - 1).max(1) as usize
        } else {
            n / 2
        };
        let mut p = pipe(&order, k);
        p.note = "manual pipeline split".into();
        return p;
    }
    if strategy == "Auto Speed" {
        let mut p = single(&big);
        if p.fits {
            p.note = format!("fits on {} alone - one GPU is fastest", big.name);
            return p;
        }
    }
    // most context; if nothing fits, the smallest shortfall
    let score = |p: &VPlan| (p.max_ctx, p.cards.iter().map(|c| c.kv).fold(f64::INFINITY, f64::min));
    let gt = |a: (i64, f64), b: (i64, f64)| a.0 > b.0 || (a.0 == b.0 && a.1 > b.1);
    let mut best: Option<VPlan> = None;
    for order in [vec![small.clone(), big.clone()], vec![big.clone(), small.clone()]] {
        for k in 1..n {
            let p = pipe(&order, k);
            if best.as_ref().map(|b| gt(score(&p), score(b))).unwrap_or(true) {
                best = Some(p);
            }
        }
    }
    let mut sp = single(&big);
    let Some(mut best) = best else {
        sp.note = format!("{} alone gives as much context as a split", big.name);
        return sp;
    };
    if sp.max_ctx > 0 && !gt(score(&best), score(&sp)) {
        sp.note = format!("{} alone gives as much context as a split", big.name);
        return sp;
    }
    if best.max_ctx <= 0 {
        let short = -best.cards.iter().map(|c| c.kv).fold(f64::INFINITY, f64::min);
        best.note = format!("doesn't fit even split across both cards (short ≈ {:.1} GiB)", short / GIB);
    } else {
        best.note = "pipeline split with the most context".into();
    }
    best
}

pub fn parse_vllm_metrics(text: &str) -> HashMap<String, f64> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"(?m)^vllm:([A-Za-z_:]+)(?:\{[^}]*\})?\s+(\S+)").unwrap());
    let mut out: HashMap<String, f64> = HashMap::new();
    for m in re.captures_iter(text) {
        let k = &m[1];
        if k.ends_with("_bucket") || k.ends_with("_created") {
            continue;
        }
        let Some(f) = py_parse_float(&m[2]) else { continue };
        if f.is_nan() {
            continue;
        }
        *out.entry(k.to_string()).or_insert(0.0) += f;
    }
    out
}

/// Python `float()` for metric values (accepts +Inf, nan, 1e-3, ...).
pub fn py_parse_float(s: &str) -> Option<f64> {
    let l = s.to_lowercase();
    match l.as_str() {
        "+inf" | "inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => s.parse::<f64>().ok(),
    }
}

pub struct VRe {
    pub tag: Regex,
    pub kvmem: Regex,
    pub kvtok: Regex,
    pub conc: Regex,
    pub load: Regex,
    pub graph: Regex,
    pub ready: Regex,
    pub poll: Regex,
}

pub fn vre() -> &'static VRe {
    static R: OnceLock<VRe> = OnceLock::new();
    R.get_or_init(|| VRe {
        tag: Regex::new(r"^\(([^)]*?)\s+pid=\d+\)").unwrap(),
        kvmem: Regex::new(r"Available KV cache memory:\s*([\d.]+)\s*GiB").unwrap(),
        kvtok: Regex::new(r"GPU KV cache size:\s*([\d,]+)\s*tokens").unwrap(),
        conc: Regex::new(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x").unwrap(),
        load: Regex::new(r"Model loading took\s*([\d.]+)\s*GiB").unwrap(),
        graph: Regex::new(r"Graph capturing finished in .*?took\s*([\d.]+)\s*GiB").unwrap(),
        ready: Regex::new(r"Application startup complete").unwrap(),
        poll: Regex::new(r#""GET /(?:health|metrics) HTTP/[\d.]+" 200"#).unwrap(),
    })
}

/// Card index (in launch order) of the vLLM process that printed a log line.
pub fn vllm_rank(tag: &str) -> usize {
    for p in [r"PP(\d+)", r"TP(\d+)", r"rank=(\d+)"] {
        if let Some(m) = Regex::new(p).unwrap().captures(tag) {
            return m[1].parse().unwrap_or(0);
        }
    }
    0
}

pub fn vllm_failure_hint(lines: &[String]) -> String {
    let text = lines.join("\n");
    let m = |p: &str| Regex::new(p).unwrap().is_match(&text);
    if let Some(c) = Regex::new(r"maximum number of tokens that can be stored in KV cache \(([\d,]+)\)").unwrap().captures(&text) {
        return format!(
            "Context is larger than the KV cache vLLM could make ({} tokens).\nLower Context (press Fit), raise GPU memory, use fp8 KV cache, or span both GPUs.",
            &c[1]
        );
    }
    if m(r"(?s)Free memory on device.*less than desired GPU memory utilization") {
        return "The card didn't have GPU memory × card size free when vLLM started.\nLower GPU memory, or close what else is using that card (desktop, browser, the other pane).".into();
    }
    if m(r"(?i)No available memory for the cache blocks|Insufficient memory|out of memory|OutOfMemoryError") {
        return "Weights + activations don't fit in the GPU memory budget.\nRaise GPU memory, lower Batch tokens, tick Text only, use eager mode, or span both GPUs.".into();
    }
    if m(r"(?i)Unknown quantization method|invalid choice.*quantization|quantization.*(?:not supported|unsupported)") {
        return "This vLLM build can't do that load-time quantization on this GPU.\nPick another 'Quantize at load' option, or make an FP8 copy with the Quantize tool.".into();
    }
    if text.contains("Address already in use") {
        return "That port is already in use - pick another Port.".into();
    }
    if m(r"VLLM_ALLOW_LONG_MAX_MODEL_LEN|greater than the derived max_model_len") {
        return "Context is above the model's trained length - lower it (or add RoPE scaling in Extra args).".into();
    }
    if m(r"reasoning.parser|ReasoningParser") && text.contains("Traceback") {
        return "The reasoning parser name may not exist in this vLLM build - try (none).".into();
    }
    String::new()
}

/// venv folder of the vllm executable (…/.venv/bin/vllm -> …/.venv), or None.
pub fn vllm_venv(vllm_bin: &str) -> Option<PathBuf> {
    let p = expanduser(vllm_bin);
    let v = p.parent()?.parent()?.to_path_buf();
    if v.join("bin").join("python").exists() || v.join("pyvenv.cfg").exists() {
        Some(v)
    } else {
        None
    }
}

/// Environment the venv's activate script would set (plus the user's extra vars).
pub fn vllm_base_env(g: &Opts) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = vec![("FLASH_ATTENTION_TRITON_AMD_ENABLE".into(), "TRUE".into())];
    let set = |env: &mut Vec<(String, String)>, k: String, v: String| {
        if let Some(e) = env.iter_mut().find(|e| e.0 == k) {
            e.1 = v;
        } else {
            env.push((k, v));
        }
    };
    if let Some(venv) = vllm_venv(&gs(g, "vllm_bin", "")) {
        let mut smi: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(venv.join("lib")) {
            for e in rd.filter_map(|e| e.ok()) {
                let nm = e.file_name().to_string_lossy().into_owned();
                if nm.starts_with("python3") {
                    let p = e.path().join("site-packages/_rocm_sdk_core/share/amd_smi");
                    if p.exists() {
                        smi.push(p);
                    }
                }
            }
        }
        smi.sort();
        if let Some(last) = smi.last() {
            set(&mut env, "PYTHONPATH".into(), last.to_string_lossy().into_owned());
        }
    }
    let models = gs(g, "vllm_models_dir", "");
    if let Ok(items) = shlex_split(&gs(g, "vllm_env", "")) {
        for item in items {
            if let Some((k, v)) = item.split_once('=') {
                set(&mut env, k.trim().to_string(), v.replace("{models}", &models));
            }
        }
    }
    env
}

/// Full process environment: current env + venv activation + display_env.
pub fn vllm_process_env(g: &Opts, display_env: &[(String, String)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = std::env::vars().collect();
    let set = |env: &mut Vec<(String, String)>, k: &str, v: String| {
        if let Some(e) = env.iter_mut().find(|e| e.0 == k) {
            e.1 = v;
        } else {
            env.push((k.to_string(), v));
        }
    };
    if let Some(venv) = vllm_venv(&gs(g, "vllm_bin", "")) {
        set(&mut env, "VIRTUAL_ENV", venv.to_string_lossy().into_owned());
        let path = std::env::var("PATH").unwrap_or_default();
        set(&mut env, "PATH", format!("{}:{}", venv.join("bin").display(), path));
        env.retain(|e| e.0 != "PYTHONHOME");
    }
    for (k, v) in display_env {
        let cur = env.iter().find(|e| &e.0 == k).map(|e| e.1.clone());
        match (k.as_str(), cur) {
            ("PYTHONPATH", Some(c)) if !c.is_empty() && !c.split(':').any(|x| x == v) => {
                set(&mut env, k, format!("{v}:{c}"));
            }
            _ => set(&mut env, k, v.clone()),
        }
    }
    env
}

/// GPU list from the vLLM venv's PyTorch (used when llama-server isn't available).
pub fn detect_devices_torch(g: &Opts) -> (Vec<Device>, String) {
    let Some(venv) = vllm_venv(&gs(g, "vllm_bin", "")) else {
        return (vec![], "vLLM venv not found (Settings → vLLM binary).".into());
    };
    let code = "import json,torch;print('DEVS'+json.dumps([[torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory] for i in range(torch.cuda.device_count())]))";
    let env = vllm_process_env(g, &vllm_base_env(g));
    let out = match run_capture(
        &[venv.join("bin/python").to_string_lossy().into_owned(), "-c".into(), code.into()],
        Duration::from_secs(120),
        Some(&env),
    ) {
        Ok(o) => o,
        Err(e) => return (vec![], format!("Could not ask PyTorch for GPUs:\n{e}")),
    };
    let re = Regex::new(r"DEVS(\[.*\])").unwrap();
    let Some(m) = re.captures(&out.stdout) else {
        return (vec![], format!("PyTorch in the vLLM venv returned no GPUs.\n\n{}", tail_chars(&out.stderr, 3000)));
    };
    let list: Vec<Value> = serde_json::from_str(&m[1]).unwrap_or_default();
    let devs = list
        .iter()
        .enumerate()
        .map(|(i, x)| Device {
            backend: "ROCm".into(),
            index: i as i64,
            id: format!("ROCm{i}"),
            name: x.get(0).and_then(|v| v.as_str()).unwrap_or("GPU").to_string(),
            total_mib: (v_f64(x.get(1)).unwrap_or(0.0) / MIB) as i64,
            sysfs: None,
        })
        .collect();
    (devs, String::new())
}
