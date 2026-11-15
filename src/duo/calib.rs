//! Calibration data for GPTQ recipes (`gptq … calib=$CALIB`): the Hessians of every linear's
//! input, recorded by Radiance's calibration mode and written as the RADGRAM2 files rad-convert's
//! quantisers read.
//!
//! The pipeline, run from the Convert window:
//!   1. corpus   a JSONL file ({"prompt": …} or {"text": …} a line), or a folder of text files
//!               made into one, documents cut to what the KV pool holds
//!   2. record   `radiance --config` on the bf16 checkpoint with `calib-corpus` / `calib-dir`,
//!               over every card chosen (tp), in layer slices when host RAM can't hold every
//!               layer's Hessians at once (docs/TOOLS.md, "The calibration mode")
//!   3. convert  each `<checkpoint stem>.safetensors` (the packed upper triangle) into
//!               `gram.r0.<container stem>.bin` (RADGRAM2: a 32-byte header and the whole k x k
//!               matrix), named by the container's weights; the weights that read one input get
//!               symlinks to one file
//!
//! Supported: dense Qwen3.5 / Qwen3.8 (`qwen35_bf16` records them). Other families record their
//! Hessians another way (Gemma 4: arch/gemma4's tap, docs/ARCHITECTURES.md "The 31B by GPTQ").

use serde_json::Value;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

use super::app::App;
use super::devices::Device;
use super::proc::{self, Proc, ProcMsg};
use super::radiance::{checkpoint_kind, uses_config};
use super::util::*;

/// The Create-calibration panel of the Convert window.
pub struct Calib {
    pub shown: bool,
    pub src: String,
    pub corpus: String,
    pub out: String,
    /// "all cards" or one card's "id: name"
    pub gpu: String,
    pub max_len: String,
    pub keep_raw: bool,
    pub status: String,
    pub stage: Stage,
    pub proc: Option<Proc>,
    slices: Vec<(i64, i64)>,
    slice: usize,
    user_stopped: bool,
    t0: Option<Instant>,
    /// the run's folders: raw (safetensors) and grams (RADGRAM2)
    raw: PathBuf,
    grams: PathBuf,
    corpus_file: PathBuf,
    tx: Sender<ProcMsg>,
    rx: Receiver<ProcMsg>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Idle,
    Corpus,
    Record,
    Convert,
}

impl Calib {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Calib {
            shown: false,
            src: String::new(),
            corpus: String::new(),
            out: String::new(),
            gpu: "all cards".into(),
            max_len: "16384".into(),
            keep_raw: true,
            status: String::new(),
            stage: Stage::Idle,
            proc: None,
            slices: vec![],
            slice: 0,
            user_stopped: false,
            t0: None,
            raw: PathBuf::new(),
            grams: PathBuf::new(),
            corpus_file: PathBuf::new(),
            tx,
            rx,
        }
    }

    pub fn busy(&self) -> bool {
        self.stage != Stage::Idle
    }
}

impl Default for Calib {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------- the model --
/// The text model's config (a VL release nests it under `text_config`).
fn text_config(dir: &Path) -> Value {
    let cfg: Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    match cfg.get("text_config") {
        Some(t) if t.is_object() => t.clone(),
        _ => cfg,
    }
}

fn has_vision(dir: &Path) -> bool {
    let cfg: Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    cfg.get("vision_config").map(|v| v.is_object()).unwrap_or(false)
}

/// Why LLM Duo can't record Hessians from this checkpoint, or None when it can.
pub fn unsupported(dir: &Path) -> Option<String> {
    if !dir.join("config.json").is_file() {
        return Some("pick a Hugging Face checkpoint folder (config.json + safetensors) as the calibration source".into());
    }
    let k = checkpoint_kind(dir);
    match k.family() {
        "qwen35" => {}
        "qwen35moe" => {
            return Some("a Qwen3.5/3.8 MoE records its experts' Hessians with Radiance's older tap at one card (docs/TOOLS.md, MOE-W4.md); LLM Duo only drives the dense models' calibration mode".into())
        }
        "gemma4" => {
            return Some("Gemma 4 records its Hessians with arch/gemma4's own tap, fed by requests (docs/ARCHITECTURES.md, \"The 31B by GPTQ\"); LLM Duo only drives the dense Qwen3.5/3.8 calibration mode".into())
        }
        _ => {
            return Some(format!(
                "Radiance's calibration mode records dense Qwen3.5/3.8 ({}); LLM Duo can't record Hessians for this model",
                if k.model_type.is_empty() { "this checkpoint's model_type is unknown".to_string() } else { format!("this is {}", k.model_type) }
            ))
        }
    }
    if k.quantised == Some(true) {
        return Some("the calibration source must be the bf16 checkpoint (this one is quantised); a finetune's bf16 release, or the base model's".into());
    }
    None
}

fn tri(k: f64) -> f64 {
    k * (k + 1.0) / 2.0
}

/// (layers, bytes of host memory the recorded Hessians of one layer take). Per layer the tap
/// keeps four inputs' upper triangles in f32: the block input and the MLP input (hidden wide),
/// the attention or delta net output (the wider of the two), and down_proj's (intermediate wide).
pub fn layer_bytes(dir: &Path) -> (i64, f64) {
    let t = text_config(dir);
    let n = |k: &str| t.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
    let layers = n("num_hidden_layers") as i64;
    let h = n("hidden_size");
    let i = n("intermediate_size");
    let attn = n("num_attention_heads") * if n("head_dim") > 0.0 { n("head_dim") } else { h / n("num_attention_heads").max(1.0) };
    let lin = n("linear_num_value_heads") * n("linear_value_head_dim");
    let o = attn.max(lin).max(h);
    (layers, 4.0 * (2.0 * tri(h) + tri(o) + tri(i)))
}

/// Layer ranges, each small enough that its Hessians fit `budget` bytes of host memory.
pub fn plan_slices(layers: i64, per_layer: f64, budget: f64) -> Vec<(i64, i64)> {
    if layers <= 0 {
        return vec![];
    }
    let per = if per_layer <= 0.0 { layers } else { ((budget / per_layer).floor() as i64).clamp(1, layers) };
    let n = (layers + per - 1) / per;
    // even slices: 64 layers at 40 a slice is 32 + 32, not 40 + 24
    let size = (layers + n - 1) / n;
    (0..n).map(|s| (s * size, ((s + 1) * size).min(layers))).filter(|(a, b)| a < b).collect()
}

// --------------------------------------------------------- the name map --
/// The container weights (stems, without `.weight`) whose GPTQ reads the Hessian of the input
/// these checkpoint linears share. Both the bf16 plugin's names and the quantised plugin's fused
/// ones are given, so a recipe for either finds its file.
pub fn container_stems(linears: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut push = |s: String| {
        if !out.contains(&s) {
            out.push(s);
        }
    };
    for lin in linears.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if lin == "lm_head" || lin.ends_with(".lm_head") {
            push("output".into());
            continue;
        }
        let Some(pos) = lin.find(".layers.") else { continue };
        let rest = &lin[pos + 8..];
        let Some((idx, suffix)) = rest.split_once('.') else { continue };
        let Ok(l) = idx.parse::<i64>() else { continue };
        let names: &[&str] = match suffix {
            "self_attn.q_proj" => &["attn_qg"],
            "self_attn.k_proj" => &["attn_k"],
            "self_attn.v_proj" => &["attn_v"],
            "self_attn.o_proj" => &["attn_output"],
            "linear_attn.in_proj_qkv" => &["ssm_inz", "ssm_in"],
            "linear_attn.in_proj_z" => &["ssm_inz", "ssm_z"],
            "linear_attn.in_proj_a" | "linear_attn.in_proj_b" => &["ssm_ab"],
            "linear_attn.out_proj" => &["ssm_out"],
            "mlp.gate_proj" | "mlp.up_proj" => &["ffn_gate_up"],
            "mlp.down_proj" => &["ffn_down"],
            _ => &[],
        };
        for n in names {
            push(format!("blk.{l}.{n}"));
        }
    }
    out
}

// ----------------------------------------------------------- converting --
/// One calibration file: its metadata and where its packed triangle starts.
struct RawGram {
    k: usize,
    rows: u64,
    domain: u32,
    linears: String,
    data_at: u64,
}

fn read_raw_header(f: &mut std::fs::File) -> Result<RawGram, String> {
    let mut n8 = [0u8; 8];
    f.read_exact(&mut n8).map_err(|e| e.to_string())?;
    let hl = u64::from_le_bytes(n8);
    if hl > 1 << 20 {
        return Err("not a safetensors file".into());
    }
    let mut h = vec![0u8; hl as usize];
    f.read_exact(&mut h).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_slice(&h).map_err(|e| format!("header: {e}"))?;
    let meta = v.get("__metadata__").cloned().unwrap_or(Value::Null);
    let ms = |k: &str| meta.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let k: usize = ms("k").parse().map_err(|_| "no k in the metadata")?;
    let t = v.get("hessian_upper").ok_or("no hessian_upper tensor")?;
    let start = t.get("data_offsets").and_then(|o| o.get(0)).and_then(|x| x.as_u64()).unwrap_or(0);
    let len = t.get("shape").and_then(|s| s.get(0)).and_then(|x| x.as_u64()).unwrap_or(0);
    if len as usize != k * (k + 1) / 2 {
        return Err(format!("hessian_upper holds {len} values, not the {} of a {k}-wide triangle", k * (k + 1) / 2));
    }
    Ok(RawGram { k, rows: ms("rows").parse().unwrap_or(0), domain: ms("domain").parse().unwrap_or(0), linears: ms("linears"), data_at: 8 + hl + start })
}

/// Write `gram.r0.<stem>.bin` for the first stem the file's linears map to, and symlinks for the
/// rest. Returns the stems written (empty: no container weight reads this input).
pub fn convert_one(src: &Path, out_dir: &Path) -> Result<Vec<String>, String> {
    let mut f = std::fs::File::open(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let g = read_raw_header(&mut f).map_err(|e| format!("{}: {e}", src.display()))?;
    let stems = container_stems(&g.linears);
    if stems.is_empty() {
        return Ok(vec![]);
    }
    let k = g.k;
    f.seek(SeekFrom::Start(g.data_at)).map_err(|e| e.to_string())?;
    let mut bytes = vec![0u8; k * (k + 1) / 2 * 4];
    f.read_exact(&mut bytes).map_err(|e| format!("{}: short ({e})", src.display()))?;
    let tri: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    drop(bytes);
    // (a, b), a <= b, in the packed upper triangle: the rows before a hold
    // sum over r < a of (k - r) = a*k - a*(a-1)/2 entries, then b - a into row a
    let at = |i: usize, j: usize| -> f32 {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        tri[a * k - a * a.saturating_sub(1) / 2 + (b - a)]
    };
    let first = out_dir.join(format!("gram.r0.{}.bin", stems[0]));
    let tmp = first.with_extension("bin.part");
    {
        let mut w = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?);
        let mut hdr = [0u8; 32];
        hdr[..8].copy_from_slice(b"RADGRAM2");
        hdr[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&0u32.to_le_bytes());
        hdr[16..24].copy_from_slice(&g.rows.to_le_bytes());
        hdr[24..28].copy_from_slice(&g.domain.to_le_bytes());
        w.write_all(&hdr).map_err(|e| e.to_string())?;
        let mut row = vec![0u8; k * 4];
        for i in 0..k {
            for j in 0..k {
                row[j * 4..j * 4 + 4].copy_from_slice(&at(i, j).to_le_bytes());
            }
            w.write_all(&row).map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, &first).map_err(|e| e.to_string())?;
    for s in &stems[1..] {
        let link = out_dir.join(format!("gram.r0.{s}.bin"));
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(first.file_name().unwrap(), &link).map_err(|e| format!("{}: {e}", link.display()))?;
    }
    Ok(stems)
}

/// What a calibration folder holds: (RADGRAM2 files, raw safetensors files not yet converted).
pub fn grams_in(dir: &Path) -> (usize, usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return (0, 0) };
    let (mut g, mut r) = (0, 0);
    for e in rd.filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().into_owned();
        if n.starts_with("gram.r") && n.ends_with(".bin") {
            g += 1;
        } else if n.ends_with(".input.safetensors") {
            r += 1;
        }
    }
    (g, r)
}

// --------------------------------------------------------------- corpus --
/// A JSONL corpus (each line `{"prompt": …}`), made from `src`: a .jsonl file whose lines carry
/// `prompt` or `text`, or a folder of .txt / .md / .jsonl files. Documents longer than
/// `max_chars` are cut into pieces. Returns (documents, characters).
pub fn build_corpus(src: &Path, dst: &Path, max_chars: usize) -> Result<(usize, usize), String> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(dst).map_err(|e| format!("{}: {e}", dst.display()))?);
    let (mut docs, mut chars) = (0usize, 0usize);
    let mut emit = |text: &str, w: &mut std::io::BufWriter<std::fs::File>| -> Result<(), String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        let cs: Vec<char> = text.chars().collect();
        for piece in cs.chunks(max_chars.max(256)) {
            let s: String = piece.iter().collect();
            if s.trim().len() < 16 {
                continue;
            }
            writeln!(w, "{}", serde_json::json!({ "prompt": s })).map_err(|e| e.to_string())?;
            docs += 1;
            chars += s.chars().count();
        }
        Ok(())
    };
    let mut files: Vec<PathBuf> = vec![];
    if src.is_dir() {
        let mut stack = vec![src.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?.filter_map(|e| e.ok()) {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if matches!(p.extension().and_then(|x| x.to_str()), Some("txt" | "md" | "jsonl")) {
                    files.push(p);
                }
            }
        }
        files.sort();
    } else if src.is_file() {
        files.push(src.to_path_buf());
    } else {
        return Err(format!("{} does not exist", src.display()));
    }
    for p in &files {
        if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
            let rd = std::io::BufReader::new(std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?);
            for (n, line) in rd.lines().enumerate() {
                let line = line.map_err(|e| format!("{}: {e}", p.display()))?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line).map_err(|e| format!("{} line {}: {e}", p.display(), n + 1))?;
                let t = v.get("prompt").or_else(|| v.get("text")).and_then(|x| x.as_str());
                match t {
                    Some(t) => emit(t, &mut w)?,
                    None => return Err(format!("{} line {}: no \"prompt\" or \"text\" string", p.display(), n + 1)),
                }
            }
        } else {
            let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            emit(&t, &mut w)?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    if docs == 0 {
        return Err("the corpus has no documents".into());
    }
    Ok((docs, chars))
}

// ------------------------------------------------------------------ app --
impl App {
    fn cal_devices(&self) -> Vec<Device> {
        match self.conv.cal.gpu.as_str() {
            "all cards" | "" => self.devices.clone(),
            other => {
                let id = other.split(':').next().unwrap_or("");
                self.devices.iter().filter(|d| d.id == id).cloned().collect()
            }
        }
    }

    pub fn cal_gpu_choices(&self) -> Vec<String> {
        let mut v = vec!["all cards".to_string()];
        v.extend(self.devices.iter().map(|d| format!("{}: {}", d.id, d.name)));
        v
    }

    fn cal_default_out(&self, src: &str) -> String {
        let name = Path::new(src).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        let root = gs(self.settings.g(), "radiance_models_dir", "");
        let root = if root.trim().is_empty() { dirs().data.join("calib") } else { PathBuf::from(root).join("calib") };
        root.join(name).to_string_lossy().into_owned()
    }

    /// Open the panel, with the Convert source as the checkpoint.
    pub fn cal_show(&mut self) {
        let c = &mut self.conv.cal;
        c.shown = true;
        if c.stage == Stage::Idle && c.src.is_empty() {
            c.src = self.conv.src.clone();
        }
        if self.conv.cal.out.is_empty() && !self.conv.cal.src.is_empty() {
            self.conv.cal.out = self.cal_default_out(&self.conv.cal.src.clone());
        }
    }

    pub fn cal_set_src(&mut self, v: &str) {
        let auto = self.conv.cal.out.is_empty() || self.conv.cal.out == self.cal_default_out(&self.conv.cal.src.clone());
        self.conv.cal.src = v.trim().trim_end_matches('/').to_string();
        if auto {
            self.conv.cal.out = self.cal_default_out(&self.conv.cal.src.clone());
        }
    }

    /// The panel's summary: what the run needs and will do.
    pub fn cal_info(&self) -> String {
        let c = &self.conv.cal;
        if c.src.is_empty() {
            return "Pick the bf16 checkpoint to record from (the model the recipe will quantise).".into();
        }
        let src = Path::new(&c.src);
        if let Some(why) = unsupported(src) {
            return format!("⚠ {why}");
        }
        let mut parts = vec![];
        let size = std::fs::read_dir(src)
            .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors")).map(|e| e.metadata().map(|m| m.len() as f64).unwrap_or(0.0)).sum::<f64>())
            .unwrap_or(0.0);
        let devs = self.cal_devices();
        let vram: f64 = devs.iter().map(|d| d.total_mib as f64 * MIB).sum();
        parts.push(format!("weights {:.1} GiB on {} card(s) with {:.0} GiB", size / GIB, devs.len(), vram / GIB));
        if size > vram * 0.9 {
            parts.push("⚠ the bf16 weights don't fit these cards: pick all cards (or more VRAM)".into());
        }
        let (layers, per) = layer_bytes(src);
        let (_, avail) = super::prefix::meminfo();
        let slices = plan_slices(layers, per, avail * 0.6);
        parts.push(format!(
            "{} layers, {:.1} GiB of Hessians in host RAM per layer → {} pass(es) over the corpus{}",
            layers,
            per / GIB,
            slices.len(),
            if slices.len() > 1 { format!(" ({} available RAM)", fmt_gib(avail)) } else { String::new() }
        ));
        let raw = per * layers as f64;
        let need = raw * if c.keep_raw { 3.0 } else { 2.0 };
        let out = Path::new(&c.out);
        let free = out.ancestors().find(|a| a.exists()).and_then(super::quantize::disk_free).unwrap_or(0.0);
        parts.push(format!("disk ≈ {:.0} GiB (free {:.0} GiB)", need / GIB, free / GIB));
        if free > 0.0 && free < need {
            parts.push("⚠ not enough free disk space in the output folder".into());
        }
        parts.join("   •   ")
    }

    fn cal_log(&mut self, line: &str) {
        self.conv_log_pub(line);
    }

    pub fn cal_start(&mut self) {
        if self.conv.cal.busy() || self.conv.proc.is_some() {
            self.msg("info", "A conversion or a calibration run is already going.");
            return;
        }
        let g = self.settings.g().clone();
        let bin = expanduser(&gs(&g, "radiance_bin", ""));
        if !bin.is_file() {
            self.msg("error", "Radiance binary not found. Set it in Settings.");
            return;
        }
        if !uses_config(&bin) {
            self.msg("error", "Recording Hessians needs Radiance 2.0 or later (its calibration mode is configured in a YAML file).");
            return;
        }
        let c = &self.conv.cal;
        let src = PathBuf::from(&c.src);
        if let Some(why) = unsupported(&src) {
            self.msg("warning", &format!("Can't record Hessians from this checkpoint:\n\n{why}"));
            return;
        }
        if c.corpus.trim().is_empty() {
            self.msg(
                "info",
                "Pick a corpus: a .jsonl file ({\"prompt\": …} or {\"text\": …} a line) or a folder of .txt / .md files.\n\nUse text like what the model will do (chat, code, tool calls, your language). Radiance's own calibrations used 1-2.5 million tokens.",
            );
            return;
        }
        if c.out.trim().is_empty() {
            self.msg("info", "Name an output folder.");
            return;
        }
        let max_len: i64 = match c.max_len.trim().replace(['_', ','], "").parse() {
            Ok(n) if n >= 1024 => n,
            _ => {
                self.msg("info", "Max tokens per document must be a number of at least 1024.");
                return;
            }
        };
        let devs = self.cal_devices();
        if devs.is_empty() {
            self.msg("error", "No GPU to record on.");
            return;
        }
        let out = expanduser(c.out.trim());
        let (raw, grams) = (out.join("raw"), out.join("grams"));
        for d in [&raw, &grams] {
            if let Err(e) = std::fs::create_dir_all(d) {
                self.msg("error", &format!("{}: {e}", d.display()));
                return;
            }
        }
        let (layers, per) = layer_bytes(&src);
        let (_, avail) = super::prefix::meminfo();
        let slices = plan_slices(layers, per, avail * 0.6);
        if slices.is_empty() {
            self.msg("error", "config.json names no num_hidden_layers.");
            return;
        }
        let corpus_src = expanduser(c.corpus.trim());
        let corpus_file = out.join("corpus.jsonl");
        {
            let c = &mut self.conv.cal;
            c.slices = slices;
            c.slice = 0;
            c.raw = raw;
            c.grams = grams;
            c.corpus_file = corpus_file.clone();
            c.user_stopped = false;
            c.t0 = Some(Instant::now());
            c.stage = Stage::Corpus;
            c.status = "Preparing the corpus…".into();
        }
        self.push_event(serde_json::json!({"type": "logClear", "side": "convert"}));
        self.cal_log(&format!("[LLM Duo] calibration: {} → {}", src.display(), out.display()));
        let tx = self.job_tx.clone();
        // about 3.5 characters a token; leave room for the template
        let max_chars = (max_len as usize).saturating_mul(3);
        std::thread::spawn(move || {
            let r = build_corpus(&corpus_src, &corpus_file, max_chars);
            let _ = tx.send(Box::new(move |app: &mut App| app.cal_corpus_done(r)));
        });
    }

    fn cal_corpus_done(&mut self, r: Result<(usize, usize), String>) {
        if self.conv.cal.stage != Stage::Corpus {
            return;
        }
        match r {
            Ok((docs, chars)) => {
                self.cal_log(&format!(
                    "[LLM Duo] corpus: {} documents, ≈ {} tokens → {}",
                    fmt_thousands(docs as i64),
                    fmt_thousands((chars / 4) as i64),
                    self.conv.cal.corpus_file.display()
                ));
                if chars / 4 < 200_000 {
                    self.cal_log("[LLM Duo] note: under ~200K tokens is a small calibration; Radiance's own used 1-2.5 million");
                }
                self.conv.cal.stage = Stage::Record;
                self.cal_next_slice();
            }
            Err(e) => self.cal_fail(&format!("corpus: {e}")),
        }
    }

    fn cal_yaml(&self, lo: i64, hi: i64) -> String {
        let c = &self.conv.cal;
        let q = |s: &str| serde_json::to_string(s).unwrap_or_default();
        let src = expanduser(c.src.trim());
        let max_len: i64 = c.max_len.trim().replace(['_', ','], "").parse().unwrap_or(16384);
        let mut y = String::new();
        y += "# written by LLM Duo: Radiance's calibration mode (docs/TOOLS.md)\n";
        y += "server:\n  host: 127.0.0.1\n  port: 18991\n";
        y += "models:\n  calib:\n";
        y += &format!("    model: {}\n", q(&src.to_string_lossy()));
        y += &format!("    tp: {}\n", self.cal_devices().len().max(1));
        y += &format!("    max-model-len: {max_len}\n");
        y += "    max-num-seqs: 8\n    kv-cache-dtype: bf16\n";
        if has_vision(&src) {
            y += "    mm-max-patches: 0\n"; // text only: the vision tower stays off the cards
        }
        y += &format!("    calib-corpus: {}\n", q(&c.corpus_file.to_string_lossy()));
        y += &format!("    calib-dir: {}\n", q(&c.raw.to_string_lossy()));
        y += &format!("    calib-layers: \"{lo}:{hi}\"\n");
        y
    }

    fn cal_next_slice(&mut self) {
        let i = self.conv.cal.slice;
        let Some(&(lo, hi)) = self.conv.cal.slices.get(i) else {
            self.cal_convert();
            return;
        };
        let n = self.conv.cal.slices.len();
        let g = self.settings.g().clone();
        let bin = expanduser(&gs(&g, "radiance_bin", "")).to_string_lossy().into_owned();
        let yaml = self.cal_yaml(lo, hi);
        let path = self.conv.cal.raw.parent().unwrap_or(Path::new(".")).join(format!("calib-{lo}-{hi}.yaml"));
        if let Err(e) = std::fs::write(&path, &yaml) {
            self.cal_fail(&format!("{}: {e}", path.display()));
            return;
        }
        let mut env: Vec<(String, String)> = std::env::vars().collect();
        let mut set = |k: &str, v: String| {
            env.retain(|e| e.0 != k);
            env.push((k.to_string(), v));
        };
        let home = gs(&g, "radiance_home", "");
        if !home.trim().is_empty() {
            set("RADIANCE_HOME", expanduser(home.trim()).to_string_lossy().into_owned());
        }
        set("HIP_VISIBLE_DEVICES", self.cal_devices().iter().map(|d| d.index.to_string()).collect::<Vec<_>>().join(","));
        let cmd = vec![bin, "--config".into(), path.to_string_lossy().into_owned()];
        self.cal_log(&format!("[LLM Duo] pass {}/{n}: layers {lo}..{}", i + 1, hi - 1));
        self.cal_log(&format!("$ {}", shlex_join(&cmd)));
        for l in yaml.lines() {
            self.cal_log(&format!("    {l}"));
        }
        while self.conv.cal.rx.try_recv().is_ok() {}
        match proc::spawn(&cmd, &env, false, self.conv.cal.tx.clone()) {
            Ok(pr) => self.conv.cal.proc = Some(pr),
            Err(e) => {
                self.cal_fail(&format!("could not start radiance: {e}"));
                return;
            }
        }
        self.conv.cal.status = format!("Recording pass {}/{n} (layers {lo}..{})…", i + 1, hi - 1);
    }

    pub fn cal_poll(&mut self) {
        for _ in 0..500 {
            let Ok(msg) = self.conv.cal.rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(l) => {
                    if l.contains("calib:") {
                        let n = self.conv.cal.slices.len();
                        let el = self.conv.cal.t0.map(|t| t.elapsed().as_secs_f64() / 60.0).unwrap_or(0.0);
                        self.conv.cal.status = format!("{el:.0} min  •  pass {}/{n}  •  {}", self.conv.cal.slice + 1, l.trim());
                    }
                    self.cal_log(&l);
                }
                ProcMsg::Exit(id, rc) => {
                    if self.conv.cal.proc.as_ref().map(|p| p.id) != Some(id) {
                        continue;
                    }
                    self.conv.cal.proc = None;
                    if self.conv.cal.user_stopped {
                        self.cal_fail("stopped");
                    } else if rc != 0 {
                        self.cal_fail(&format!("radiance exited with code {rc} during pass {} - see the log", self.conv.cal.slice + 1));
                    } else {
                        self.conv.cal.slice += 1;
                        self.cal_next_slice();
                    }
                }
            }
        }
    }

    fn cal_convert(&mut self) {
        self.conv.cal.stage = Stage::Convert;
        self.conv.cal.status = "Converting the Hessians for rad-convert…".into();
        let (raw, grams, keep) = (self.conv.cal.raw.clone(), self.conv.cal.grams.clone(), self.conv.cal.keep_raw);
        let mut files: Vec<PathBuf> = std::fs::read_dir(&raw)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors")).collect())
            .unwrap_or_default();
        files.sort();
        if files.is_empty() {
            self.cal_fail(&format!("radiance wrote no Hessians to {}", raw.display()));
            return;
        }
        self.cal_log(&format!("[LLM Duo] converting {} files to RADGRAM2 in {}", files.len(), grams.display()));
        let tx = self.job_tx.clone();
        std::thread::spawn(move || {
            let n = files.len();
            let mut written = 0usize;
            for (i, f) in files.iter().enumerate() {
                let r = convert_one(f, &grams);
                let name = f.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
                let msg = match &r {
                    Ok(s) if s.is_empty() => format!("[LLM Duo] {}/{n} {name}: no container weight reads it, skipped", i + 1),
                    Ok(s) => format!("[LLM Duo] {}/{n} {name} → {}", i + 1, s.join(", ")),
                    Err(e) => format!("[LLM Duo] {}/{n} {name}: {e}", i + 1),
                };
                if let Ok(s) = &r {
                    written += s.len();
                    if !keep && !s.is_empty() {
                        let _ = std::fs::remove_file(f);
                    }
                }
                let failed = r.is_err();
                let _ = tx.send(Box::new(move |app: &mut App| {
                    app.conv.cal.status = format!("Converting {}/{n}…", i + 1);
                    app.cal_log(&msg);
                }));
                if failed {
                    let _ = tx.send(Box::new(|app: &mut App| app.cal_fail("a file could not be converted - see the log")));
                    return;
                }
            }
            let _ = tx.send(Box::new(move |app: &mut App| app.cal_done(written)));
        });
    }

    fn cal_done(&mut self, written: usize) {
        if self.conv.cal.stage != Stage::Convert {
            return;
        }
        let mins = self.conv.cal.t0.map(|t| t.elapsed().as_secs_f64() / 60.0).unwrap_or(0.0);
        let grams = self.conv.cal.grams.to_string_lossy().into_owned();
        self.conv.cal.stage = Stage::Idle;
        self.conv.cal.status = format!("Done in {mins:.0} min - {written} Hessian files in {grams}");
        self.cal_log(&format!("[LLM Duo] done: {written} Hessian files. Calibration data is set to {grams}"));
        self.conv.calib = grams;
        self.conv_update_info();
        self.set_top_status("Calibration data is ready");
    }

    fn cal_fail(&mut self, why: &str) {
        let c = &mut self.conv.cal;
        c.stage = Stage::Idle;
        c.proc = None;
        c.status = if why == "stopped" { "Stopped".into() } else { format!("Failed: {why}") };
        self.cal_log(&format!("[LLM Duo] calibration {}", if why == "stopped" { "stopped".to_string() } else { format!("failed: {why}") }));
        self.set_top_status(format!("Calibration: {}", if why == "stopped" { "stopped" } else { "failed" }));
    }

    pub fn cal_stop(&mut self) {
        let c = &mut self.conv.cal;
        match c.stage {
            Stage::Record => {
                c.user_stopped = true;
                if let Some(pr) = &c.proc {
                    proc::term_group(pr.pid);
                }
                c.status = "Stopping…".into();
            }
            Stage::Corpus | Stage::Convert => {
                // the thread finishes its file; its result is then ignored
                c.stage = Stage::Idle;
                c.status = "Stopped".into();
            }
            Stage::Idle => {}
        }
    }
}

fn fmt_gib(b: f64) -> String {
    format!("{:.0} GiB", b / GIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems() {
        let full = "model.language_model.layers.3.self_attn.q_proj,model.language_model.layers.3.self_attn.k_proj,model.language_model.layers.3.self_attn.v_proj";
        assert_eq!(container_stems(full), vec!["blk.3.attn_qg", "blk.3.attn_k", "blk.3.attn_v"]);
        let lin = "m.layers.0.linear_attn.in_proj_qkv,m.layers.0.linear_attn.in_proj_z,m.layers.0.linear_attn.in_proj_a,m.layers.0.linear_attn.in_proj_b";
        assert_eq!(container_stems(lin), vec!["blk.0.ssm_inz", "blk.0.ssm_in", "blk.0.ssm_z", "blk.0.ssm_ab"]);
        assert_eq!(container_stems("m.layers.12.mlp.gate_proj,m.layers.12.mlp.up_proj"), vec!["blk.12.ffn_gate_up"]);
        assert_eq!(container_stems("m.layers.12.mlp.down_proj"), vec!["blk.12.ffn_down"]);
        assert_eq!(container_stems("lm_head"), vec!["output"]);
        assert!(container_stems("m.layers.1.something_else").is_empty());
    }

    #[test]
    fn slices() {
        assert_eq!(plan_slices(64, 1.0, 100.0), vec![(0, 64)]);
        assert_eq!(plan_slices(64, 1.0, 40.0), vec![(0, 32), (32, 64)]);
        assert_eq!(plan_slices(64, 1.0, 0.5), (0..64).map(|i| (i, i + 1)).collect::<Vec<_>>());
        assert!(plan_slices(0, 1.0, 1.0).is_empty());
    }

    #[test]
    fn convert_roundtrip() {
        let dir = std::env::temp_dir().join(format!("llmduo-calib-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // a 3x3 symmetric H: [[1,2,3],[2,4,5],[3,5,6]] -> upper packed 1 2 3 4 5 6
        let tri: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hdr = r#"{"__metadata__":{"k":"3","rows":"77","linears":"m.layers.5.mlp.gate_proj,m.layers.5.mlp.up_proj"},"hessian_upper":{"dtype":"F32","shape":[6],"data_offsets":[0,24]}}"#;
        let mut h = hdr.to_string();
        while h.len() % 8 != 0 {
            h.push(' ');
        }
        let mut f = std::fs::File::create(dir.join("x.input.safetensors")).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(h.as_bytes()).unwrap();
        for v in &tri {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(f);
        let s = convert_one(&dir.join("x.input.safetensors"), &dir).unwrap();
        assert_eq!(s, vec!["blk.5.ffn_gate_up"]);
        let b = std::fs::read(dir.join("gram.r0.blk.5.ffn_gate_up.bin")).unwrap();
        assert_eq!(&b[..8], b"RADGRAM2");
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 77);
        let m: Vec<f32> = b[32..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(m, vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0]);
        assert_eq!(grams_in(&dir), (1, 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Converts a real calibration run: LLMDUO_CALIB_RAW=<raw dir> LLMDUO_CALIB_OUT=<grams dir>
    /// cargo test convert_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn convert_real() {
        let (Ok(raw), Ok(out)) = (std::env::var("LLMDUO_CALIB_RAW"), std::env::var("LLMDUO_CALIB_OUT")) else { return };
        let mut files: Vec<PathBuf> = std::fs::read_dir(&raw).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        for f in files.iter().filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors")) {
            let s = convert_one(f, Path::new(&out)).unwrap();
            println!("{} -> {:?}", f.file_name().unwrap().to_string_lossy(), s);
        }
    }

    #[test]
    fn corpus() {
        let dir = std::env::temp_dir().join(format!("llmduo-corpus-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("in")).unwrap();
        std::fs::write(dir.join("in/a.txt"), "x".repeat(700)).unwrap();
        std::fs::write(dir.join("in/b.jsonl"), "{\"text\": \"hello there, a document of some length\"}\n\n{\"prompt\": \"another document, long enough\"}\n").unwrap();
        let (docs, _) = build_corpus(&dir.join("in"), &dir.join("c.jsonl"), 300).unwrap();
        assert_eq!(docs, 3 + 2); // 700 chars in pieces of 300: 300, 300, 100
        let bad = dir.join("bad.jsonl");
        std::fs::write(&bad, "{\"x\": 1}\n").unwrap();
        assert!(build_corpus(&bad, &dir.join("c2.jsonl"), 300).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
