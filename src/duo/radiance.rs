//! Radiance backend: the parts that don't need the app — model scanning, `rad-info` metadata,
//! log and /metrics parsing, failure hints, the recipe list for the convert tool.
//!
//! Radiance plans memory itself: it claims every card it is given (less `--gpu-headroom-mib`)
//! and splits it between resident weights and a paged KV pool, so LLM Duo does not estimate
//! VRAM the way it does for llama.cpp. It records what the engine reports instead (the KV pool
//! in tokens) and Fit asks the engine for its plan when nothing is recorded yet.

use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::util::*;

pub const RAD_KV_TYPES: [&str; 4] = ["bf16", "fp8", "fp8_v4", "k6_v4"];
pub const RAD_SPEC: [&str; 9] = ["auto", "0", "1", "2", "3", "4", "5", "6", "8"];
pub const RAD_DRAFT_WINDOW: [&str; 2] = ["fixed", "adaptive"];
pub const RAD_VISION: [&str; 3] = ["(default)", "vram", "ram"];
pub const RAD_EMBED: [&str; 3] = ["(default)", "ram", "vram"];
pub const RAD_PLACEMENT: [&str; 4] = ["auto", "all_vram", "layer_offload", "expert_tiered"];
pub const RAD_REASONING: [&str; 7] = ["(default)", "none", "minimal", "low", "medium", "high", "xhigh"];
pub const RAD_TP_WIRE: [&str; 2] = ["exact", "wht6"];
pub const RAD_STRATEGIES: [&str; 1] = ["Tensor parallel"];

/// Options that change how Radiance splits the card. A recorded KV pool is only reused for the
/// same values (context is not one: the pool is the same at any `--max-model-len` it can hold).
pub const RAD_POOL_KEYS: [&str; 10] =
    ["r_kv", "parallel", "r_batched", "r_headroom", "r_vision", "r_embed", "r_placement", "r_host_pool", "r_spec", "r_prefill_span"];

// ------------------------------------------------------------------ tools --
/// Where the rad-* tools live, set from the settings (Info::load has no access to them).
static TOOLS: OnceLock<Mutex<(String, String)>> = OnceLock::new();

pub fn set_tools(radiance_bin: &str, home: &str) {
    let m = TOOLS.get_or_init(|| Mutex::new((String::new(), String::new())));
    *m.lock().unwrap() = (radiance_bin.to_string(), home.to_string());
}

fn tools() -> (String, String) {
    TOOLS.get().map(|m| m.lock().unwrap().clone()).unwrap_or_default()
}

/// A tool next to the radiance binary (`rad-info`, `rad-convert`), else from PATH.
pub fn tool_path(radiance_bin: &str, name: &str) -> PathBuf {
    let b = expanduser(radiance_bin);
    if let Some(d) = b.parent() {
        let c = d.join(name);
        if c.is_file() {
            return c;
        }
    }
    PathBuf::from(name)
}

/// Run a short command with a deadline; (exit ok, stdout + stderr).
pub fn run_capture(cmd: &[String], env: &[(String, String)], timeout: Duration) -> Result<(bool, String), String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        c.env(k, v);
    }
    let mut child = c.spawn().map_err(|e| format!("{}: {e}", cmd[0]))?;
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let to = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let te = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = err.read_to_string(&mut s);
        s
    });
    let t0 = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t0.elapsed() < timeout => std::thread::sleep(Duration::from_millis(30)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let text = to.join().unwrap_or_default() + &te.join().unwrap_or_default();
    match status {
        Some(s) => Ok((s.success(), text)),
        None => Err(format!("{} did not finish within {} s", cmd[0], timeout.as_secs())),
    }
}

// ----------------------------------------------------------------- models --
/// `.rad` containers and checkpoint folders (config.json + safetensors) under the models folder.
pub fn scan_rad_models(root: &str) -> Vec<String> {
    let root = PathBuf::from(root);
    let mut out = vec![];
    if !root.is_dir() {
        return out;
    }
    fn walk(d: &Path, root: &Path, out: &mut Vec<String>, depth: usize) {
        if depth > 8 {
            return;
        }
        if d != root && super::vllm::is_vllm_model_dir(d) {
            if let Ok(r) = d.strip_prefix(root) {
                out.push(r.to_string_lossy().into_owned() + "/");
            }
            return; // a checkpoint's subfolders (draft heads, vision) are part of it
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut ents: Vec<_> = rd.filter_map(|e| e.ok()).collect();
        ents.sort_by_key(|e| e.file_name());
        for e in ents {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let ft = e.file_type();
            if ft.as_ref().map(|t| t.is_dir()).unwrap_or(false) || (ft.map(|t| t.is_symlink()).unwrap_or(false) && p.is_dir()) {
                walk(&p, root, out, depth + 1);
            } else if name.to_lowercase().ends_with(".rad") {
                if let Ok(r) = p.strip_prefix(root) {
                    out.push(r.to_string_lossy().into_owned());
                }
            }
        }
    }
    walk(&root, &root, &mut out, 0);
    out.sort_by_key(|a| a.to_lowercase());
    out
}

#[derive(Clone, Debug, Default)]
pub struct RadInfo {
    pub path: PathBuf,
    pub is_dir: bool,
    pub bytes: f64,
    pub arch: String,
    pub model: String,
    pub source_quant: String,
    pub format: String,
    /// `rad-info` output (the summary), `rad-info --meta` output
    pub summary_text: String,
    pub meta_text: String,
    pub meta: Vec<(String, String)>,
    pub n_ctx_train: i64,
    pub n_layer: usize,
    pub vision: bool,
    pub draft: bool,
    /// rad-info missing / failed: what went wrong (the model can still be served)
    pub note: String,
}

fn dir_bytes(d: &Path) -> f64 {
    let mut t = 0.0;
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.filter_map(|e| e.ok()) {
            let n = e.file_name().to_string_lossy().to_lowercase();
            if n.ends_with(".safetensors") || n.ends_with(".gguf") || n.ends_with(".bin") {
                t += e.metadata().map(|m| m.len() as f64).unwrap_or(0.0);
            }
        }
    }
    t
}

/// "  architecture  qwen4exp" -> ("architecture", "qwen4exp"). Keys may hold one space
/// ("source quant"), so the split is on the first run of two or more spaces.
fn kv_line(l: &str) -> Option<(String, String)> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"^\s*([A-Za-z0-9_.:/\-]+(?: [A-Za-z0-9_.:/\-]+)?)\s{2,}(.+?)\s*$").unwrap());
    let m = re.captures(l)?;
    Some((m[1].to_string(), m[2].to_string()))
}

fn leading_int(s: &str) -> Option<i64> {
    let t: String = s.trim().trim_start_matches(['=', ':']).trim().chars().take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '_').collect();
    let t = t.replace([',', '_'], "");
    if t.is_empty() { None } else { t.parse().ok() }
}

impl RadInfo {
    pub fn load(path: &Path) -> Result<RadInfo, String> {
        let mut r = RadInfo { path: path.to_path_buf(), ..Default::default() };
        if path.is_dir() {
            r.is_dir = true;
            r.bytes = dir_bytes(path);
            match super::vllm::VllmModelInfo::load(path) {
                Ok(v) => {
                    r.arch = v.arch.clone();
                    r.n_ctx_train = v.n_ctx_train;
                    r.n_layer = v.n_layer;
                    r.vision = v.multimodal;
                    r.draft = v.nextn != 0;
                    r.source_quant = v.quant.clone();
                    r.meta_text = v.metadata_text();
                    r.format = "checkpoint folder (served at its own precision)".into();
                }
                Err(e) => r.note = format!("config.json: {e}"),
            }
            return Ok(r);
        }
        let md = std::fs::metadata(path).map_err(|e| e.to_string())?;
        r.bytes = md.len() as f64;
        let (bin, home) = tools();
        let info = tool_path(&bin, "rad-info");
        let mut env = vec![];
        if !home.trim().is_empty() {
            env.push(("RADIANCE_HOME".to_string(), expanduser(&home).to_string_lossy().into_owned()));
        }
        let p = path.to_string_lossy().into_owned();
        match run_capture(&[info.to_string_lossy().into_owned(), p.clone()], &env, Duration::from_secs(30)) {
            Ok((ok, text)) => {
                if !ok && text.trim().is_empty() {
                    r.note = "rad-info failed".into();
                }
                r.summary_text = text;
            }
            Err(e) => r.note = format!("rad-info not available ({e}) - set the Radiance binary in Settings"),
        }
        if r.note.is_empty() {
            if let Ok((_, text)) = run_capture(&[info.to_string_lossy().into_owned(), "--meta".into(), p], &env, Duration::from_secs(30)) {
                r.meta_text = text;
            }
        }
        r.parse();
        Ok(r)
    }

    fn parse(&mut self) {
        for l in self.summary_text.lines().take(20) {
            if let Some((k, v)) = kv_line(l) {
                match k.as_str() {
                    "architecture" => self.arch = v,
                    "model" => self.model = v,
                    "source quant" => self.source_quant = v,
                    "format" => self.format = v,
                    _ => {}
                }
            }
        }
        for l in self.meta_text.lines() {
            let t = l.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            // "key  value", "key = value", "key: value" or "key  type  value"
            let (k, v) = if let Some((k, v)) = t.split_once(" = ") {
                (k.trim().to_string(), v.trim().to_string())
            } else if let Some((k, v)) = kv_line(l) {
                (k, v)
            } else if let Some((k, v)) = t.split_once(": ") {
                (k.trim().to_string(), v.trim().to_string())
            } else {
                continue;
            };
            let last = v.split_whitespace().last().unwrap_or("").to_string();
            let kl = k.to_lowercase();
            let num = leading_int(&last).or_else(|| leading_int(&v));
            if self.n_ctx_train == 0
                && (kl.ends_with("context_length") || kl.ends_with("max_position_embeddings") || kl.ends_with("n_ctx_train")
                    || kl.ends_with("max_seq_len") || kl.ends_with("seq_length"))
                && !kl.contains("vision")
                && !kl.starts_with("draft")
            {
                if let Some(n) = num.filter(|n| *n >= 512) {
                    self.n_ctx_train = n;
                }
            }
            if self.n_layer == 0 && (kl.ends_with("num_hidden_layers") || kl.ends_with("block_count") || kl.ends_with("n_layer"))
                && !kl.contains("vision") && !kl.starts_with("draft")
            {
                if let Some(n) = num.filter(|n| *n > 0 && *n < 1000) {
                    self.n_layer = n as usize;
                }
            }
            if kl.contains("vision") || kl.contains("visual") || kl.contains("mm_") {
                self.vision = true;
            }
            if kl.starts_with("draft") || kl.contains("nextn") || kl.contains("mtp") {
                self.draft = true;
            }
            if self.arch.is_empty() && (kl == "architecture" || kl.ends_with("general.architecture")) {
                self.arch = last.clone();
            }
            self.meta.push((k, v));
        }
        let low = self.summary_text.to_lowercase();
        if low.contains("vision") || low.contains("visual") {
            self.vision = true;
        }
        if low.contains("draft.") || low.contains("draft head") {
            self.draft = true;
        }
    }

    pub fn name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![];
        if !self.arch.is_empty() {
            parts.push(format!("arch {}", self.arch));
        }
        if !self.model.is_empty() && self.model != self.name() {
            parts.push(self.model.clone());
        }
        parts.push(format!("{:.2} GiB", self.bytes / GIB));
        if !self.source_quant.is_empty() && self.source_quant != "(none)" {
            parts.push(format!("from {}", self.source_quant));
        }
        if self.n_layer != 0 {
            parts.push(format!("{} layers", self.n_layer));
        }
        if self.n_ctx_train != 0 {
            parts.push(format!("train ctx {}", fmt_thousands(self.n_ctx_train)));
        }
        if self.vision {
            parts.push("vision".into());
        }
        if self.draft {
            parts.push("draft head".into());
        }
        if self.is_dir {
            parts.push("checkpoint folder".into());
        }
        let mut s = parts.join("  •  ");
        if !self.note.is_empty() {
            s += &format!("\n⚠ {}", self.note);
        }
        s
    }

    pub fn metadata_text(&self) -> String {
        let mut t = String::new();
        if !self.summary_text.is_empty() {
            t += "==== rad-info ====\n";
            t += &self.summary_text;
            t += "\n";
        }
        if !self.meta_text.is_empty() {
            t += "\n==== rad-info --meta ====\n";
            t += &self.meta_text;
        }
        if t.is_empty() {
            t = format!("{}\n\n(no metadata: {})", self.path.display(), if self.note.is_empty() { "nothing to show" } else { &self.note });
        }
        t
    }

    pub fn sig(&self) -> String {
        format!("rad:{}:{}:{}", self.arch, self.n_layer, (self.bytes / MIB) as i64)
    }
}

// --------------------------------------------------------------- the log --
pub struct RRe {
    pub serving: Regex,
    pub kv_tokens: Regex,
    pub kv_gib: Regex,
    pub backs: Regex,
    pub settled: Regex,
    pub arena: Regex,
    pub chat501: Regex,
    pub warn: Regex,
    pub err: Regex,
    pub refused: Regex,
}

pub fn rre() -> &'static RRe {
    static R: OnceLock<RRe> = OnceLock::new();
    R.get_or_init(|| RRe {
        serving: Regex::new(r"\bserving on\b").unwrap(),
        // "kv cache 11.59 GiB = 262,160 tokens of context" (wording varies a little by version)
        kv_tokens: Regex::new(r"(?i)\bkv\b.*?(\d[\d,_]*)\s+tokens(?: of context)?\b").unwrap(),
        kv_gib: Regex::new(r"(?i)\bkv cache\b[^\d]*?([\d.]+)\s*(GiB|MiB)").unwrap(),
        // "kv: the pool backs 100% of the worst case ... 1.0 full-length sessions"
        backs: Regex::new(r"(?i)pool backs\s+([\d.]+)\s*%.*?([\d.]+)\s+full-length sessions?").unwrap(),
        // "card 0 settled at 104 MiB free"
        settled: Regex::new(r"(?i)card\s+(\d+)\s+settled at\s+([\d.]+)\s*(GiB|MiB)\s+free").unwrap(),
        arena: Regex::new(r"(?i)activation arena\b[^\d]*?([\d.]+)\s*(GiB|MiB)").unwrap(),
        chat501: Regex::new(r"chat is 501").unwrap(),
        warn: Regex::new(r"^\s*(?:\[[^\]]*\]\s*)?W\s").unwrap(),
        err: Regex::new(r"^\s*(?:\[[^\]]*\]\s*)?E\s").unwrap(),
        refused: Regex::new(r"(?i)context_length_exceeded|exceeds the KV").unwrap(),
    })
}

pub fn gib_of(v: &str, unit: &str) -> f64 {
    let x: f64 = v.parse().unwrap_or(0.0);
    if unit.eq_ignore_ascii_case("mib") { x * MIB } else { x * GIB }
}

/// The KV pool in tokens from one log line, if it states it.
pub fn kv_tokens_of(line: &str) -> Option<i64> {
    let re = rre();
    if re.backs.is_match(line) {
        return None; // "the pool backs N% ... M full-length sessions" is a ratio, not the size
    }
    let low = line.to_lowercase();
    if re.err.is_match(line) || low.contains("did not fit") || low.contains("refused") || low.contains("short by") {
        return None; // an error naming a size is not the pool
    }
    if !(low.contains("tokens of context") || low.contains("kv cache") || low.contains("kv pool")) {
        return None;
    }
    static C: OnceLock<Regex> = OnceLock::new();
    let ctx_re = C.get_or_init(|| Regex::new(r"(\d[\d,_]*)\s+tokens of context").unwrap());
    let m = ctx_re.captures(line).or_else(|| re.kv_tokens.captures(line))?;
    let n: i64 = m[1].replace([',', '_'], "").parse().ok()?;
    if n >= 256 { Some(n) } else { None }
}

// ----------------------------------------------------------- VRAM budget --
/// The startup log's "VRAM budget, per card" table and the cache's per-sequence state.
///
/// The pool Radiance builds is capped at Max seqs x context, so the pool alone is only what
/// was asked for. What the card could hold is the pool plus the table's `unusable` line (what
/// neither the cache nor the experts could address), at the pool's own cost per token.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    pub elastic: Option<f64>,
    pub kv_bytes: Option<f64>,
    pub unusable: Option<f64>,
    /// per-sequence state inside the pool (recurrent / conv state, sliding windows)
    pub state: Option<f64>,
    pub kv_tokens: Option<i64>,
}

fn bytes_of(v: &str, unit: &str) -> f64 {
    let x: f64 = v.parse().unwrap_or(0.0);
    match unit {
        "GiB" => x * GIB,
        "MiB" => x * MIB,
        "KiB" => x * 1024.0,
        _ => x,
    }
}

impl Budget {
    /// Feed one log line; true when it carried something.
    pub fn line(&mut self, line: &str) -> bool {
        static R: OnceLock<(Regex, Regex)> = OnceLock::new();
        let (row, state) = R.get_or_init(|| {
            (
                Regex::new(r"^\s*(?:I\s+)?\s*(elastic|kv cache|unusable)\s+([\d.]+)\s*(B|KiB|MiB|GiB)\b").unwrap(),
                Regex::new(r"([\d.]+)\s*(B|KiB|MiB|GiB) of (?:it|that) is per-sequence state").unwrap(),
            )
        });
        let mut hit = false;
        if let Some(c) = row.captures(line) {
            let b = bytes_of(&c[2], &c[3]);
            match &c[1] {
                "elastic" => self.elastic = Some(b),
                "kv cache" => self.kv_bytes = Some(b),
                _ => self.unusable = Some(b),
            }
            hit = true;
        }
        if let Some(c) = state.captures(line) {
            self.state = Some(bytes_of(&c[1], &c[2]));
            hit = true;
        }
        if let Some(n) = kv_tokens_of(line) {
            self.kv_tokens = Some(n);
            hit = true;
        }
        hit
    }

    /// Tokens the card could hold at these memory settings (None until the log has said enough).
    pub fn capacity(&self) -> Option<i64> {
        let tokens = self.kv_tokens?;
        let kv = self.kv_bytes?;
        let unusable = self.unusable.unwrap_or(0.0);
        let state = self.state.unwrap_or(0.0).min(kv * 0.9);
        let per_tok = (kv - state) / tokens.max(1) as f64;
        if per_tok <= 0.0 {
            return None;
        }
        Some(tokens + (unusable / per_tok) as i64)
    }
}

/// The client address of a "req refused from HOST:PORT -> 401 … api key" line.
pub fn refused_401_from(line: &str) -> Option<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"req refused from \[?([0-9A-Fa-f:.]+?)\]?:\d+ -> 401\b.*api key").unwrap());
    re.captures(line).map(|c| c[1].to_string())
}

/// Loopback, private (RFC 1918 / unique local) and link-local addresses.
pub fn is_local_addr(a: &str) -> bool {
    match a.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v)) => v.is_loopback() || v.is_private() || v.is_link_local(),
        Ok(std::net::IpAddr::V6(v)) => {
            let s = v.segments()[0];
            v.is_loopback() || (s & 0xfe00) == 0xfc00 || (s & 0xffc0) == 0xfe80
                || v.to_ipv4_mapped().map(|m| m.is_loopback() || m.is_private() || m.is_link_local()).unwrap_or(false)
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------- metrics --
/// Prometheus text from Radiance's /metrics: both families (`vllm:` and `radiance:`), keys kept
/// with their family prefix. Series with labels are summed; histogram buckets are skipped.
pub fn parse_rad_metrics(text: &str) -> HashMap<String, f64> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"(?m)^((?:vllm|radiance):[A-Za-z0-9_:]+)(?:\{[^}]*\})?\s+(\S+)").unwrap());
    let mut out: HashMap<String, f64> = HashMap::new();
    for m in re.captures_iter(text) {
        let k = &m[1];
        if k.ends_with("_bucket") || k.ends_with("_created") {
            continue;
        }
        let Some(f) = super::vllm::py_parse_float(&m[2]) else { continue };
        if !f.is_finite() {
            continue;
        }
        *out.entry(k.to_string()).or_insert(0.0) += f;
    }
    out
}

// ---------------------------------------------------------------- failure --
pub fn rad_failure_hint(lines: &[String]) -> String {
    let text = lines.join("\n");
    let m = |p: &str| Regex::new(p).unwrap().is_match(&text);
    if m(r"resolve only to a reference implementation|libref") {
        return "Some ops have no kernel for this card (only the reference implementation).\nCheck Radiance home in Settings (it must hold kernels/, architectures/ and quantizers/), or rebuild Radiance for this GPU.".into();
    }
    if m(r"DID NOT FIT|did not fit the KV pool") {
        return "The memory plan did not fit.\nLower Context or Max seqs, use an fp8 KV cache, lower Batch tokens (a smaller activation arena), put the vision tower in RAM, or give it a Host pool (MoE models).".into();
    }
    if m(r"unknown option '([^']*)'") {
        let o = Regex::new(r"unknown option '([^']*)'").unwrap().captures(&text).map(|c| c[1].to_string()).unwrap_or_default();
        return format!("This Radiance build does not know the option {o}.\nRemove it from Extra args (or update Radiance; run radiance --help for its options).");
    }
    if m(r"(?i)does not read '[^']*' encoded") {
        return "No loaded kernel reads one of this container's weight encodings.\nConvert it with a recipe this build serves, or check Radiance home in Settings.".into();
    }
    if m(r"(?i)pool allocation refused|backing allocation of .* refused") {
        return "The card (or pinned host memory) refused an allocation - something else took memory after Radiance measured the card.\nStop the other process on that GPU (the other pane, a desktop app), or lower Host pool.".into();
    }
    if m(r"(?i)out of memory|would not back it") {
        return "Something else is holding VRAM on this card. Stop it (rocm-smi --showpids) and load again.".into();
    }
    if m(r"(?i)Address already in use|bind.*in use") {
        return "That port is already in use - pick another Port.".into();
    }
    if m(r"(?i)vision-tower-placement.*(?:refused|no tower|cannot)|no (?:vision )?tower") {
        return "This model has no vision tower to move to RAM - set Vision tower to (default).".into();
    }
    if m(r"(?i)kv-cache-dtype.*(?:refus|not offered|unsupported)|refuses? (?:fp8_v4|k6_v4)") {
        return "That KV cache type is only offered for some models (fp8_v4 / k6_v4: Qwen3.8-27B and Flash-Next). Use fp8 or bf16.".into();
    }
    if m(r"(?i)no queue advance|GPU Hang|MODE1 reset") {
        return "The GPU stopped running Radiance's queue and was reset.\nUsually compute wave save/restore is off: cat /sys/module/amdgpu/parameters/cwsr_enable must print 1. A GPU voltage offset is the other known cause.".into();
    }
    if m(r"(?i)HSA_STATUS_ERROR_OUT_OF_RESOURCES|segmentation fault|SIGSEGV") {
        return "A kernel could not load for lack of VRAM headroom.\nRaise Headroom to the value the startup warning names (W lines in the log).".into();
    }
    if m(r"(?i)draft-model is not implemented") {
        return "Radiance takes its drafter from the container - merge one with the Convert tool (Draft model) instead of passing --draft-model.".into();
    }
    if m(r"(?i)no such file|cannot open|failed to open") && m(r"(?i)radiance_home|architectures|kernels") {
        return "Radiance could not find its plugins. Set Radiance home in Settings (e.g. ~/radiance/build/radiance_home).".into();
    }
    if let Some(c) = Regex::new(r"(?m)^\s*(?:\[[^\]]*\]\s*)?E\s+(.{8,})$").unwrap().captures_iter(&text).last() {
        return format!("Radiance reported: {}", c[1].trim());
    }
    String::new()
}

// ---------------------------------------------------------------- recipes --
#[derive(Clone, Debug)]
pub struct Recipe {
    pub name: String,
    pub path: PathBuf,
    /// the comment lines at the top (what the recipe is for; often names --set keys it needs)
    pub about: String,
    /// the whole file (its rules decide what the trunk is quantised to)
    pub text: String,
}

pub fn list_recipes(dir: &str) -> Vec<Recipe> {
    let d = expanduser(dir);
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(&d) else { return out };
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        let n = e.file_name().to_string_lossy().into_owned();
        if !n.ends_with(".recipe") {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap_or_default();
        let about: Vec<String> = text
            .lines()
            .take_while(|l| l.trim().starts_with('#') || l.trim().is_empty())
            .filter(|l| l.trim().starts_with('#'))
            .map(|l| l.trim().trim_start_matches('#').trim().to_string())
            .filter(|l| !l.to_uppercase().starts_with("PATTERN"))
            .take(14)
            .collect();
        out.push(Recipe { name: n.trim_end_matches(".recipe").to_string(), path: p, about: about.join("\n"), text });
    }
    out.sort_by_key(|r| r.name.to_lowercase());
    out
}

/// `KEY=VALUE` metadata a recipe's comments ask for, in either spelling a header uses:
/// `--set quantization_config.quant_method=fp8`, `--set "…weight_block_size=128 128"`, or the bare
/// `quantization_config.fmt=e4m3, …` form. Later duplicates of a key are dropped.
pub fn recipe_sets(comments: &str) -> Vec<String> {
    static R: OnceLock<(Regex, Regex)> = OnceLock::new();
    let (flag, bare) = R.get_or_init(|| {
        (
            Regex::new(r#"--set\s+(?:"([^"]+)"|'([^']+)'|(\S+=\S+))"#).unwrap(),
            Regex::new(r#"(quantization_config\.[A-Za-z0-9_.]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s,;"'`)]+(?:[ \t]+\d+)*))"#).unwrap(),
        )
    });
    let mut out: Vec<(String, String)> = vec![];
    let mut push = |kv: &str| {
        if let Some((k, v)) = kv.split_once('=') {
            let (k, v) = (k.trim().to_string(), v.trim().trim_end_matches(['.', ',']).to_string());
            if !k.is_empty() && !out.iter().any(|x| x.0 == k) {
                out.push((k, v));
            }
        }
    };
    for c in flag.captures_iter(comments) {
        let kv = c.get(1).or(c.get(2)).or(c.get(3)).map(|m| m.as_str()).unwrap_or("");
        push(kv);
    }
    for c in bare.captures_iter(comments) {
        let v = c.get(2).or(c.get(3)).or(c.get(4)).map(|m| m.as_str()).unwrap_or("");
        push(&format!("{}={v}", &c[1]));
    }
    out.into_iter().map(|(k, v)| format!("{k}={v}")).collect()
}

/// The Metadata box's text for a list of `KEY=VALUE`: one shell word each (quoted when needed).
pub fn sets_text(sets: &[String]) -> String {
    sets.iter().map(|s| shlex_quote(s)).collect::<Vec<_>>().join(" ")
}

/// What a recipe does to the trunk (the `blk.*` weights; drafter rules left out).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecipeKind {
    /// grids the trunk rules write: "fp4_e2m1", "fp8_e4m3", "i4", "i8", "table", …
    pub codes: Vec<String>,
    pub fwht: bool,
    pub draft_rules: bool,
}

pub fn recipe_kind(text: &str) -> RecipeKind {
    let mut k = RecipeKind::default();
    for l in text.lines() {
        let t = l.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let mut parts = t.split_whitespace();
        let (Some(pat), Some(_q)) = (parts.next(), parts.next()) else { continue };
        let pl = pat.to_lowercase();
        if pl.starts_with("dflash") || pl.starts_with("draft") || pl.starts_with("dspark") || pl.starts_with("mtp") {
            k.draft_rules = true;
            continue;
        }
        if !pl.starts_with("blk.") {
            continue;
        }
        for opt in parts {
            if let Some(c) = opt.strip_prefix("codes=") {
                if !k.codes.iter().any(|x| x == c) {
                    k.codes.push(c.to_string());
                }
            } else if opt.starts_with("table=") && !k.codes.iter().any(|x| x == "table") {
                k.codes.push("table".into());
            } else if opt.contains("fwht") {
                k.fwht = true;
            }
        }
    }
    k
}

impl RecipeKind {
    pub fn quantises_trunk(&self) -> bool {
        !self.codes.is_empty()
    }
    /// A short name for the format, for output file names: "mxfp4", "fp8", "w8a8", "w4", …
    pub fn tag(&self, recipe_name: &str) -> String {
        let has = |c: &str| self.codes.iter().any(|x| x == c);
        let rn = recipe_name.to_lowercase();
        if has("fp4_e2m1") {
            "mxfp4".into()
        } else if has("i8") || has("u8") {
            if rn.contains("i8a8") { "i8a8".into() } else { "w8a8".into() }
        } else if has("i4") || has("u4") {
            if rn.contains("w4a8") { "w4a8".into() } else if rn.contains("w4a16") { "w4a16".into() } else { "w4".into() }
        } else if has("i5") || has("u5") {
            "w5".into()
        } else if has("fp8_e4m3") {
            "fp8".into()
        } else if has("table") {
            "w4nl".into()
        } else {
            String::new()
        }
    }
}

/// What a checkpoint folder is: its family and whether it is already quantised.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CkptKind {
    /// config.json's model_type (top level, else text_config's)
    pub model_type: String,
    pub architectures: Vec<String>,
    /// Some(true): quantization_config present or quantised tensors; Some(false): bf16/f16/f32
    pub quantised: Option<bool>,
}

fn safetensors_dtypes(path: &Path) -> Vec<String> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return vec![] };
    let mut n = [0u8; 8];
    if f.read_exact(&mut n).is_err() {
        return vec![];
    }
    let len = u64::from_le_bytes(n);
    if len == 0 || len > 64 << 20 {
        return vec![];
    }
    let mut buf = vec![0u8; len as usize];
    if f.read_exact(&mut buf).is_err() {
        return vec![];
    }
    let Ok(v) = serde_json::from_slice::<Value>(&buf) else { return vec![] };
    v.as_object()
        .map(|o| o.iter().filter(|(k, _)| *k != "__metadata__").filter_map(|(_, t)| t.get("dtype").and_then(|d| d.as_str()).map(String::from)).collect())
        .unwrap_or_default()
}

pub fn checkpoint_kind(dir: &Path) -> CkptKind {
    let mut k = CkptKind::default();
    let cfg: Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let tc = cfg.get("text_config").cloned().unwrap_or(Value::Null);
    k.model_type = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| tc.get("model_type").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    k.architectures = cfg
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if cfg.get("quantization_config").map(|v| !v.is_null()).unwrap_or(false)
        || tc.get("quantization_config").map(|v| !v.is_null()).unwrap_or(false)
    {
        k.quantised = Some(true);
        return k;
    }
    if cfg.is_null() {
        return k;
    }
    // no quantization_config: look at the tensors of the first shard
    let shard = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get("weight_map").and_then(|m| m.as_object()).and_then(|m| m.values().next().and_then(|x| x.as_str()).map(String::from)))
        .map(|f| dir.join(f))
        .or_else(|| Some(dir.join("model.safetensors")).filter(|p| p.is_file()));
    let dts = shard.map(|p| safetensors_dtypes(&p)).unwrap_or_default();
    if dts.is_empty() {
        k.quantised = Some(false); // no config, no readable tensors: a plain release
        return k;
    }
    let q = dts.iter().filter(|d| matches!(d.as_str(), "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "I32" | "F4")).count();
    k.quantised = Some(q * 10 > dts.len());
    k
}

impl CkptKind {
    pub fn family(&self) -> &'static str {
        let mt = self.model_type.to_lowercase();
        let ar = self.architectures.join(" ").to_lowercase();
        if mt.starts_with("qwen3_5") || mt.starts_with("qwen35") || ar.contains("qwen3_5") || ar.contains("qwen35") {
            if mt.contains("moe") || ar.contains("moe") { "qwen35moe" } else { "qwen35" }
        } else if mt.starts_with("gemma4") || ar.contains("gemma4") {
            "gemma4"
        } else if mt.starts_with("gemma3") || ar.contains("gemma3") {
            "gemma3"
        } else if mt == "llama" {
            "llama"
        } else {
            ""
        }
    }
}

/// The `--set` keys that pick the quantised serving plugin when a bf16 checkpoint is converted
/// with a recipe that quantises its trunk (a quantised release carries them in its own config).
pub fn known_sets(family: &str, rk: &RecipeKind) -> Option<(Vec<String>, String)> {
    let has = |c: &str| rk.codes.iter().any(|x| x == c);
    let fp8 = || {
        vec![
            "quantization_config.quant_method=fp8".to_string(),
            "quantization_config.fmt=e4m3".into(),
            "quantization_config.weight_block_size=128 128".into(),
            "quantization_config.activation_scheme=dynamic".into(),
        ]
    };
    match family {
        "qwen35" | "qwen35moe" if (has("i8") || has("u8")) && rk.fwht => Some((
            vec!["quantization_config.quant_method=w8a8".into(), "quantization_config.fmt=fwht128".into()],
            "select qwen35_w8a8, the int8 plugin".into(),
        )),
        "qwen35" | "qwen35moe" if rk.quantises_trunk() => {
            Some((fp8(), format!("select {}_fp8, which serves block-FP8, MXFP4 and int4 trunks", family)))
        }
        "gemma4" if has("i4") || has("u4") => Some((
            vec!["quantization_config.quant_method=compressed-tensors".into()],
            "select gemma4_bf16, the plugin that serves Gemma 4 31B at w4a8".into(),
        )),
        "llama" if has("fp8_e4m3") => None, // llama_dense_fp8 is chosen without keys
        _ => None,
    }
}

/// The log of a failed plan: a bf16 plugin was asked to hold quantised weights.
pub fn bf16_plugin_mismatch(lines: &[String]) -> bool {
    lines.iter().any(|l| l.contains("is declared bf16") && l.contains("holds codes as"))
}

pub fn opt_str(v: Option<&Value>) -> String {
    v.map(py_str).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_lines() {
        assert_eq!(kv_tokens_of("I kv cache 11.59 GiB = 262,160 tokens of context"), Some(262160));
        assert_eq!(kv_tokens_of("I   kv cache: fp8, 40.00 KiB a token, 262160 tokens of context (1 x 262144)"), Some(262160));
        assert_eq!(kv_tokens_of("I kv: the pool backs 100% of the worst case, 1.0 full-length sessions"), None);
        assert_eq!(kv_tokens_of("I prefill chunks of at most 2048 tokens"), None);
        assert_eq!(kv_tokens_of("I the cache holds 117,000 tokens of context"), Some(117000));
        assert_eq!(kv_tokens_of("E KV group 'kv_attn': 1 x 262144 tokens did not fit the KV pool (DID NOT FIT)"), None);
        let b = rre().backs.captures("I kv: the pool backs 87% of the worst case (2 x 131072 tokens): 1.7 full-length sessions").unwrap();
        assert_eq!(&b[1], "87");
        assert_eq!(&b[2], "1.7");
        let s = rre().settled.captures("I card 0 settled at 1.21 GiB free (the budget read 1.30)").unwrap();
        assert_eq!(&s[2], "1.21");
        assert!(rre().serving.is_match("I serving on http://0.0.0.0:8080 (chat, completions)"));
    }

    #[test]
    fn budget() {
        let log = "I VRAM budget, per card:\n  card total          31.86 GiB\n  elastic             13.82 GiB   the experts keep\n    experts                 0 B   0% of it\n    kv cache           3.21 GiB   23% of it -- backs 1 sequences all at 95000 tokens\n    unusable          10.60 GiB   neither side could take it\nI kv: the cache can grow to 3.21 GiB a card -- 95008 tokens of context, 1.0 sessions at 95000 -- with every block it lends the experts taken back; 297.38 MiB of that is per-sequence state, and nothing is lent\nI   kv cache                           3.21 GiB a card: 95008 tokens, 1.0 sessions at 95000";
        let mut b = Budget::default();
        for l in log.lines() {
            b.line(l);
        }
        assert_eq!(b.kv_tokens, Some(95008));
        let cap = b.capacity().unwrap();
        assert!(cap > 420_000 && cap < 460_000, "{cap}");
        // a pool the card itself limits: nothing unusable, the pool is the capacity
        let mut c = Budget { kv_tokens: Some(1000), kv_bytes: Some(GIB), unusable: Some(0.0), ..Default::default() };
        assert_eq!(c.capacity(), Some(1000));
        c.unusable = None;
        assert_eq!(c.capacity(), Some(1000));
    }

    #[test]
    fn refused() {
        let l = "I req refused from 127.0.0.1:45698 -> 401 authentication_error: invalid api key";
        assert_eq!(refused_401_from(l).as_deref(), Some("127.0.0.1"));
        assert!(is_local_addr("127.0.0.1") && is_local_addr("192.168.1.20") && is_local_addr("10.0.0.5") && is_local_addr("::1"));
        assert!(!is_local_addr("8.8.8.8"));
        assert_eq!(refused_401_from("I req refused from [::1]:5000 -> 401 authentication_error: invalid api key").as_deref(), Some("::1"));
        assert!(refused_401_from("I req refused from 127.0.0.1:1 -> 400 invalid_request_error: max_tokens").is_none());
    }

    #[test]
    fn metrics() {
        let t = "# HELP x\nvllm:num_requests_running{model_name=\"m\"} 1\nradiance:decode_tokens_total 120\n\
                 radiance:draft_verify_depth_total{depth=\"2\"} 3\nradiance:draft_verify_depth_total{depth=\"3\"} 4\n\
                 vllm:time_to_first_token_seconds_bucket{le=\"1\"} 2\nvllm:time_to_first_token_seconds_sum 0.5\n";
        let m = parse_rad_metrics(t);
        assert_eq!(m["vllm:num_requests_running"], 1.0);
        assert_eq!(m["radiance:decode_tokens_total"], 120.0);
        assert_eq!(m["radiance:draft_verify_depth_total"], 7.0);
        assert!(!m.contains_key("vllm:time_to_first_token_seconds_bucket"));
    }

    #[test]
    fn info_parse() {
        let mut r = RadInfo {
            summary_text: "/m/gemma.rad\n  format        RAD2, 19.73 GiB, little-endian\n  architecture  gemma4\n  model         gemma-4-31b\n  source quant  compressed-tensors\n".into(),
            meta_text: "  gemma4.context_length      u32   262144\n  text_config.num_hidden_layers = 60\n  vision_config.hidden_size  1152\n".into(),
            ..Default::default()
        };
        r.parse();
        assert_eq!(r.arch, "gemma4");
        assert_eq!(r.source_quant, "compressed-tensors");
        assert_eq!(r.n_ctx_train, 262144);
        assert_eq!(r.n_layer, 60);
        assert!(r.vision);
    }

    #[test]
    fn hint() {
        let l = vec!["I x".to_string(), "E unknown option '--foo'".to_string()];
        assert!(rad_failure_hint(&l).contains("--foo"));
    }

    #[test]
    fn sets() {
        let a = "Gemma 4 31B or any bf16 finetune\nneeds --set quantization_config.quant_method=compressed-tensors";
        assert_eq!(recipe_sets(a), vec!["quantization_config.quant_method=compressed-tensors".to_string()]);
        let b = "bf16 Qwen3.8-27B as block fp8: --set quantization_config.quant_method=fp8 --set quantization_config.fmt=e4m3\n--set \"quantization_config.weight_block_size=128 128\" --set quantization_config.activation_scheme=dynamic";
        let v = recipe_sets(b);
        assert_eq!(v.len(), 4, "{v:?}");
        assert_eq!(v[2], "quantization_config.weight_block_size=128 128");
        let c = "needs quantization_config.quant_method=w8a8, quantization_config.fmt=fwht128";
        assert_eq!(recipe_sets(c), vec!["quantization_config.quant_method=w8a8".to_string(), "quantization_config.fmt=fwht128".into()]);
        let d = "keys: quantization_config.weight_block_size=128 128 and quantization_config.fmt=e4m3.";
        assert_eq!(recipe_sets(d), vec!["quantization_config.weight_block_size=128 128".to_string(), "quantization_config.fmt=e4m3".into()]);
        assert_eq!(sets_text(&v[2..3]), "'quantization_config.weight_block_size=128 128'");
    }

    #[test]
    fn kinds() {
        let r = "# PATTERN QUANTISER OPTIONS\nblk.*.ffn_*.weight  rtn codes=fp4_e2m1 group=32 scale=e8m0\noutput.weight rtn codes=fp8_e4m3 block=128x128\ndflash.*.weight cast dtype=bf16\n";
        let k = recipe_kind(r);
        assert_eq!(k.codes, vec!["fp4_e2m1".to_string()]);
        assert!(k.draft_rules);
        assert_eq!(k.tag("q38-27b-nvfp4-df2"), "mxfp4");
        let (sets, _) = known_sets("qwen35", &k).unwrap();
        assert_eq!(sets[0], "quantization_config.quant_method=fp8");
        let w8 = recipe_kind("blk.*.weight rtn codes=i8 group=128 transform=fwht128\n");
        assert_eq!(known_sets("qwen35", &w8).unwrap().0[0], "quantization_config.quant_method=w8a8");
        assert_eq!(w8.tag("q38-27b-w8a8-df2"), "w8a8");
        let g = recipe_kind("blk.*.weight rtn codes=i4 group=128 rule=search scale=f16\n");
        assert_eq!(known_sets("gemma4", &g).unwrap().0, vec!["quantization_config.quant_method=compressed-tensors".to_string()]);
        assert_eq!(g.tag("gemma4-31b-dense-bf16-w4a8"), "w4a8");
        assert!(known_sets("", &g).is_none());
        let ck = CkptKind { model_type: "qwen3_5".into(), ..Default::default() };
        assert_eq!(ck.family(), "qwen35");
        assert!(bf16_plugin_mismatch(&["E declare: weight 'blk.0.ffn_gate_up.weight' is declared bf16, and 'blk.0.ffn_gate_up.weight' holds codes as fp4_e2m1*e8m0[1x32]: only".to_string()]));
    }
}
