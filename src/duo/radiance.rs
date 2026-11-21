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

/// 2.0 dropped fp8 (GLM-5.3-Flash only) and the packed fp8_v4 / k6_v4 caches; i8 is the 8-bit cache.
pub const RAD_KV_TYPES: [&str; 2] = ["bf16", "i8"];
pub const RAD_SPEC: [&str; 9] = ["auto", "0", "1", "2", "3", "4", "5", "6", "8"];
pub const RAD_DRAFT_WINDOW: [&str; 2] = ["fixed", "adaptive"];
pub const RAD_VISION: [&str; 3] = ["(default)", "vram", "ram"];
pub const RAD_EMBED: [&str; 3] = ["(default)", "ram", "vram"];
/// `--ngram-placement`: where an n-gram embedding table lives (Qwen3.8-Flash-Next: 47.7 GiB).
pub const RAD_NGRAM: [&str; 4] = ["(default)", "vram", "ram", "disk"];
pub const RAD_PLACEMENT: [&str; 4] = ["auto", "all_vram", "layer_offload", "expert_tiered"];
/// `--control-vector`: disabled loads nothing; off / on load the vector found for the model and
/// start requests without / with it (a request's `control_vector` field overrides).
pub const RAD_CVEC: [&str; 3] = ["disabled", "off", "on"];
/// `--state-cache-dtype`: the recurrent (delta-net) state cache; f16 is lossy, qwen35 plugins only.
pub const RAD_STATE_DTYPE: [&str; 3] = ["(default)", "f32", "f16"];
pub const RAD_PREFETCH: [&str; 2] = ["on", "off"];
/// `--tp-attn-split`: how a prefill-sized step's attention splits over the ranks.
pub const RAD_ATTN_SPLIT: [&str; 2] = ["heads", "rows"];
/// `--p2p`: how the collectives reach the other cards.
pub const RAD_P2P: [&str; 3] = ["auto", "on", "off"];

/// How a free-text option's value is checked before Load (blank always means "not set").
#[derive(Clone, Copy, Debug)]
pub enum Check {
    /// a number in [min, max] (`open_min`: above min), `int` a whole number, `auto` also takes "auto"
    Num { min: f64, max: f64, open_min: bool, int: bool, auto: bool },
    /// `none` or `yarn:FACTOR[:ORIG]`
    Rope,
    /// N in 0..=64, `auto` or `auto:SHARE` with 0 < SHARE <= 4
    Interleave,
}

const fn num(min: f64, max: f64, open_min: bool, int: bool) -> Check {
    Check::Num { min, max, open_min, int, auto: false }
}

/// A Radiance option typed as text: passed as `flag value` when not blank.
pub struct RadTextOpt {
    pub key: &'static str,
    pub flag: &'static str,
    pub label: &'static str,
    pub check: Check,
}

const INF: f64 = f64::INFINITY;

/// The free-text options: the Advanced ones, then the sampling defaults (radiance --help).
pub const RAD_TEXT_OPTS: [RadTextOpt; 19] = [
    RadTextOpt { key: "r_rope", flag: "--rope-scaling", label: "RoPE scaling", check: Check::Rope },
    RadTextOpt { key: "r_prefill_interleave", flag: "--prefill-interleave", label: "Prefill interleave", check: Check::Interleave },
    RadTextOpt { key: "r_prefix_ttl", flag: "--prefix-cache-ttl", label: "Prefix cache TTL", check: num(0.0, INF, false, true) },
    RadTextOpt { key: "r_temp", flag: "--temp", label: "Temperature", check: num(0.0, INF, false, false) },
    RadTextOpt { key: "r_top_k", flag: "--top-k", label: "Top-k", check: num(0.0, INF, false, true) },
    RadTextOpt { key: "r_top_p", flag: "--top-p", label: "Top-p", check: num(0.0, 1.0, true, false) },
    RadTextOpt { key: "r_min_p", flag: "--min-p", label: "Min-p", check: num(0.0, 1.0, false, false) },
    RadTextOpt { key: "r_typical_p", flag: "--typical-p", label: "Typical-p", check: num(0.0, 1.0, true, false) },
    RadTextOpt { key: "r_presence", flag: "--presence-penalty", label: "Presence penalty", check: num(-2.0, 2.0, false, false) },
    RadTextOpt { key: "r_frequency", flag: "--frequency-penalty", label: "Frequency penalty", check: num(-2.0, 2.0, false, false) },
    RadTextOpt { key: "r_rep_pen", flag: "--repetition-penalty", label: "Repetition penalty", check: num(0.0, INF, true, false) },
    RadTextOpt { key: "r_rep_last_n", flag: "--repeat-last-n", label: "Penalty window", check: num(-1.0, INF, false, true) },
    RadTextOpt { key: "r_dry_mult", flag: "--dry-multiplier", label: "DRY multiplier", check: num(0.0, INF, false, false) },
    RadTextOpt { key: "r_dry_base", flag: "--dry-base", label: "DRY base", check: num(1.0, INF, false, false) },
    RadTextOpt { key: "r_dry_len", flag: "--dry-allowed-length", label: "DRY allowed length", check: num(1.0, INF, false, true) },
    RadTextOpt { key: "r_dry_last_n", flag: "--dry-penalty-last-n", label: "DRY window", check: num(-1.0, INF, false, true) },
    RadTextOpt { key: "r_xtc_prob", flag: "--xtc-probability", label: "XTC probability", check: num(0.0, 1.0, false, false) },
    RadTextOpt { key: "r_xtc_thresh", flag: "--xtc-threshold", label: "XTC threshold", check: num(0.0, 1.0, false, false) },
    RadTextOpt {
        key: "r_max_tokens",
        flag: "--default-max-tokens",
        label: "Default max tokens",
        check: Check::Num { min: 1.0, max: INF, open_min: false, int: true, auto: true },
    },
];

/// What is wrong with a free-text option's value, None when it is blank or valid.
pub fn text_opt_problem(o: &RadTextOpt, v: &str) -> Option<String> {
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    let bad = |why: &str| Some(format!("{} \"{v}\" {why}", o.label));
    match o.check {
        Check::Num { min, max, open_min, int, auto } => {
            if auto && v == "auto" {
                return None;
            }
            let n = if int { v.parse::<i64>().map(|n| n as f64).ok() } else { v.parse::<f64>().ok().filter(|f| f.is_finite()) };
            let Some(n) = n else {
                return bad(if int { "is not a whole number" } else { "is not a number" });
            };
            let low = if open_min { n <= min } else { n < min };
            if low || n > max {
                let lo = if open_min { format!("above {min}") } else { format!("at least {min}") };
                return bad(&if max.is_finite() { format!("must be {lo} and at most {max}") } else { format!("must be {lo}") });
            }
            None
        }
        Check::Rope => match rope_spec(v) {
            Ok(_) => None,
            Err(e) => bad(&e),
        },
        Check::Interleave => {
            if v == "auto" {
                return None;
            }
            if let Some(sh) = v.strip_prefix("auto:") {
                return match sh.parse::<f64>() {
                    Ok(x) if x > 0.0 && x <= 4.0 => None,
                    _ => bad("needs a share above 0 and at most 4 after auto:"),
                };
            }
            match v.parse::<i64>() {
                Ok(n) if (0..=64).contains(&n) => None,
                _ => bad("must be 0 to 64, auto or auto:SHARE"),
            }
        }
    }
}

/// `--rope-scaling`: Ok(None) for `none`, Ok(Some((factor, orig))) for `yarn:FACTOR[:ORIG]`.
pub fn rope_spec(v: &str) -> Result<Option<(f64, Option<i64>)>, String> {
    let v = v.trim();
    if v.is_empty() || v == "none" {
        return Ok(None);
    }
    let Some(rest) = v.strip_prefix("yarn:") else {
        return Err("must be yarn:FACTOR[:ORIG] or none".into());
    };
    let mut it = rest.splitn(2, ':');
    let f = it.next().unwrap_or("").parse::<f64>().ok().filter(|f| f.is_finite() && *f > 1.0);
    let Some(f) = f else {
        return Err("needs a FACTOR above 1 (yarn:2)".into());
    };
    match it.next() {
        None => Ok(Some((f, None))),
        Some(o) => match o.parse::<i64>() {
            Ok(n) if n > 0 => Ok(Some((f, Some(n)))),
            _ => Err("needs ORIG as a positive whole number (the trained length)".into()),
        },
    }
}

/// The history window's label for an option this module adds.
pub fn rad_opt_label(k: &str) -> Option<&'static str> {
    if let Some(o) = RAD_TEXT_OPTS.iter().find(|o| o.key == k) {
        return Some(o.label);
    }
    Some(match k {
        "r_cvec" => "Control vector",
        "r_state_dtype" => "State cache",
        "r_kv_resident" => "KV resident",
        "r_weight_prefetch" => "Weight prefetch",
        "r_disk_tier" => "Weights disk tier",
        "r_tp_attn_split" => "Attention split",
        "r_p2p" => "Peer access",
        _ => return None,
    })
}

pub const RAD_REASONING: [&str; 7] = ["(default)", "none", "minimal", "low", "medium", "high", "xhigh"];
pub const RAD_TP_WIRE: [&str; 2] = ["exact", "wht6"];
pub const RAD_STRATEGIES: [&str; 1] = ["Tensor parallel"];

/// Options that change how Radiance splits the card. A recorded KV pool is only reused for the
/// same values (context is not one: the pool is the same at any `--max-model-len` it can hold).
pub const RAD_POOL_KEYS: [&str; 11] = [
    "r_kv", "parallel", "r_batched", "r_headroom", "r_vision", "r_embed", "r_placement", "r_host_pool", "r_spec", "r_spec_ngram", "r_prefill_span",
];

/// The environment variable the API key travels in when the build takes `--api-key-env`.
pub const RAD_KEY_ENV: &str = "RADIANCE_API_KEY";

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

/// Does this radiance binary list `flag` in its --help? Remembered per binary and modification
/// time (an update replaces the file). An unknown option stops the server, so a binary that can't
/// be asked counts as not having it.
pub fn binary_has_flag(bin: &Path, flag: &str) -> bool {
    type Key = (PathBuf, Option<std::time::SystemTime>);
    static HELP: OnceLock<Mutex<HashMap<Key, String>>> = OnceLock::new();
    let key = (bin.to_path_buf(), std::fs::metadata(bin).and_then(|m| m.modified()).ok());
    let mut m = HELP.get_or_init(Default::default).lock().unwrap();
    let help = m.entry(key).or_insert_with(|| {
        let cmd = [bin.to_string_lossy().into_owned(), "--help".to_string()];
        run_capture(&cmd, &[], Duration::from_secs(5)).map(|x| x.1).unwrap_or_default()
    });
    help_has_flag(help, flag)
}

/// Radiance 2.0 and later read every setting from a YAML file (`radiance --config FILE`); the
/// command-line flags are gone from the server.
pub fn uses_config(bin: &Path) -> bool {
    binary_has_flag(bin, "--config")
}

/// Radiance 2.1.0 fixed draft trees with the i8 KV cache (#454: Gemma 4 + DFlash2 stopped on its
/// first request). The binary states no version without serving, so this reads its `--help`: 2.1.0
/// added `scheduler.stuck-exit` in the same release.
pub fn tree_i8_fixed(bin: &Path) -> bool {
    binary_has_flag(bin, "stuck-exit")
}

/// Where a pane's generated Radiance configuration is written (`tag`: "left", "left-plan", …).
pub fn config_path(tag: &str) -> PathBuf {
    dirs().data.join("radiance").join(format!("{tag}.yaml"))
}

/// Turn a 1.x-style flag line (`cmd[0]` the radiance binary, then its flags) into the YAML file
/// 2.0 reads, with `rad-yaml` from the same folder: Radiance's own parser, so every flag (Extra
/// args included) lands at the level it belongs to, and a flag the server would refuse is refused
/// here with the server's message. Writes it to `path` and returns the YAML text.
pub fn write_config(cmd: &[String], env: &[(String, String)], path: &Path) -> Result<String, String> {
    let yaml = rad_yaml(cmd, env)?;
    write_private(path, &yaml)?;
    Ok(yaml)
}

/// The YAML `rad-yaml` prints for a 1.x-style flag line (`cmd[0]` the radiance binary).
pub fn rad_yaml(cmd: &[String], env: &[(String, String)]) -> Result<String, String> {
    use std::process::{Command, Stdio};
    let bin = cmd.first().ok_or("empty command")?;
    let tool = tool_path(bin, "rad-yaml");
    let mut c = Command::new(&tool);
    c.args(&cmd[1..]).stdin(Stdio::null()).env_clear();
    for (k, v) in env {
        c.env(k, v);
    }
    let out = c.output().map_err(|e| format!("{}: {e}\n(rad-yaml ships with Radiance 2.0, next to the radiance binary)", tool.display()))?;
    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if !out.status.success() {
        return Err(if err.is_empty() { format!("rad-yaml failed ({})", out.status) } else { err });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Write a configuration readable by its owner only (an api-key given in Extra args lands in it).
pub fn write_private(path: &Path, text: &str) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    Ok(())
}

/// The configuration as log lines, indented under the command, with any `api-key:` hidden.
pub fn yaml_log_lines(yaml: &str) -> Vec<String> {
    yaml.lines()
        .map(|l| match l.split_once("api-key:") {
            Some((pre, _)) if pre.trim().is_empty() => format!("    {pre}api-key: ••••"),
            _ => format!("    {l}"),
        })
        .collect()
}

fn help_has_flag(help: &str, flag: &str) -> bool {
    help.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')).any(|w| w == flag)
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

/// What a model serves, where that isn't plain text generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RadKind {
    #[default]
    Generate,
    Embedding,
    Rerank,
    /// d1, clef, clef-flash, Mica, Laya: `decision.type` in the container
    Decision,
}

const RERANK_ARCH: &str = "Mistral3ForSequenceClassification";

fn kind_of_arch(arch: &str) -> RadKind {
    match arch {
        "nemotron35_embed" | "embgemma2" | "nomicbert" | "Mistral3Model" => RadKind::Embedding,
        RERANK_ARCH => RadKind::Rerank,
        "clef" | "laya" => RadKind::Decision,
        _ => RadKind::Generate,
    }
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
    /// A generator (image, video, audio, 3D shape): its plugin sizes the step itself.
    pub gen: Option<&'static GenStep>,
    /// Bytes of the n-gram embedding table (`*ple_ngram*` weights), 0 where the model has none.
    /// It is placed on its own (`--ngram-placement`), not with the other weights.
    pub ngram_bytes: f64,
    /// A drafter Radiance 2.0 verifies as a tree (Gemma 4 with DFlash2, RadArchProbe::tree_pays).
    /// A tree's acceptance cannot move an i8 KV cache's rows, so with speculation on and
    /// `kv-cache-dtype: i8` the engine stops on the first request; LLM Duo serves it at bf16.
    pub tree_drafter: bool,
    pub kind: RadKind,
    /// `decision.type` of a decision model
    pub decision: String,
    /// rad-info missing / failed: what went wrong (the model can still be served)
    pub note: String,
}

/// Whether the n-gram table sits on the cards: `vram` puts all of it on every card; Radiance's
/// own choice (auto) takes vram only when it fits beside the other weights; ram and disk keep it off.
/// `total` is the container, `head` the headroom, `card_min` the smallest card, in bytes.
pub fn ngram_on_card(choice: &str, total: f64, table: f64, tp: f64, head: f64, card_min: f64) -> bool {
    table > 0.0
        && match choice {
            "vram" => true,
            "(default)" | "auto" => (total - table) / tp + table + head <= card_min,
            _ => false,
        }
}

/// The weights one card holds: the container less an off-card n-gram table (an on-card one is
/// whole on every card) and less the host pool (`host_pool`: the model's total, shared by the
/// ranks), split over `tp` cards.
pub fn card_weight_bytes(total: f64, table: f64, table_on_card: bool, host_pool: f64, tp: f64) -> f64 {
    let tp = tp.max(1.0);
    let split = (total - table - host_pool).max(0.0) / tp;
    split + if table_on_card { table } else { 0.0 }
}

/// The n-gram table's size from `rad-info --planes '*ngram*'` (one `offset bytes name` a line).
pub fn ngram_plane_bytes(text: &str) -> f64 {
    text.lines()
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            let (_off, bytes, name) = (w.next()?.parse::<u64>().ok()?, w.next()?.parse::<u64>().ok()?, w.next()?);
            name.contains("ngram").then_some(bytes as f64)
        })
        .sum()
}

pub fn dir_bytes(d: &Path) -> f64 {
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
                    r.kind = kind_of_arch(&r.arch);
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
            let tool = info.to_string_lossy().into_owned();
            if let Ok((_, text)) = run_capture(&[tool.clone(), "--meta".into(), p.clone()], &env, Duration::from_secs(30)) {
                r.meta_text = text;
            }
            if let Ok((true, text)) = run_capture(&[tool, "--planes".into(), "*ngram*".into(), p], &env, Duration::from_secs(30)) {
                r.ngram_bytes = ngram_plane_bytes(&text);
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
            if kl == "decision.type" {
                self.decision = last.clone();
            }
            if kl == "architectures" && v.contains(RERANK_ARCH) {
                self.kind = RadKind::Rerank;
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
        // the test Radiance's gemma4 plugins make (arch/common/rad_gemma4_dflash.h: dflash2_probe)
        let meta = |k: &str| self.meta.iter().find(|m| m.0 == k).map(|m| m.1.split_whitespace().last().unwrap_or("").to_string());
        let draft_arch = self.meta.iter().find(|m| m.0 == "draft.architectures").map(|m| m.1.clone()).unwrap_or_default();
        let block = meta("draft.block_size").or_else(|| meta("draft.dflash_config.block_size")).and_then(|b| b.parse::<i64>().ok()).unwrap_or(0);
        self.tree_drafter = self.arch.to_lowercase().starts_with("gemma4") && draft_arch.contains("DFlash2") && block > 1;
        self.gen = gen_step(&self.arch);
        if !self.decision.is_empty() {
            self.kind = RadKind::Decision;
        } else if self.kind == RadKind::Generate {
            self.kind = kind_of_arch(&self.arch);
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
        match self.kind {
            RadKind::Generate => {
                if let Some(g) = self.gen {
                    parts.push(format!("{} generator", g.what));
                }
            }
            RadKind::Embedding => parts.push("embedding model: /v1/embeddings".into()),
            RadKind::Rerank => parts.push("rerank model: /rerank, /v1/rerank".into()),
            RadKind::Decision if self.decision.is_empty() => parts.push("decision model: /v1/systemone".into()),
            RadKind::Decision => parts.push(format!("decision model ({}): /v1/systemone", self.decision)),
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
        // 1.x: "serving on http://…"; 2.0: "model NAME serves  (/v1/completions; …)" once it is open
        serving: Regex::new(r"\bserving on\b|\bmodel \S+ serves\s").unwrap(),
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

/// What one token of context costs in the prefix cache's host tier (--prefix-cache-mib), in
/// bytes, from the "kv tiers:" line Radiance prints when the tier is on. Both forms name it:
/// "... copied to host memory at once (2.00 GiB, 8 entries, 64000 tokens of context) ..." and
/// "--prefix-cache-mib 512 holds 9000 tokens of context (29.00 KiB per 4-token block) ...".
pub fn tier_bytes_per_token(line: &str) -> Option<f64> {
    static R: OnceLock<(Regex, Regex)> = OnceLock::new();
    let (whole, block) = R.get_or_init(|| {
        (
            Regex::new(r"kv tiers:.*?\(([\d.]+)\s*(B|KiB|MiB|GiB|TiB),\s*\d+ entries,\s*(\d+) tokens of context\)").unwrap(),
            Regex::new(r"kv tiers:.*?\(([\d.]+)\s*(B|KiB|MiB|GiB|TiB) per (\d+)-token block\)").unwrap(),
        )
    });
    let c = block.captures(line).or_else(|| whole.captures(line))?;
    let unit = match &c[2] {
        "TiB" => GIB * 1024.0,
        u => bytes_of("1", u),
    };
    let bytes = c[1].parse::<f64>().ok()? * unit;
    let tokens: f64 = c[3].parse().ok()?;
    if bytes > 0.0 && tokens > 0.0 { Some(bytes / tokens) } else { None }
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
    if low.contains("kv tiers:") {
        return None; // the prefix cache's host tier, not the pool
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

/// Radiance's warning that one full-length sequence's cache leaves the experts less VRAM than the
/// host pool cannot hold (core/mem/vram_budget.cpp): (that cache, the experts' share, what they
/// need on the card), in bytes. The planner then refuses the experts it pushes off ("DID NOT FIT").
pub fn expert_shortfall(line: &str) -> Option<(f64, f64, f64)> {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| {
        Regex::new(
            r"sequence's cache \(([\d.]+)\s*(B|KiB|MiB|GiB)\) leaves the experts ([\d.]+)\s*(B|KiB|MiB|GiB), short of the ([\d.]+)\s*(B|KiB|MiB|GiB)",
        )
        .unwrap()
    });
    let c = re.captures(line)?;
    Some((bytes_of(&c[1], &c[2]), bytes_of(&c[3], &c[4]), bytes_of(&c[5], &c[6])))
}

/// A rank's "load: rank N, mover took …" line: placement passed for that card (it comes after the
/// planner's verdict, before any weight is copied).
pub fn is_rank_placed(line: &str) -> bool {
    line.contains("load: rank ") && line.contains("mover took")
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

// ------------------------------------------------------------- generators --
/// Whether a generator's requests run guidance (two branches a step: the prompt and a negative one)
/// at the plugin's default cfg.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guidance {
    Off,
    On,
    /// the default depends on the variant (a turbo / distilled one runs without)
    ByVariant,
}

/// A generator plugin's step, from Radiance 2.1's `arch/<arch>/<arch>.cpp`. Radiance sizes
/// `--max-num-batched-tokens` itself from the plugin (`RadArchProbe::gen_rows`) when it is not
/// given; a step carries every branch's latent tokens and, in the joint-step layout, each branch's
/// conditioning rows (`core/gen/driver.h`, `entry_rows`). A request whose step exceeds it is
/// refused: "an image with its conditioning needs N rows a step and a step carries M".
#[derive(Debug)]
pub struct GenStep {
    pub arch: &'static str,
    /// "image", "video", "audio", "3D shape"
    pub what: &'static str,
    /// the plugin's `gen_rows`, what Radiance takes when Batch tokens is not given (0: computed
    /// inside the plugin, not known here)
    pub auto_rows: i64,
    /// image models: pixels per latent token a side, the largest image's latent tokens, and the
    /// conditioning rows a branch adds to a step (0: conditioned in a pass of its own)
    pub latent_scale: i64,
    pub max_tokens: i64,
    pub cond_rows: i64,
    /// rows each reference picture of an edit adds a branch (flux3)
    pub ref_rows: i64,
    pub guidance: Guidance,
}

const fn img(arch: &'static str, latent_scale: i64, max_tokens: i64, cond_rows: i64, ref_rows: i64, guidance: Guidance) -> GenStep {
    GenStep {
        arch,
        what: "image",
        auto_rows: 2 * (max_tokens + cond_rows + ref_rows),
        latent_scale,
        max_tokens,
        cond_rows,
        ref_rows,
        guidance,
    }
}

const fn other(arch: &'static str, what: &'static str, auto_rows: i64) -> GenStep {
    GenStep { arch, what, auto_rows, latent_scale: 0, max_tokens: 0, cond_rows: 0, ref_rows: 0, guidance: Guidance::ByVariant }
}

pub const GEN_STEPS: [GenStep; 11] = [
    // kMaxImageTokens 16384, kCondRows 512; cfg 0 for turbo, 4.5 otherwise
    img("krea2", 16, 16384, 512, 0, Guidance::ByVariant),
    // kMaxImageTokens 9216, kCondRows 512; cfg 4.0 (1.0 for variant 2)
    img("anima", 16, 9216, 512, 0, Guidance::ByVariant),
    // kMaxImageTokens 36864, kCondRows 77 x 8; cfg 6.0
    img("sdxl", 8, 36864, 616, 0, Guidance::On),
    // kMaxImageTokens 16800, joint grid: the conditioning has its own pass; cfg 1.0
    img("qwenimage21", 16, 16800, 0, 0, Guidance::Off),
    // kMaxImageTokens 16384, kMaxText 1040, a 1024 x 1024 reference 4569 rows; cfg 3.5 (1.0 flow map)
    img("flux3", 16, 16384, 1040, 4569, Guidance::ByVariant),
    other("ltx2", "video", 0),
    other("minimaxh3", "video", 0),
    other("hunyuan3d2", "3D shape", 0),
    other("minimaxmusic3", "audio", 2048),
    other("fishomni", "audio", 0),
    other("breeze", "audio", 0),
];

pub fn gen_step(arch: &str) -> Option<&'static GenStep> {
    GEN_STEPS.iter().find(|g| g.arch == arch)
}

impl GenStep {
    pub fn is_image(&self) -> bool {
        self.max_tokens > 0
    }
    /// Rows a step needs for the largest image with the longest prompt, one branch / two.
    pub fn rows(&self, branches: i64) -> i64 {
        branches * (self.max_tokens + self.cond_rows)
    }
    /// The largest square image (pixels a side, about: the size multiple rounds it down) whose
    /// step fits `batched` rows with the longest prompt.
    pub fn side_fitting(&self, batched: i64, branches: i64) -> i64 {
        let t = (batched / branches.max(1) - self.cond_rows).clamp(0, self.max_tokens);
        ((t as f64).sqrt().floor() as i64) * self.latent_scale
    }
    /// The largest image's side, if it were square.
    pub fn max_side(&self) -> i64 {
        ((self.max_tokens as f64).sqrt().floor() as i64) * self.latent_scale
    }
}

/// What Batch tokens means for a generator: (red, text) when requests will be refused, None when
/// it holds the largest request. `auto`: Batch tokens is left to Radiance.
pub fn gen_batch_problem(g: &GenStep, batched: i64, auto: bool) -> Option<(bool, String)> {
    if auto {
        return None;
    }
    if !g.is_image() {
        return Some((
            false,
            format!(
                "a {} generator sizes its step itself{}; tick \"Radiance sizes it\" so Batch tokens, Max seqs and Context are left to it",
                g.what,
                if g.auto_rows > 0 { format!(" ({} rows)", fmt_thousands(g.auto_rows)) } else { String::new() }
            ),
        ));
    }
    let (one, two) = (g.rows(1), g.rows(2));
    let side = g.max_side();
    if batched < one {
        let fit = g.side_fitting(batched, 1);
        return Some((
            true,
            format!(
                "Batch tokens {} is below the {} rows a step needs for the largest image (about {side}x{side}) with a full prompt: larger images and long prompts are refused when requested (about {fit}x{fit} fits). Set Batch tokens to {} ({} without guidance), or tick \"Radiance sizes it\"",
                fmt_thousands(batched),
                fmt_thousands(one),
                fmt_thousands(two),
                fmt_thousands(one)
            ),
        ));
    }
    if batched < two && g.guidance != Guidance::Off {
        let fit = g.side_fitting(batched, 2);
        return Some((
            g.guidance == Guidance::On,
            format!(
                "with guidance (a negative prompt and cfg{}) a step carries both branches: the largest image then needs {} rows, and Batch tokens {} holds about {fit}x{fit}. Set {} or tick \"Radiance sizes it\" if you use guidance",
                if g.guidance == Guidance::On { ", this model's default" } else { "; this model's turbo variants run without" },
                fmt_thousands(two),
                fmt_thousands(batched),
                fmt_thousands(two)
            ),
        ));
    }
    None
}

/// A container's architecture from `rad-info` (its summary only), for checks that need no more.
pub fn rad_arch(path: &Path) -> Option<String> {
    if path.is_dir() {
        return None;
    }
    let (bin, home) = tools();
    let info = tool_path(&bin, "rad-info");
    let mut env = vec![];
    if !home.trim().is_empty() {
        env.push(("RADIANCE_HOME".to_string(), expanduser(&home).to_string_lossy().into_owned()));
    }
    let (_, text) = run_capture(&[info.to_string_lossy().into_owned(), path.to_string_lossy().into_owned()], &env, Duration::from_secs(30)).ok()?;
    text.lines().take(20).filter_map(kv_line).find(|(k, _)| k == "architecture").map(|(_, v)| v)
}

// ---------------------------------------------------------------- failure --
pub fn rad_failure_hint(lines: &[String]) -> String {
    let text = lines.join("\n");
    let m = |p: &str| Regex::new(p).unwrap().is_match(&text);
    // the last E line of this refusal only says the fix was named above, so quote the one above
    if let Some(c) = Regex::new(r"incompatible configuration: (.+)").unwrap().captures(&text) {
        return format!(
            "Radiance refuses to serve on this machine's configuration:\n{}\n\nFix that. --proceed-with-incompatible-configuration in Extra args starts it anyway, unsupported.",
            c[1].trim()
        );
    }
    if m(r"--spec-ngram drafts by prompt lookup alone") {
        return "This container has its own draft head, so Prompt lookup can't take over.\nSet Prompt lookup to 0 and use Spec tokens: lookup then fills in wherever the head declines.".into();
    }
    if m(r"--spec-ngram \d+ sets the speculative window") {
        return "--spec-ngram and --num-speculative-tokens disagree.\nUse the Prompt lookup field, or remove one of them from Extra args.".into();
    }
    if m(r"--api-key-env: \S+ is unset or empty") {
        return "The variable named by --api-key-env is unset or empty.\nSet it in Settings (Radiance environment) or put the key in Settings instead.".into();
    }
    if m(r"resolve only to a reference implementation|libref") {
        return "Some ops have no kernel for this card (only the reference implementation).\nCheck Radiance home in Settings (it must hold kernels/, architectures/ and quantizers/), or rebuild Radiance for this GPU.".into();
    }
    if m(r"DID NOT FIT|did not fit the KV pool") {
        return "The memory plan did not fit.\nLower Context or Max seqs, use the i8 KV cache, lower Batch tokens (a smaller activation arena), put the vision tower in RAM, or give it a Host pool (MoE models).".into();
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
    if m(r"cannot be read with direct I/O[^\n]*n-gram table") {
        return "N-gram table → disk reads the table with direct I/O, and the filesystem holding the container refuses it.\nMove the container to an ext4, XFS or btrfs drive, or set N-gram table to ram if RAM allows.".into();
    }
    if m(r"n-gram copy .*--ngram-placement ram") && m(r"the host pool is full") {
        return "Radiance copied the n-gram table into pinned RAM, and the Host pool was left too little pinned memory for the experts that do not fit the cards.\nSet N-gram table to disk (read from the container; an NVMe keeps decoding fast), or lower Host pool if you want the table in RAM.".into();
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
    if m(r"(?i)kv-cache-dtype.*(?:refus|not offered|unsupported|no longer served|takes bf16)|refuses? (?:fp8_v4|k6_v4)") {
        return "Radiance does not serve that KV cache type for this model. Use i8 or bf16 (fp8 is GLM-5.3-Flash only).".into();
    }
    if let Some(c) = Regex::new(r"'([^']+)' is stored as ([^,]+?), which this build does not serve").unwrap().captures(&text) {
        let drafter = c[1].starts_with("dflash") || c[1].starts_with("mtp") || c[1].contains("draft");
        return format!(
            "Radiance 2.0 no longer serves {} weights ('{}'): block FP8, MXFP4, M2XFP4, NVFP4 and ParoQuant containers are refused.\n{}Convert the checkpoint again with an int8 recipe (the IQ8R ones), or keep Radiance 1.17 for this container.",
            &c[2], &c[1],
            if drafter { "It is the drafter's weight: Spec tokens = 0 serves the model without it.\n" } else { "" }
        );
    }
    if m(r"(?i)takes its settings from a YAML file") {
        return "This Radiance (2.0+) only reads a YAML configuration, and LLM Duo could not hand it one.\nCheck that rad-yaml is next to the radiance binary.".into();
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
    // a plugin states why on a line of its own; the E lines after it only point back up
    if let Some(c) = Regex::new(r"(?m)^radiance: (.{8,})$").unwrap().captures_iter(&text).last() {
        return format!("Radiance reported: {}", c[1].trim());
    }
    let pointer = Regex::new(r"^startup stopped:|rad_arch_declare returned").unwrap();
    let e_lines = Regex::new(r"(?m)^\s*(?:\[[^\]]*\]\s*)?E\s+(.{8,})$").unwrap();
    if let Some(c) = e_lines.captures_iter(&text).filter(|c| !pointer.is_match(c[1].trim())).last() {
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

/// Recipes that name data files relative to the Radiance repository (`vq=data/codebooks/…`,
/// `tcq_lut=data/codebooks/…`): any rule option whose value starts with `data/`.
pub fn recipe_needs_repo_cwd(text: &str) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"(?m)^\s*[^#\s].*\s[A-Za-z0-9_]+=(?:\./)?data/").unwrap());
    re.is_match(text)
}

/// What a recipe's usage line (`#   rad-convert <…> …` in its header) says to convert from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeSource {
    /// a folder: "<… pipeline directory>", "<… dir>"
    Dir,
    /// one weights file: "<checkpoint>.safetensors", "X-Q4_K_M.gguf"
    File,
}

/// The source kind the recipe's own usage line names; None when it doesn't say
/// (e.g. "<flint checkpoint>") or has no usage line.
pub fn recipe_source(text: &str) -> Option<RecipeSource> {
    let line = text.lines().take_while(|l| l.trim_start().starts_with('#') || l.trim().is_empty()).find_map(|l| {
        let l = l.trim_start().trim_start_matches('#').trim_start();
        l.strip_prefix("rad-convert ")
    })?;
    // the input is the first argument; a <placeholder> may hold spaces
    let arg = if line.starts_with('<') {
        let end = line.find('>')?;
        let rest = &line[end + 1..];
        format!("{}{}", &line[..=end], rest.split_whitespace().next().filter(|_| !rest.starts_with(' ')).unwrap_or(""))
    } else {
        line.split_whitespace().next()?.to_string()
    };
    let lc = arg.to_lowercase();
    if lc.ends_with(".safetensors") || lc.ends_with(".gguf") {
        Some(RecipeSource::File)
    } else if lc.ends_with("directory>") || lc.ends_with(" dir>") || lc.ends_with("folder>") {
        Some(RecipeSource::Dir)
    } else {
        None
    }
}

/// A recipe rule reads `$CALIB` (GPTQ's calibration Hessians).
pub fn recipe_needs_calib(text: &str) -> bool {
    recipe_calib_values(text).next().is_some()
}

/// The `calib=` values of a recipe's rules (comments skipped): `$CALIB`, or a fixed folder.
fn recipe_calib_values(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .flat_map(|l| l.split_whitespace())
        .filter_map(|w| w.strip_prefix("calib="))
        .filter(|v| !v.is_empty())
}

/// The fixed folder a recipe's GPTQ rules read Hessians from (`calib=/calib`, as recipes taken
/// from published containers have), instead of `$CALIB`. None when every rule uses a variable.
pub fn recipe_fixed_calib(text: &str) -> Option<String> {
    recipe_calib_values(text).find(|v| !v.starts_with('$')).map(String::from)
}

/// The recipe with every fixed `calib=` folder replaced by `$CALIB`, so it reads the folder
/// given in CALIB (LLM Duo's Calibration data). Comments and everything else stay as they are.
pub fn recipe_calib_to_env(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for l in text.split_inclusive('\n') {
        if l.trim_start().starts_with('#') {
            out.push_str(l);
            continue;
        }
        let mut rest = l;
        while let Some(i) = rest.find("calib=") {
            let at_word = i == 0 || rest[..i].ends_with(char::is_whitespace);
            out.push_str(&rest[..i + 6]);
            rest = &rest[i + 6..];
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            if at_word && end > 0 && !rest.starts_with('$') {
                out.push_str("$CALIB");
                rest = &rest[end..];
            }
        }
        out.push_str(rest);
    }
    out
}

/// The repository a recipes folder belongs to: `<repo>/data/recipes`, with the codebooks beside it.
pub fn repo_of_recipe(recipe: &Path) -> Option<PathBuf> {
    let data = recipe.parent()?.parent()?;
    data.join("codebooks").is_dir().then(|| data.parent().map(Path::to_path_buf)).flatten()
}

/// The decision format of a checkpoint folder that needs telling at convert time.
pub fn decision_kind(dir: &Path) -> Option<&'static str> {
    if dir.join("joint_head_config.json").is_file() {
        return Some("clef");
    }
    if is_laya_dir(dir) {
        return Some("laya");
    }
    None
}

/// A Laya release keeps its parts in folders and may have no config.json of its own.
pub fn is_laya_dir(dir: &Path) -> bool {
    dir.join("rl_agent_config.json").is_file() && dir.join("encoder/config.json").is_file()
}

/// Laya checkpoint folders under `root`.
pub fn scan_laya(root: &str) -> Vec<PathBuf> {
    fn walk(d: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth > 8 {
            return;
        }
        if is_laya_dir(d) {
            out.push(d.to_path_buf());
            return;
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.filter_map(|e| e.ok()) {
            let hidden = e.file_name().to_string_lossy().starts_with('.');
            if !hidden && e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                walk(&e.path(), out, depth + 1);
            }
        }
    }
    let mut out = vec![];
    walk(Path::new(root), &mut out, 0);
    out.sort();
    out
}

/// What a checkpoint folder is: its family and whether it is already quantised.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CkptKind {
    /// config.json's model_type (top level, else text_config's)
    pub model_type: String,
    pub architectures: Vec<String>,
    /// Some(true): quantization_config present or quantised tensors; Some(false): bf16/f16/f32
    pub quantised: Option<bool>,
    /// a mixture of experts by its config (Gemma 4 26B-A4B: enable_moe_block / num_experts)
    pub moe: bool,
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
    k.moe = [&cfg, &tc].iter().any(|c| {
        c.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false) || c.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) > 0
    });
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
            // the 26B-A4B is served by arch/gemma4 and its recipe needs no --set keys
            if self.moe { "gemma4moe" } else { "gemma4" }
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
        // dense only: int4 (w4a8, a4), int5 (w5) and int8 (i8g) trunks, as keptn's commands set it
        "gemma4" if has("i4") || has("u4") || has("u5") || has("i8") => Some((
            vec!["quantization_config.quant_method=compressed-tensors".into()],
            "select gemma4_bf16, the plugin that serves Gemma 4's quantised dense trunks".into(),
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
    fn recipe_sources() {
        let h = |u: &str| format!("# x.recipe -- about\n#\n#   rad-convert {u}\n#       -o x.rad\n\nvae.* cast dtype=bf16\n");
        assert_eq!(recipe_source(&h("<Krea-2-Raw or -Turbo pipeline directory> --recipe data/recipes/krea-2-bf16.recipe \\")), Some(RecipeSource::Dir));
        assert_eq!(recipe_source(&h("<DeepSeek-V4.1-Flash-1017 dir> --add engram-tables.safetensors \\")), Some(RecipeSource::Dir));
        assert_eq!(recipe_source(&h("<checkpoint>.safetensors --tokenizer <CLIP tokenizer.json> \\")), Some(RecipeSource::File));
        assert_eq!(recipe_source(&h("<Anima>/split_files/diffusion_models/anima-aesthetic-v1.1.safetensors \\")), Some(RecipeSource::File));
        assert_eq!(recipe_source(&h("Hy-Memory-30B-A3B-Q4_K_M.gguf \\")), Some(RecipeSource::File));
        assert_eq!(recipe_source(&h("<flint checkpoint> --add data/flint-zero-bias.safetensors \\")), None);
        assert_eq!(recipe_source("blk.*.attn_norm.weight cast dtype=f32\n"), None);
    }

    #[test]
    fn tier_lines() {
        let a = tier_bytes_per_token("W kv tiers: --prefix-cache-mib 512 holds 9000 tokens of context (29.00 KiB per 4-token block), less than one --max-model-len 16384 conversation").unwrap();
        assert!((a - 29.0 * 1024.0 / 4.0).abs() < 1e-6, "{a}");
        let b = tier_bytes_per_token("I kv tiers: a finished conversation's KV is copied to host memory at once (2.00 GiB, 8 entries, 64000 tokens of context) and its VRAM is given up").unwrap();
        assert!((b - 2.0 * GIB / 64000.0).abs() < 1e-6, "{b}");
        assert_eq!(tier_bytes_per_token("I kv tiers: off (no --prefix-cache-mib); a finished conversation's KV stays in VRAM until the cache evicts it"), None);
        assert_eq!(tier_bytes_per_token("I kv cache 11.59 GiB = 262,160 tokens of context"), None);
    }

    #[test]
    fn log_lines() {
        assert_eq!(kv_tokens_of("I kv cache 11.59 GiB = 262,160 tokens of context"), Some(262160));
        assert_eq!(kv_tokens_of("I   kv cache: fp8, 40.00 KiB a token, 262160 tokens of context (1 x 262144)"), Some(262160));
        assert_eq!(kv_tokens_of("I kv: the pool backs 100% of the worst case, 1.0 full-length sessions"), None);
        assert_eq!(kv_tokens_of("I prefill chunks of at most 2048 tokens"), None);
        assert_eq!(kv_tokens_of("I the cache holds 117,000 tokens of context"), Some(117000));
        assert_eq!(kv_tokens_of("E KV group 'kv_attn': 1 x 262144 tokens did not fit the KV pool (DID NOT FIT)"), None);
        assert_eq!(kv_tokens_of("W kv tiers: --prefix-cache-mib 512 holds 9000 tokens of context (1 MiB per 256-token block), less than one"), None);
        assert_eq!(kv_tokens_of("I kv tiers: a finished conversation's KV is copied to host memory at once (2.00 GiB, 8 entries, 64000 tokens of context) and"), None);
        let b = rre().backs.captures("I kv: the pool backs 87% of the worst case (2 x 131072 tokens): 1.7 full-length sessions").unwrap();
        assert_eq!(&b[1], "87");
        assert_eq!(&b[2], "1.7");
        let s = rre().settled.captures("I card 0 settled at 1.21 GiB free (the budget read 1.30)").unwrap();
        assert_eq!(&s[2], "1.21");
        assert!(rre().serving.is_match("I serving on http://0.0.0.0:8080 (chat, completions)"));
        assert!(rre().serving.is_match("I model q38-27b serves  (/v1/completions; /v1/chat/completions with tools and reasoning)"));
        assert!(!rre().serving.is_match("I server: listening on http://127.0.0.1:8080 (16 threads, up to 64)"));
        // Radiance 2.2, Gemma 4 26B-A4B w8a8 on one 32 GB card
        let (kv, exp, need) = expert_shortfall("W one --max-model-len sequence's cache (5.35 GiB) leaves the experts 19.26 GiB, short of the 28.85 GiB the pinned host pool cannot hold: raise --host-pool-mib or lower --max-model-len").unwrap();
        assert!((kv / GIB - 5.35).abs() < 1e-9 && (exp / GIB - 19.26).abs() < 1e-9 && (need / GIB - 28.85).abs() < 1e-9);
        assert!(expert_shortfall("I kv: the pool backs 100% of the worst case").is_none());
        assert!(is_rank_placed("I[1] load: rank 1, mover took 18.14 GiB, weights pool 173.75 KiB free of 18.14 GiB"));
        assert!(is_rank_placed("I load: rank 0, mover took 23.89 GiB, weights pool 0 B free of 24.83 GiB"));
        assert!(!is_rank_placed("I model Gemma-4: loading on card(s) 0"));
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
    fn text_opts() {
        let o = |k: &str| RAD_TEXT_OPTS.iter().find(|o| o.key == k).unwrap();
        assert!(text_opt_problem(o("r_temp"), "").is_none());
        assert!(text_opt_problem(o("r_temp"), "0.7").is_none());
        assert!(text_opt_problem(o("r_temp"), "-1").is_some());
        assert!(text_opt_problem(o("r_temp"), "warm").is_some());
        assert!(text_opt_problem(o("r_top_p"), "0").is_some(), "top_p is above 0");
        assert!(text_opt_problem(o("r_top_p"), "1").is_none());
        assert!(text_opt_problem(o("r_top_k"), "20.5").is_some());
        assert!(text_opt_problem(o("r_presence"), "2.5").is_some());
        assert!(text_opt_problem(o("r_rep_last_n"), "-1").is_none());
        assert!(text_opt_problem(o("r_max_tokens"), "auto").is_none());
        assert!(text_opt_problem(o("r_max_tokens"), "0").is_some());
        assert!(text_opt_problem(o("r_prefill_interleave"), "auto:1").is_none());
        assert!(text_opt_problem(o("r_prefill_interleave"), "auto:5").is_some());
        assert!(text_opt_problem(o("r_prefill_interleave"), "65").is_some());
        assert!(text_opt_problem(o("r_rope"), "yarn:2").is_none());
        assert!(text_opt_problem(o("r_rope"), "yarn:1").is_some());
        assert_eq!(rope_spec("yarn:2:131072"), Ok(Some((2.0, Some(131072)))));
        assert_eq!(rope_spec("none"), Ok(None));
        assert!(rope_spec("linear:2").is_err());
        for o in RAD_TEXT_OPTS.iter() {
            assert_eq!(rad_opt_label(o.key), Some(o.label));
        }
    }

    #[test]
    fn gen_steps() {
        let k = gen_step("krea2").unwrap();
        // what Radiance gives Batch tokens itself: 2 x (16384 + 512)
        assert_eq!(k.auto_rows, 33792);
        assert_eq!((k.rows(1), k.rows(2)), (16896, 33792));
        assert_eq!(k.max_side(), 2048);
        // the refused 2048 x 2048 request at 16384: red, and it names the numbers to set
        let (red, t) = gen_batch_problem(k, 16384, false).unwrap();
        assert!(red && t.contains("16,896") && t.contains("33,792"), "{t}");
        // 17408 holds it without guidance; turbo runs without, so only a note
        let (red, _) = gen_batch_problem(k, 17408, false).unwrap();
        assert!(!red);
        assert!(gen_batch_problem(k, 33792, false).is_none());
        assert!(gen_batch_problem(k, 2048, true).is_none(), "Radiance sizes it");
        assert_eq!(gen_step("qwenimage21").unwrap().rows(1), 16800);
        assert!(gen_batch_problem(gen_step("qwenimage21").unwrap(), 16896, false).is_none(), "cfg 1.0: no guidance");
        assert_eq!(gen_step("flux3").unwrap().auto_rows, 43986);
        assert!(gen_batch_problem(gen_step("ltx2").unwrap(), 4096, false).is_some());
        assert!(gen_step("qwen35").is_none());
    }

    #[test]
    fn card_weights() {
        const G: f64 = 1024.0 * 1024.0 * 1024.0;
        // Flash-Next IQR-V4 at tp 2: 113.32 GiB, a 47.68 GiB n-gram table on disk, 24 GiB host pool
        let (total, table) = (113.32 * G, 47.68 * G);
        let card = 31.9 * G;
        assert!(!ngram_on_card("disk", total, table, 2.0, 0.1 * G, card));
        assert!(!ngram_on_card("(default)", total, table, 2.0, 0.1 * G, card), "auto cannot fit it on a card");
        let w = card_weight_bytes(total, table, false, 24.0 * G, 2.0);
        assert!((w / G - 20.82).abs() < 0.01, "{}", w / G);
        // a plain model: everything split, nothing off the cards
        assert_eq!(card_weight_bytes(30.0 * G, 0.0, false, 0.0, 2.0), 15.0 * G);
        // vram: the whole table on every card
        assert_eq!(card_weight_bytes(12.0 * G, 2.0 * G, true, 0.0, 2.0), 7.0 * G);
    }

    #[test]
    fn ngram_table_size() {
        // rad-info --planes '*ngram*' on qwen3.8-next-flash-iqr-v4.rad (the table and its scale)
        let t = "75370496 51200245760 blk.1.ple_ngram.weight\n51275616256 2 blk.1.ple_ngram.weight\n";
        assert_eq!(ngram_plane_bytes(t), 51200245762.0);
        assert_eq!(ngram_plane_bytes(""), 0.0);
        assert_eq!(ngram_plane_bytes("rad-info: no weight matches\n"), 0.0);
    }

    #[test]
    fn hint_ngram() {
        // from a Flash-Next load whose n-gram table went to RAM (auto) and starved the host pool
        let pinned = vec![
            "  n-gram copy              -     47.68 GiB   (--ngram-placement ram, pinned RAM the card reads, filled at load)".to_string(),
            "    2935 x blk.*.ffn_gate_up_exps.*.weight (3.40 GiB): the host pool is full and no --weights-disk-tier was given".to_string(),
            "E startup stopped: out of memory (the reason is said above the startup report)".to_string(),
        ];
        let h = rad_failure_hint(&pinned);
        assert!(h.contains("N-gram table to disk"), "{h}");
        // the same out of memory without the n-gram copy keeps the old hint
        assert!(rad_failure_hint(&pinned[1..]).contains("holding VRAM"));
        let dio = vec!["libavx: /mnt/x.rad cannot be read with direct I/O (Invalid argument); the n-gram table is read that way. A local filesystem".to_string()];
        assert!(rad_failure_hint(&dio).contains("ext4, XFS or btrfs"));
    }

    #[test]
    fn hint() {
        let l = vec!["I x".to_string(), "E unknown option '--foo'".to_string()];
        assert!(rad_failure_hint(&l).contains("--foo"));
        let f = vec!["radiance: 'dflash.fc.weight' is stored as block fp8, which this build does not serve. Convert the checkpoint again".to_string()];
        assert!(rad_failure_hint(&f).contains("Spec tokens = 0"));
        let cwsr = vec![
            "E incompatible configuration: cwsr-off: amdgpu cwsr_enable is 0 -- the card hangs. Fix: amdgpu.cwsr_enable=1 as a module option".to_string(),
            "E radiance does not start on a configuration it does not support: the fix for each is said above.".to_string(),
        ];
        assert!(rad_failure_hint(&cwsr).contains("amdgpu.cwsr_enable=1"));
        let started = vec!["W incompatible configuration, started anyway: cwsr-off: x. Fix: y".to_string()];
        assert_eq!(rad_failure_hint(&started), "");
        let head = vec!["E --spec-ngram drafts by prompt lookup alone, and this container declares the draft head 'dflash'; serve it".to_string()];
        assert!(rad_failure_hint(&head).contains("Prompt lookup"));
        let window = vec!["E --spec-ngram 5 sets the speculative window to 5, and --num-speculative-tokens asks for 4; give one".to_string()];
        assert!(rad_failure_hint(&window).contains("disagree"));
        let env = vec!["E --api-key-env: RADIANCE_API_KEY is unset or empty; refusing to serve keyless".to_string()];
        assert!(rad_failure_hint(&env).contains("--api-key-env"));
        // from a live 1.15.0 start: a block head's window, refused inside the plugin
        let block = vec![
            "radiance: DFlash2's convolution masks the block position with block_size - 1, so the query block has to be a power of two. This serve asks for 5 draft tokens, which makes it 6; the checkpoint was trained at 8 (7 draft tokens).".to_string(),
            "E qwen35_fp8: rad_arch_declare returned unsupported".to_string(),
            "I STARTUP REPORT -- radiance 1.15.0 -- STARTUP STOPPED; this is what it had resolved when it did".to_string(),
            "E startup stopped: unsupported (the reason is said above the startup report)".to_string(),
        ];
        assert!(rad_failure_hint(&block).contains("7 draft tokens"));
        assert_eq!(rad_failure_hint(&block[1..]), "", "nothing but pointers: no reason to show");
    }

    #[test]
    fn live_log() {
        // from a live 1.15.0 start of a Gemma 4 container at context 16384
        let log = [
            "I activation arena: not lent -- the 'gemma4_bf16' plugin does not declare at a smaller step size (RadArchProbe::shape_probe_ok)",
            "I kv: the pool backs 100% of the worst case (1 sequences all at 16384 tokens), 1.0 full-length sessions; a larger --host-pool-mib gives the cache more",
            "  elastic             10.53 GiB   the experts keep what the host pool cannot hold; the cache takes the rest",
            "    kv cache           1.81 GiB   17% of it -- backs 1 sequences all at 16384 tokens at once, which is the worst case and",
            "    unusable           8.73 GiB   neither side could take it: both are at their ceiling",
            "I kv tiers: off (no --prefix-cache-mib); a finished conversation's KV stays in VRAM until the cache evicts it",
            "I kv: the cache can grow to 1.81 GiB a card -- 16384 tokens of context, 1.0 sessions at 16384 -- with every block it lends the experts taken back; 0 B of that is per-sequence state, and nothing is lent: it is all the cache's",
        ];
        let mut b = Budget::default();
        for l in log {
            b.line(l);
        }
        assert_eq!(b.kv_tokens, Some(16384));
        let cap = b.capacity().unwrap();
        assert!(cap > 90_000 && cap < 100_000, "{cap}");
        assert!(rre().arena.captures(log[0]).is_none());
        let c = rre().backs.captures(log[1]).unwrap();
        assert_eq!((&c[1], &c[2]), ("100", "1.0"));
        let s = rre().settled.captures("I card 0 settled at 8.87 GiB free against 141.43 MiB (the 96.00 MiB --gpu-headroom-mib and").unwrap();
        assert_eq!((&s[1], &s[2], &s[3]), ("0", "8.87", "GiB"));
        assert!(rre().serving.is_match("I serving on http://127.0.0.1:18937  (/v1/completions; /v1/chat/completions with tools and reasoning; a bearer key is required)"));
    }

    #[test]
    fn yaml_masked() {
        let y = "server:\n  port: 8080\n  api-key: hunter2\nmodels:\n  m:\n    model: /m.rad\n";
        let l = yaml_log_lines(y);
        assert_eq!(l[2], "      api-key: ••••");
        assert_eq!(l[1], "      port: 8080");
        assert!(!l.join("\n").contains("hunter2"));
    }

    #[test]
    fn help_flags() {
        let h = "  --spec-ngram K   drafts\n  --api-key KEY  the key\n  --spec-ngram-min N  shortest";
        assert!(help_has_flag(h, "--spec-ngram"));
        // 2.1.0's help lists scheduler.stuck-exit (the #454 fix is in the same release)
        assert!(help_has_flag("scheduler:\n  stuck-exit: DURATION   a device stuck this long", "stuck-exit"));
        assert!(!help_has_flag("scheduler:\n  load-timeout: DURATION", "stuck-exit"));
        assert!(!help_has_flag(h, "--api-key-env"), "--api-key must not read as --api-key-env");
        assert!(!help_has_flag(h, "--spec"));
        use std::os::unix::fs::PermissionsExt;
        let bin = std::env::temp_dir().join(format!("llmduo-help-{}", std::process::id()));
        std::fs::write(&bin, "#!/bin/sh\necho '  --api-key-env NAME  the key'\n").unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(binary_has_flag(&bin, "--api-key-env"));
        assert!(!binary_has_flag(&bin, "--allow-host"));
        let _ = std::fs::remove_file(&bin);
        assert!(!binary_has_flag(&bin, "--api-key-env"), "a binary that can't be asked has no flags");
    }

    #[test]
    fn tree_drafter() {
        let meta = "  draft.architectures                           DFlash2DraftModel\n  draft.block_size                              8\n";
        let mut g = RadInfo { summary_text: "  architecture  gemma4\n".into(), meta_text: meta.into(), ..Default::default() };
        g.parse();
        assert!(g.tree_drafter, "Gemma 4 with DFlash2 is verified as a tree");
        let mut q = RadInfo { summary_text: "  architecture  qwen35\n".into(), meta_text: meta.into(), ..Default::default() };
        q.parse();
        assert!(!q.tree_drafter, "Qwen's DFlash2 is a chain");
        let v1 = "  draft.architectures  DFlashDraftModel\n  draft.block_size  16\n";
        let mut d = RadInfo { summary_text: "  architecture  gemma4\n".into(), meta_text: v1.into(), ..Default::default() };
        d.parse();
        assert!(!d.tree_drafter, "z-lab DFlash is a chain");
    }

    #[test]
    fn model_kinds() {
        let mut r = RadInfo { summary_text: "  architecture  laya\n".into(), meta_text: "  decision.type  str  laya\n".into(), ..Default::default() };
        r.parse();
        assert_eq!((r.kind, r.decision.as_str()), (RadKind::Decision, "laya"));
        assert!(r.summary().contains("decision model (laya)"));
        // a decision model on an ordinary trunk (Mica on qwen35, d1 on lfm2)
        let mut r = RadInfo { summary_text: "  architecture  qwen35_bf16\n".into(), meta_text: "  decision.type  str  mica\n".into(), ..Default::default() };
        r.parse();
        assert_eq!(r.kind, RadKind::Decision);
        let mut r = RadInfo { summary_text: "  architecture  nemotron35_embed\n".into(), meta_text: format!("  architectures  str  {RERANK_ARCH}\n"), ..Default::default() };
        r.parse();
        assert_eq!(r.kind, RadKind::Rerank);
        let mut r = RadInfo { summary_text: "  architecture  nemotron35_embed\n".into(), ..Default::default() };
        r.parse();
        assert_eq!(r.kind, RadKind::Embedding);
        let mut r = RadInfo { summary_text: "  architecture  gemma4\n".into(), ..Default::default() };
        r.parse();
        assert_eq!(r.kind, RadKind::Generate);
    }

    #[test]
    fn recipe_paths() {
        let iq = "# run from the repository root (vq=data/codebooks/x.txt in a comment does not count)\n\
                  blk.0.ssm_inz.weight gptq vq=data/codebooks/iqr-vq3-4x4096.txt group=128 calib=$CALIB\n";
        assert!(recipe_needs_repo_cwd(iq) && recipe_needs_calib(iq));
        // IQR-V2..V4 name their trellis tables with tcq_lut=
        let tcq = "token_embd.weight gptq codes=u3 tcq=12 tcq_rows=256 tcq_lut=data/codebooks/iqr-tcq-L12-k3.txt group=128\n";
        assert!(recipe_needs_repo_cwd(tcq));
        let plain = "#   CALIB=<dir> rad-convert uses vq=data/codebooks/a.txt and $CALIB here\nblk.*.weight rtn codes=i8 group=128\n";
        assert!(!recipe_needs_repo_cwd(plain) && !recipe_needs_calib(plain));
    }

    #[test]
    fn decision_folders() {
        let root = std::env::temp_dir().join(format!("llmduo-dec-{}", std::process::id()));
        let (clef, laya, plain) = (root.join("clef"), root.join("nest/laya"), root.join("plain"));
        std::fs::create_dir_all(&clef).unwrap();
        std::fs::create_dir_all(laya.join("encoder")).unwrap();
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(clef.join("joint_head_config.json"), "{}").unwrap();
        std::fs::write(laya.join("rl_agent_config.json"), "{}").unwrap();
        std::fs::write(laya.join("encoder/config.json"), "{}").unwrap();
        std::fs::write(plain.join("config.json"), "{}").unwrap();
        assert_eq!(decision_kind(&clef), Some("clef"));
        assert_eq!(decision_kind(&laya), Some("laya"));
        assert_eq!(decision_kind(&plain), None);
        assert_eq!(scan_laya(&root.to_string_lossy()), vec![laya.clone()]);
        std::fs::create_dir_all(root.join("r/data/recipes")).unwrap();
        std::fs::create_dir_all(root.join("r/data/codebooks")).unwrap();
        assert_eq!(repo_of_recipe(&root.join("r/data/recipes/a.recipe")), Some(root.join("r")));
        assert_eq!(repo_of_recipe(&clef.join("a.recipe")), None);
        let _ = std::fs::remove_dir_all(&root);
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
    fn calib_paths() {
        let fixed = "output.weight rtn codes=i8 group=128\nblk.0.attn_qkv.weight gptq codes=u4 zero=u8 cd=0 calib=/calib\nblk.*.attn_o.weight rtn codes=u4\n";
        assert!(recipe_needs_calib(fixed));
        assert_eq!(recipe_fixed_calib(fixed).as_deref(), Some("/calib"));
        let env = "# needs CALIB, e.g. calib=/somewhere\nblk.*.weight gptq codes=i4 calib=$CALIB\n";
        assert!(recipe_needs_calib(env));
        assert_eq!(recipe_fixed_calib(env), None);
        assert!(!recipe_needs_calib("# gptq calib=/x\nblk.*.weight rtn codes=i8\n"));
        assert!(!recipe_needs_calib("blk.*.weight rtn codes=i8 kld-calib-note=1\n"));
        assert_eq!(
            recipe_calib_to_env("# keep calib=/calib here\nblk.0.a.weight gptq calib=/calib cd=2\nblk.1.a.weight gptq cd=0 calib=/calib\nblk.2.a.weight gptq calib=$CALIB\nnocalib=/x\n"),
            "# keep calib=/calib here\nblk.0.a.weight gptq calib=$CALIB cd=2\nblk.1.a.weight gptq cd=0 calib=$CALIB\nblk.2.a.weight gptq calib=$CALIB\nnocalib=/x\n"
        );
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
        // keptn's w5 and i8g recipes need the same key; the 26B-A4B MoE (arch/gemma4) needs none
        for codes in ["u5 zero=u8", "i8", "u4 zero=u8"] {
            let r = recipe_kind(&format!("blk.*.attn_qkv.weight rtn codes={codes} group=128 scale=f16\n"));
            assert!(known_sets("gemma4", &r).is_some(), "{codes}");
            assert!(known_sets("gemma4moe", &r).is_none(), "{codes}");
        }
        let moe = CkptKind { model_type: "gemma4".into(), moe: true, ..Default::default() };
        assert_eq!(moe.family(), "gemma4moe");
        assert_eq!(CkptKind { model_type: "gemma4".into(), ..Default::default() }.family(), "gemma4");
        let ck = CkptKind { model_type: "qwen3_5".into(), ..Default::default() };
        assert_eq!(ck.family(), "qwen35");
        assert!(bf16_plugin_mismatch(&["E declare: weight 'blk.0.ffn_gate_up.weight' is declared bf16, and 'blk.0.ffn_gate_up.weight' holds codes as fp4_e2m1*e8m0[1x32]: only".to_string()]));
    }
}
