//! One GPU pane: options, model selection, estimates and Fit (the Python `GpuPane`).
//!
//! Pane methods that need the rest of the app are written as `impl App` methods taking the
//! pane index `p` (0 = left, 1 = right); that keeps Rust's borrow rules simple.

use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use super::app::{App, Pending, Timer};
use super::consts::*;
use super::devices::{dev_label, gpu_stats, Device};
use super::estimate::{dual_estimate, est_simple, estimate_vram, file_size, model_sig, pcie_link, plan_dual, DualPlan, Est, Manual};
use super::gguf::{ModelInfo, MtpCand};
use super::mtp::{find_mtp_candidates, stats_fingerprint, stats_summary};
use super::proc::{Proc, ProcMsg};
use super::util::*;
use super::radiance::RadInfo;
use super::vllm::{VAdjust, VPlan, VllmModelInfo};

// --------------------------------------------------------------------- info --
#[derive(Clone)]
pub enum Info {
    G(Arc<ModelInfo>),
    V(Arc<VllmModelInfo>),
    R(Arc<RadInfo>),
}

impl Info {
    /// Read a model's metadata the way `backend` needs it.
    pub fn load(path: &std::path::Path, backend: &str) -> Result<Info, String> {
        if backend == RADIANCE {
            Ok(Info::R(Arc::new(RadInfo::load(path)?)))
        } else if backend == VLLM {
            Ok(Info::V(Arc::new(VllmModelInfo::load(path)?)))
        } else {
            let mut i = ModelInfo::load(path)?;
            i.mtp_candidates = find_mtp_candidates(path, &i);
            Ok(Info::G(Arc::new(i)))
        }
    }
    /// `load`, but a parser panic (an unexpected file) becomes an error instead of
    /// silently killing the worker thread.
    pub fn load_safe(path: &std::path::Path, backend: &str) -> Result<Info, String> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Info::load(path, backend))) {
            Ok(r) => r,
            Err(e) => {
                let m = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown error".into());
                Err(format!("internal error while reading it ({m})"))
            }
        }
    }
    pub fn is_vllm(&self) -> bool {
        matches!(self, Info::V(_))
    }
    /// Which backends can use this metadata (GGUF metadata serves llama.cpp and KoboldCpp).
    pub fn fits(&self, backend: &str) -> bool {
        match self {
            Info::G(_) => backend == LLAMA || backend == KOBOLD,
            Info::V(_) => backend == VLLM,
            Info::R(_) => backend == RADIANCE,
        }
    }
    pub fn path(&self) -> &PathBuf {
        match self {
            Info::G(i) => &i.path,
            Info::V(i) => &i.path,
            Info::R(i) => &i.path,
        }
    }
    pub fn summary(&self) -> String {
        match self {
            Info::G(i) => i.summary(),
            Info::V(i) => i.summary(),
            Info::R(i) => i.summary(),
        }
    }
    pub fn metadata_text(&self) -> String {
        match self {
            Info::G(i) => i.metadata_text(),
            Info::V(i) => i.metadata_text(),
            Info::R(i) => i.metadata_text(),
        }
    }
    pub fn sig(&self) -> String {
        match self {
            Info::G(i) => model_sig(i),
            Info::V(i) => format!("{}:{}:{}:{}:{}", i.arch, i.n_layer, i.n_embd as i64, i.n_expert, i.nextn),
            Info::R(i) => i.sig(),
        }
    }
    pub fn n_layer(&self) -> usize {
        match self {
            Info::G(i) => i.n_layer,
            Info::V(i) => i.n_layer,
            Info::R(i) => i.n_layer,
        }
    }
    pub fn chat_template(&self) -> Option<String> {
        match self {
            Info::G(i) => i.chat_template(),
            Info::V(i) => i.chat_template.clone(),
            Info::R(_) => None,
        }
    }
    pub fn g(&self) -> Option<&Arc<ModelInfo>> {
        if let Info::G(i) = self { Some(i) } else { None }
    }
    pub fn v(&self) -> Option<&Arc<VllmModelInfo>> {
        if let Info::V(i) = self { Some(i) } else { None }
    }
    pub fn r(&self) -> Option<&Arc<RadInfo>> {
        if let Info::R(i) = self { Some(i) } else { None }
    }
}

// --------------------------------------------------------------- messages --
pub enum Metrics {
    Llama(HashMap<String, f64>),
    Vllm(HashMap<String, f64>),
    Kobold(Value),
    Rad(HashMap<String, f64>),
}

pub enum PaneMsg {
    Info(PathBuf, Result<Info, String>),
    /// Radiance's /stats JSON (decode floor, startup time), read now and then
    RadStats { proc_id: u64, stats: Value },
    Http { proc_id: u64, health: Option<u16>, metrics: Option<Metrics> },
}

#[derive(Clone, Debug)]
pub struct Mon {
    pub dev: Device,
    pub base: Option<f64>,
    pub gtt: Option<f64>,
    pub peak: f64,
    pub expected: f64,
    pub full: bool,
    pub dv: Option<f64>,
    pub dg: Option<f64>,
    pub gtt_noted: f64,
}

pub const BOOL_VARS: [&str; 6] = ["jinja", "relax_tpl", "dual_moe_spread", "eager", "text_only", "kquiet"];
pub const ALL_VARS: [&str; 48] = [
    "ctx", "ngl", "fa", "kv", "n_cpu_moe", "parallel", "threads", "batch", "ubatch", "port", "jinja", "relax_tpl",
    "dual_strategy", "dual_mode", "dual_ts", "dual_first", "dual_moe_spread", "mmproj", "mtp", "mtp_n", "extra",
    "gpu_util", "kv_dtype", "dtype", "max_batched", "cpu_offload", "eager", "text_only", "reasoning", "quant", "kbuild",
    // Radiance
    "r_kv", "r_batched", "r_headroom", "r_spec", "r_spec_ngram", "r_ngram_min", "r_ngram_max", "r_draft_window", "r_vision", "r_embed", "r_placement", "r_host_pool",
    "r_prefix_mib", "r_prefill_span", "r_reasoning", "r_tp_wire",
    // llama.cpp Dual GPU: the card the MTP drafter runs on ("(auto)" or a device id)
    "mtp_dev",
];
// (kquiet is added separately; ALL_VARS order is the Python self.v order)

pub type CalibHit = (Option<Opts>, Option<Info>, Option<String>);

pub struct Pane {
    pub side: &'static str,
    pub idx: usize,
    pub device: Option<Device>,
    pub info: Option<Info>,
    /// metadata read in flight: (model path, started)
    pub info_reading: Option<(PathBuf, Instant)>,
    /// last model path whose metadata read finished (ok or error)
    pub info_done: Option<PathBuf>,
    /// Fit was clicked while the metadata was still being read: run it when it arrives
    pub fit_after_info: bool,
    pub models: Vec<String>,
    pub proc: Option<Proc>,
    pub user_stopped: bool,
    pub proc_tx: Sender<ProcMsg>,
    pub proc_rx: Receiver<ProcMsg>,
    pub msg_tx: Sender<PaneMsg>,
    pub msg_rx: Receiver<PaneMsg>,
    pub inject: VecDeque<String>,
    pub lines: VecDeque<String>,
    pub cmd: Vec<String>,
    pub env_extra: Vec<(String, String)>,
    pub start_time: Option<Instant>,
    pub busy: i64,
    pub buffers: Vec<(String, f64)>,
    pub last_prompt: Option<(i64, f64)>,
    pub last_gen: Option<(i64, f64)>,
    pub last_accept: Option<f64>,
    pub populating: bool,
    pub loaded: Option<(String, Opts, f64)>,
    pub autoload: bool,
    pub gtt_base: Option<f64>,
    pub spilled: bool,
    pub spill_reported: bool,
    pub ready_time: Option<Instant>,
    pub expect_full: bool,
    pub expected_vram: f64,
    pub mon: Vec<Mon>,
    pub last_plan: Option<DualPlan>,
    pub req: Map<String, Value>,
    pub stats_ids_loaded: Option<(String, String)>,
    pub vram_base: Option<f64>,
    pub metrics_prev: Option<HashMap<String, f64>>,
    pub http_busy: bool,
    pub backend: String,
    // vLLM
    pub vbuf: std::collections::BTreeMap<usize, HashMap<String, f64>>,
    pub vkv_tokens: Option<i64>,
    pub vconc: Option<f64>,
    pub vplan: Option<VPlan>,
    pub v_adj_used: VAdjust,
    /// None = never looked at ("x" in Python), Some(None) = saved as None
    pub v_reason_saved: Option<Option<String>>,
    pub v_borrowed: Option<String>,
    pub v_launch_plan: Option<VPlan>,
    pub v_launch_adj: VAdjust,
    // Radiance
    /// what the startup log said: KV pool tokens, full-length sessions, % of worst case
    pub r_kv_tokens: Option<i64>,
    pub r_sessions: Option<f64>,
    pub r_backs: Option<f64>,
    pub r_kv_bytes: Option<f64>,
    /// the startup log's VRAM budget table: what the card could hold, not only what was asked for
    pub r_budget: super::radiance::Budget,
    pub r_arena: Option<f64>,
    pub r_settled: Vec<(i64, f64)>,
    /// Fit's plan check: a short radiance run that is stopped once it states its KV pool
    pub r_probe: Option<super::pane_r::Probe>,
    /// devices of the running load (for HIP_VISIBLE_DEVICES / monitors)
    pub r_devs: Vec<Device>,
    pub r_stats: Option<Value>,
    pub r_stats_t: Option<Instant>,
    /// "req refused … 401 invalid api key" lines from local addresses seen this run (only the
    /// first is shown: Radiance's own chat page sends some requests without the key)
    pub r_401: u64,
    // KoboldCpp
    pub k_sess: (i64, f64, i64, f64, i64),
    pub k_env_notes: Vec<String>,
    pub k_last_build: Option<String>,
    pub k_waiting: bool,
    pub k_gens: Option<i64>,
    pub k_logged: i64,
    pub k_lag: i64,
    // option variables (Tk StringVar / BooleanVar)
    pub vars: HashMap<String, Value>,
    // UI state
    pub model_var: String,
    pub model_info: String,
    pub est_text: String,
    pub est_color: String,
    pub est_pct: f64,
    pub dual_info: String,
    pub gpu_name: String,
    pub gpu_stat: String,
    pub gpu_stat2: String,
    pub vram_pct: f64,
    pub vram_pct2: f64,
    pub dual_note: String,
    pub mtp_note: String,
    pub v_mtp_note: String,
    pub k_note: (String, String),
    pub badge: (String, String),
    pub state: String,
    pub status: HashMap<&'static str, String>,
    pub mmproj_values: Vec<String>,
    pub mtp_values: Vec<String>,
    pub mtp_enabled: bool,
    pub v_mtp_values: Vec<String>,
    pub v_mtp_enabled: bool,
    pub v_text_enabled: bool,
    pub v_quant_enabled: bool,
    pub ngl_max: i64,
    pub web_enabled: bool,
    pub dual_mode_values: Vec<String>,
    pub mtp_cands: Vec<MtpCand>,
    pub mtp_saved: Option<Value>,
    pub mtp_r2l: HashMap<String, String>,
    pub mtp_l2r: HashMap<String, String>,
    pub pane_disabled: bool,
    pub borrow_cache: HashMap<(String, String), CalibHit>,
    pub dual_borrowed: Option<String>,
    pub req_start: Option<Instant>,
    // performance dashboard (llama.cpp)
    /// completed / aborted requests of the current run (newest last)
    pub dash_reqs: VecDeque<Value>,
    /// (unix time, /metrics snapshot)
    pub dash_metrics: VecDeque<(f64, HashMap<String, f64>)>,
    /// a request is in flight: (unix start, prompt tokens, slot context)
    pub dash_active: Option<(f64, Option<i64>, Option<i64>)>,
    /// prompt processing progress of the request in flight (0..1)
    pub dash_progress: Option<f64>,
    pub dash_slot_ctx: Option<i64>,
    /// task ids the server cancelled (a chat app often cancels a helper task while the real
    /// request carries on, so only a cancel of the request's own task means it was aborted)
    pub dash_cancelled: Vec<i64>,
    /// live generation of the request in flight: (tokens so far, recent t/s)
    pub dash_gen: Option<(i64, f64)>,
    /// unix time of Load / of becoming ready
    pub dash_load_t: Option<f64>,
    pub dash_ready_t: Option<f64>,
    pub show_dual: bool,
    /// the user said "load anyway" to a red estimate
    pub load_confirmed: bool,
    /// the estimate puts more on a card than the card holds (not merely "close to full")
    pub est_over: bool,
}

impl Pane {
    pub fn new(side: &'static str, idx: usize, backend: String) -> Self {
        let (ptx, prx) = channel();
        let (mtx, mrx) = channel();
        let mut vars = HashMap::new();
        for k in ALL_VARS.iter().chain(["kquiet"].iter()) {
            let v = if BOOL_VARS.contains(k) { Value::Bool(false) } else { Value::String(String::new()) };
            vars.insert(k.to_string(), v);
        }
        let mut status = HashMap::new();
        for k in ["url", "uptime", "speed", "avg", "buffers", "cache", "life"] {
            status.insert(k, "—".to_string());
        }
        Pane {
            side,
            idx,
            device: None,
            info: None,
            info_reading: None,
            info_done: None,
            fit_after_info: false,
            models: vec![],
            proc: None,
            user_stopped: false,
            proc_tx: ptx,
            proc_rx: prx,
            msg_tx: mtx,
            msg_rx: mrx,
            inject: VecDeque::new(),
            lines: VecDeque::new(),
            cmd: vec![],
            env_extra: vec![],
            start_time: None,
            busy: 0,
            buffers: vec![],
            last_prompt: None,
            last_gen: None,
            last_accept: None,
            populating: false,
            loaded: None,
            autoload: false,
            gtt_base: None,
            spilled: false,
            spill_reported: false,
            ready_time: None,
            expect_full: false,
            expected_vram: 0.0,
            mon: vec![],
            last_plan: None,
            req: Map::new(),
            stats_ids_loaded: None,
            vram_base: None,
            metrics_prev: None,
            http_busy: false,
            backend,
            vbuf: Default::default(),
            vkv_tokens: None,
            vconc: None,
            vplan: None,
            v_adj_used: VAdjust::default(),
            v_reason_saved: None,
            v_borrowed: None,
            v_launch_plan: None,
            v_launch_adj: VAdjust::default(),
            r_kv_tokens: None,
            r_sessions: None,
            r_backs: None,
            r_kv_bytes: None,
            r_budget: Default::default(),
            r_arena: None,
            r_settled: vec![],
            r_probe: None,
            r_devs: vec![],
            r_stats: None,
            r_stats_t: None,
            r_401: 0,
            k_sess: (0, 0.0, 0, 0.0, 0),
            k_env_notes: vec![],
            k_last_build: None,
            k_waiting: false,
            k_gens: None,
            k_logged: 0,
            k_lag: 0,
            vars,
            model_var: String::new(),
            model_info: String::new(),
            est_text: "Estimate: select a model".into(),
            est_color: String::new(),
            est_pct: 0.0,
            dual_info: String::new(),
            gpu_name: "(no GPU assigned)".into(),
            gpu_stat: String::new(),
            gpu_stat2: String::new(),
            vram_pct: 0.0,
            vram_pct2: 0.0,
            dual_note: String::new(),
            mtp_note: String::new(),
            v_mtp_note: String::new(),
            k_note: (String::new(), "#666".into()),
            badge: ("vLLM (Experimental)".into(), EXPERIMENTAL_FG.into()),
            state: "Stopped".into(),
            status,
            mmproj_values: vec!["(none)".into()],
            mtp_values: vec!["(off)".into()],
            mtp_enabled: false,
            v_mtp_values: vec!["(off)".into()],
            v_mtp_enabled: false,
            v_text_enabled: false,
            v_quant_enabled: true,
            ngl_max: 999,
            web_enabled: false,
            dual_mode_values: vec!["none".into(), "layer".into()],
            mtp_cands: vec![],
            mtp_saved: None,
            mtp_r2l: HashMap::new(),
            mtp_l2r: HashMap::new(),
            pane_disabled: false,
            borrow_cache: HashMap::new(),
            dual_borrowed: None,
            req_start: None,
            dash_reqs: VecDeque::new(),
            dash_metrics: VecDeque::new(),
            dash_active: None,
            dash_progress: None,
            dash_slot_ctx: None,
            dash_cancelled: vec![],
            dash_gen: None,
            dash_load_t: None,
            dash_ready_t: None,
            show_dual: false,
            load_confirmed: false,
            est_over: false,
        }
    }

    pub fn is_vllm(&self) -> bool {
        self.backend == VLLM
    }
    pub fn is_kobold(&self) -> bool {
        self.backend == KOBOLD
    }
    pub fn is_rad(&self) -> bool {
        self.backend == RADIANCE
    }
    pub fn is_llama(&self) -> bool {
        self.backend == LLAMA
    }
    pub fn server_name(&self) -> &'static str {
        if self.is_rad() {
            "radiance"
        } else if self.is_vllm() {
            "vllm serve"
        } else if self.is_kobold() {
            "koboldcpp"
        } else {
            "llama-server"
        }
    }
    pub fn last_key(&self) -> String {
        if self.is_rad() {
            format!("{}:Radiance", self.side)
        } else if self.is_vllm() {
            format!("{}:vLLM", self.side)
        } else if self.is_kobold() {
            format!("{}:Kobold", self.side)
        } else {
            self.side.to_string()
        }
    }
    pub fn var_s(&self, k: &str) -> String {
        match self.vars.get(k) {
            Some(Value::String(s)) => s.clone(),
            Some(v) => py_str(v),
            None => String::new(),
        }
    }
    pub fn var_b(&self, k: &str) -> bool {
        truthy(self.vars.get(k))
    }
    /// `int(str(var.get()).strip())`, default on ValueError
    pub fn int(&self, k: &str, d: i64) -> i64 {
        let s = self.var_s(k);
        let t = s.trim().replace('_', "");
        t.parse::<i64>().unwrap_or(d)
    }
    pub fn float(&self, k: &str, d: f64) -> f64 {
        self.var_s(k).trim().parse::<f64>().unwrap_or(d)
    }
    /// Tk `var.set(val)`
    pub fn put_var(&mut self, k: &str, v: &Value) {
        let nv = if BOOL_VARS.contains(&k) {
            Value::Bool(truthy(Some(v)))
        } else {
            match v {
                Value::Null => Value::String(String::new()),
                Value::String(s) => Value::String(s.clone()),
                other => Value::String(py_str(other)),
            }
        };
        self.vars.insert(k.to_string(), nv);
    }
    pub fn info_g(&self) -> Option<Arc<ModelInfo>> {
        self.info.as_ref().and_then(|i| i.g().cloned())
    }
    pub fn info_v(&self) -> Option<Arc<VllmModelInfo>> {
        self.info.as_ref().and_then(|i| i.v().cloned())
    }
    pub fn info_r(&self) -> Option<Arc<RadInfo>> {
        self.info.as_ref().and_then(|i| i.r().cloned())
    }
    /// Selected MTP option as a file path (dropdown labels carry a match marker).
    pub fn mtp_sel(&self) -> String {
        let v = self.var_s("mtp");
        let v = if v.is_empty() { "(off)".to_string() } else { v };
        self.mtp_l2r.get(&v).cloned().unwrap_or(v)
    }
    pub fn mtp_bytes(&self, s: &Opts) -> f64 {
        let sel = gs(s, "mtp", "");
        if sel.is_empty() || sel == "(off)" {
            return 0.0;
        }
        self.mtp_cands.iter().find(|c| c.rel == sel).map(|c| c.size).unwrap_or(0.0)
    }
    pub fn set_status(&mut self, k: &'static str, v: impl Into<String>) {
        self.status.insert(k, v.into());
    }
    pub fn append_lines(&mut self, ls: &[String]) {
        for l in ls {
            if self.lines.len() >= 3000 {
                self.lines.pop_front();
            }
            self.lines.push_back(l.clone());
        }
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn calib_clamp(x: f64, lim: f64) -> f64 {
    x.max(-lim).min(lim)
}

impl App {
    // ================================================================ basics ==
    pub fn default_port(&self, p: usize) -> i64 {
        let pane = &self.panes[p];
        let g = self.settings.g();
        let side = pane.side;
        if pane.is_rad() {
            gi(g, &format!("radiance_{side}_port"), if side == "left" { 8080 } else { 8081 })
        } else if pane.is_vllm() {
            gi(g, &format!("vllm_{side}_port"), if side == "left" { 8042 } else { 8043 })
        } else if pane.is_kobold() {
            gi(g, &format!("kobold_{side}_port"), if side == "left" { 5001 } else { 5002 })
        } else {
            gi(g, &format!("{side}_port"), if side == "left" { 8080 } else { 8081 })
        }
    }

    pub fn models_root(&self, p: usize) -> PathBuf {
        let g = self.settings.g();
        PathBuf::from(if self.panes[p].is_rad() {
            gs(g, "radiance_models_dir", "")
        } else if self.panes[p].is_vllm() {
            gs(g, "vllm_models_dir", "")
        } else {
            gs(g, "models_dir", "")
        })
    }

    pub fn model_path(&self, p: usize) -> Option<PathBuf> {
        let rel = &self.panes[p].model_var;
        if rel.is_empty() { None } else { Some(self.models_root(p).join(rel)) }
    }

    pub fn no_models_text(&self, p: usize) -> String {
        if self.panes[p].is_rad() {
            format!("No .rad containers or checkpoint folders in {} (convert one with Convert…)", self.models_root(p).display())
        } else if self.panes[p].is_vllm() {
            format!("No model folders (config.json + *.safetensors) in {}", self.models_root(p).display())
        } else {
            "No .gguf files found in models folder".into()
        }
    }

    pub fn dual_active(&self, p: usize) -> bool {
        self.panes[p].side == "left" && self.dual_on() && self.dual_devices().len() == 2
    }

    pub fn dev_base(&self, p: usize) -> String {
        if self.dual_active(p) {
            let mut names: Vec<String> = self.dual_devices().iter().map(|d| d.name.clone()).collect();
            names.sort();
            return format!("DUAL:{}", names.join("+"));
        }
        self.panes[p].device.as_ref().map(|d| d.name.clone()).unwrap_or_else(|| "?".into())
    }

    pub fn dev_key(&self, p: usize) -> String {
        let key = self.dev_base(p);
        let pane = &self.panes[p];
        if pane.is_rad() {
            format!("Radiance:{key}")
        } else if pane.is_vllm() {
            format!("vLLM:{key}")
        } else if pane.is_kobold() {
            format!("Kobold:{key}")
        } else {
            key
        }
    }

    // ------------------------------------------------- builds / capabilities --
    pub fn k_default_build(&self, p: usize) -> String {
        if self.panes[p].is_kobold() {
            let g = self.settings.g();
            if gs(g, "kobold_rocm_bin", "").is_empty() && !gs(g, "kobold_vulkan_bin", "").is_empty() {
                return "Vulkan".into();
            }
            return "ROCm".into();
        }
        if !PathBuf::from(self.llama_bin("ROCm")).is_file() && PathBuf::from(self.llama_bin("Vulkan")).is_file() {
            return "Vulkan".into();
        }
        "ROCm".into()
    }

    pub fn kbuild(&self, p: usize) -> String {
        let b = self.panes[p].var_s("kbuild");
        if KOBOLD_BUILDS.contains(&b.as_str()) { b } else { self.k_default_build(p) }
    }

    pub fn k_bin(&self, p: usize, build: Option<&str>) -> String {
        let b = build.map(String::from).unwrap_or_else(|| self.kbuild(p));
        if self.panes[p].is_llama() {
            return self.llama_bin(&b);
        }
        self.kobold_bin(&b)
    }

    pub fn supports(&mut self, p: usize, flag: &str) -> bool {
        if self.panes[p].is_kobold() {
            let b = self.kbuild(p);
            return self.kobold_flags(&b).map(|f| f.contains(flag)).unwrap_or(false);
        }
        if self.panes[p].is_llama() {
            let b = self.kbuild(p);
            return self.llama_help(&b).contains(flag);
        }
        self.server_help.contains(flag)
    }

    pub fn mtp_flag(&self, p: usize) -> &'static str {
        if self.panes[p].is_kobold() { "--usemtp" } else { "--spec-type" }
    }

    pub fn split_modes_for(&mut self, p: usize) -> Vec<String> {
        if self.panes[p].is_kobold() {
            // --tensor_split = layer split; rowsplit is a hipBLAS (ROCm) option
            let mut v = vec!["none".to_string(), "layer".to_string()];
            if self.kbuild(p) == "ROCm" {
                v.push("row".into());
            }
            return v;
        }
        if self.panes[p].is_llama() {
            let b = self.kbuild(p);
            let h = self.llama_help(&b);
            return super::estimate::split_modes_supported(&h);
        }
        self.split_modes.clone()
    }

    /// Calibration / stats key, one per build. llama.cpp's ROCm build keeps the plain key.
    pub fn calib_key(&self, p: usize, base: &str) -> String {
        if self.panes[p].is_kobold() {
            return format!("Kobold-{}:{base}", self.kbuild(p));
        }
        if self.panes[p].is_llama() && self.kbuild(p) == "Vulkan" {
            return format!("Vulkan:{base}");
        }
        base.to_string()
    }

    pub fn stats_key(&self, p: usize) -> String {
        if self.panes[p].is_kobold() || self.panes[p].is_llama() {
            let b = self.dev_base(p);
            self.calib_key(p, &b)
        } else {
            self.dev_key(p)
        }
    }

    pub fn saved_options(&self, p: usize, rel: &str) -> Opts {
        let st = &self.settings;
        let pane = &self.panes[p];
        if pane.is_vllm() {
            return st.get_model(&self.dev_key(p), rel, Some(&default_vllm_model()));
        }
        if pane.is_rad() {
            return st.get_model(&self.dev_key(p), rel, Some(&default_rad_model()));
        }
        let mut s;
        if pane.is_kobold() && !st.has_model(&self.dev_key(p), rel) {
            s = st.get_model(&self.dev_base(p), rel, None); // first time: start from the llama.cpp setup
            s.insert("port".into(), Value::Null);
        } else {
            s = st.get_model(&self.dev_key(p), rel, None);
        }
        if pane.is_kobold() {
            for (k, d) in kobold_defaults() {
                s.entry(k).or_insert(d);
            }
        }
        let kb = gs(&s, "kbuild", "");
        if !KOBOLD_BUILDS.contains(&kb.as_str()) {
            s.insert("kbuild".into(), Value::from(self.k_default_build(p)));
        }
        if pane.is_kobold() && !["none", "layer", "row"].contains(&gs(&s, "dual_mode", "").as_str()) {
            s.insert("dual_mode".into(), Value::from("layer"));
        }
        s
    }

    // ============================================================== options ==
    pub fn current_options(&self, p: usize) -> Opts {
        if self.panes[p].is_vllm() {
            return self.v_current_options(p);
        }
        if self.panes[p].is_rad() {
            return self.r_current_options(p);
        }
        let v = &self.panes[p];
        let or = |k: &str, d: &str| {
            let s = v.var_s(k);
            if s.is_empty() { d.to_string() } else { s }
        };
        let df = v.var_s("dual_first");
        let mtp_dev = {
            let m = v.var_s("mtp_dev");
            let id = m.split(':').next().unwrap_or("").trim().to_string();
            if id.is_empty() { "(auto)".to_string() } else { id }
        };
        let mut o = json!({
            "ctx": v.int("ctx", 8192), "ngl": v.int("ngl", 999),
            "fa": or("fa", "on"), "kv": or("kv", "f16"),
            "n_cpu_moe": v.int("n_cpu_moe", 0),
            "parallel": v.int("parallel", 1).max(1), "threads": v.int("threads", 0),
            "batch": v.int("batch", 2048), "ubatch": v.int("ubatch", 512),
            "port": v.int("port", self.default_port(p)),
            "jinja": v.var_b("jinja"), "relax_tpl": v.var_b("relax_tpl"),
            "dual_strategy": or("dual_strategy", "Auto Balance"),
            "dual_mode": or("dual_mode", "layer"),
            "dual_ts": v.var_s("dual_ts").trim(),
            "dual_first": if df.is_empty() { String::new() } else { df.split(':').next().unwrap_or("").to_string() },
            "mtp_dev": mtp_dev,
            "dual_moe_spread": v.var_b("dual_moe_spread"), "mmproj": or("mmproj", "(none)"),
            "mtp": v.mtp_sel(), "mtp_n": v.int("mtp_n", 4).max(1),
            "extra": v.var_s("extra"),
            "kbuild": self.kbuild(p),
        });
        if v.is_kobold() {
            o["kquiet"] = Value::Bool(v.var_b("kquiet"));
        }
        o.as_object().unwrap().clone()
    }

    pub fn v_current_options(&self, p: usize) -> Opts {
        let v = &self.panes[p];
        let strat = v.var_s("dual_strategy");
        let mode = v.var_s("dual_mode");
        let kvd = v.var_s("kv_dtype");
        let dt = v.var_s("dtype");
        let q = v.var_s("quant");
        let co = v.float("cpu_offload", 0.0).max(0.0);
        let df = v.var_s("dual_first");
        let reasoning = v.var_s("reasoning").trim().to_string();
        let mut o = Map::new();
        o.insert("ctx".into(), v.int("ctx", 8192).into());
        o.insert("gpu_util".into(), json!(round3(v.float("gpu_util", 0.9).max(0.1).min(0.99))));
        o.insert("kv_dtype".into(), Value::from(if VLLM_KV_TYPES.contains(&kvd.as_str()) { kvd } else { "auto".into() }));
        o.insert("dtype".into(), Value::from(if VLLM_DTYPES.contains(&dt.as_str()) { dt } else { "auto".into() }));
        if vllm_online_quant(&q).is_some() && q != "(none)" {
            o.insert("quant".into(), Value::from(q));
        }
        o.insert("parallel".into(), v.int("parallel", 1).max(1).into());
        o.insert("max_batched".into(), v.int("max_batched", 0).max(0).into());
        o.insert("cpu_offload".into(), if co.fract() == 0.0 { json!(co as i64) } else { json!(co) });
        o.insert("eager".into(), v.var_b("eager").into());
        o.insert("text_only".into(), v.var_b("text_only").into());
        o.insert("reasoning".into(), Value::from(if reasoning.is_empty() { "(none)".to_string() } else { reasoning }));
        o.insert("relax_tpl".into(), v.var_b("relax_tpl").into());
        o.insert(
            "dual_strategy".into(),
            Value::from(if VLLM_STRATEGIES.contains(&strat.as_str()) { strat } else { "Auto VRAM".into() }),
        );
        o.insert("dual_mode".into(), Value::from(if VLLM_SPLIT_MODES.contains(&mode.as_str()) { mode } else { "pipeline".into() }));
        o.insert("dual_ts".into(), Value::from(v.var_s("dual_ts").trim()));
        o.insert(
            "dual_first".into(),
            Value::from(if df.is_empty() { String::new() } else { df.split(':').next().unwrap_or("").to_string() }),
        );
        o.insert("mtp".into(), Value::from(if v.var_s("mtp") == "(built-in)" { "(built-in)" } else { "(off)" }));
        o.insert("mtp_n".into(), v.int("mtp_n", 3).max(1).into());
        o.insert("port".into(), v.int("port", self.default_port(p)).into());
        o.insert("extra".into(), Value::from(v.var_s("extra")));
        o
    }

    /// UI change of one option (Tk var trace).
    pub fn set_var(&mut self, p: usize, key: &str, val: &Value) {
        if !self.panes[p].vars.contains_key(key) {
            return;
        }
        // the MTP dropdown shows labels; the UI sends the label back
        self.panes[p].put_var(key, val);
        self.on_option_change(p);
    }

    /// Internal var.set() that triggers the trace like Tk does.
    pub fn set_var_notify(&mut self, p: usize, key: &str, val: Value) {
        self.panes[p].put_var(key, &val);
        self.on_option_change(p);
    }

    pub fn on_option_change(&mut self, p: usize) {
        if self.panes[p].populating {
            return;
        }
        self.k_refresh_row(p);
        self.refresh_life(p);
        self.update_mtp_note(p);
        self.update_estimate(p);
        self.schedule(Timer::Save(p), 600);
    }

    pub fn save_options(&mut self, p: usize) {
        self.cancel_timer(&Timer::Save(p));
        let rel = self.panes[p].model_var.clone();
        if !rel.is_empty() && self.panes[p].device.is_some() {
            let key = self.dev_key(p);
            let opts = self.current_options(p);
            self.settings.set_model(&key, &rel, opts);
        }
    }

    // =============================================================== models ==
    pub fn set_device(&mut self, p: usize, dev: Option<Device>) {
        {
            let pane = &mut self.panes[p];
            pane.gpu_name = match &dev {
                Some(d) => format!("{}   [{}]", d.name, d.id),
                None => "(no GPU assigned — open Settings)".into(),
            };
            pane.device = dev;
        }
        self.update_gpu_stats(p);
        self.on_model_selected(p);
    }

    pub fn set_models(&mut self, p: usize, models: Vec<String>) {
        let cur = self.panes[p].model_var.clone();
        self.panes[p].models = models.clone();
        if !models.contains(&cur) {
            self.panes[p].model_var.clear();
            self.panes[p].info = None;
            self.panes[p].model_info = if models.is_empty() { self.no_models_text(p) } else { String::new() };
            self.update_estimate(p);
        }
    }

    pub fn update_gpu_stats(&mut self, p: usize) {
        if self.dual_active(p) {
            let devs = self.dual_devices();
            for (i, d) in devs.iter().enumerate() {
                let (used, total, busy) = gpu_stats(Some(d));
                let nm = d.short_name();
                let (txt, pct) = match (total, used) {
                    (Some(t), Some(u)) if t > 0.0 => (
                        format!("{nm}:   VRAM {:.1} / {:.1} GiB", u / GIB, t / GIB)
                            + &busy.map(|b| format!("   •   GPU busy {}%", b as i64)).unwrap_or_default(),
                        100.0 * u / t,
                    ),
                    _ => (format!("{nm}:   VRAM total {:.1} GiB (live stats unavailable)", d.total_mib as f64 / 1024.0), 0.0),
                };
                let pane = &mut self.panes[p];
                if i == 0 {
                    pane.gpu_stat = txt;
                    pane.vram_pct = pct;
                } else {
                    pane.gpu_stat2 = txt;
                    pane.vram_pct2 = pct;
                }
            }
            return;
        }
        let pane = &mut self.panes[p];
        let Some(dev) = pane.device.clone() else {
            pane.gpu_stat.clear();
            pane.vram_pct = 0.0;
            return;
        };
        let (used, total, busy) = gpu_stats(Some(&dev));
        match (total, used) {
            (Some(t), Some(u)) if t > 0.0 => {
                pane.gpu_stat = format!("VRAM {:.1} / {:.1} GiB", u / GIB, t / GIB)
                    + &busy.map(|b| format!("   •   GPU busy {}%", b as i64)).unwrap_or_default();
                pane.vram_pct = 100.0 * u / t;
            }
            _ => {
                pane.gpu_stat = format!("VRAM total {:.1} GiB (live stats unavailable)", dev.total_mib as f64 / 1024.0);
            }
        }
    }

    pub fn select_model(&mut self, p: usize, rel: &str) {
        if self.panes[p].proc.is_some() || !self.panes[p].models.iter().any(|m| m == rel) {
            return;
        }
        self.panes[p].model_var = rel.to_string();
        self.on_model_selected(p);
    }

    pub fn on_model_selected(&mut self, p: usize) {
        self.panes[p].borrow_cache.clear();
        let rel = self.panes[p].model_var.clone();
        let Some(path) = self.model_path(p) else { return };
        if rel.is_empty() {
            return;
        }
        let vl = self.panes[p].is_vllm();
        let rad = self.panes[p].is_rad();
        if !vl && !rad {
            // mmproj choices from the same folder
            let mut mm: Vec<String> = std::fs::read_dir(path.parent().unwrap_or(&path))
                .map(|r| {
                    r.filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|n| n.ends_with(".gguf") && n.to_lowercase().contains("mmproj"))
                        .collect()
                })
                .unwrap_or_default();
            mm.sort();
            self.panes[p].mmproj_values = std::iter::once("(none)".to_string()).chain(mm).collect();
        }
        // restore saved options
        self.panes[p].populating = true;
        let mut s = self.saved_options(p, &rel);
        if s.get("port").map(|v| v.is_null()).unwrap_or(true) {
            s.insert("port".into(), self.default_port(p).into());
        }
        if rad {
            self.r_prepare_options(p, &mut s);
        } else if vl {
            self.panes[p].v_reason_saved = Some(s.get("reasoning").and_then(|v| v.as_str()).map(String::from));
            if !VLLM_STRATEGIES.contains(&gs(&s, "dual_strategy", "").as_str()) {
                s.insert("dual_strategy".into(), "Auto VRAM".into());
            }
            if !VLLM_SPLIT_MODES.contains(&gs(&s, "dual_mode", "").as_str()) {
                s.insert("dual_mode".into(), "pipeline".into());
            }
            self.panes[p].v_mtp_values = vec!["(off)".into()];
            self.panes[p].v_mtp_note.clear();
        } else {
            if !self.panes[p].mmproj_values.contains(&gs(&s, "mmproj", "")) {
                s.insert("mmproj".into(), "(none)".into());
            }
            self.panes[p].mtp_saved = s.get("mtp").cloned().filter(|v| !v.is_null());
            s.insert("mtp".into(), "(off)".into());
            self.panes[p].mtp_values = vec!["(off)".into()];
            self.panes[p].mtp_note.clear();
            self.panes[p].mtp_l2r.clear();
            self.panes[p].mtp_r2l.clear();
        }
        let (dm, kd, dv, dr) = (default_model(), kobold_defaults(), default_vllm_model(), default_rad_model());
        let keys: Vec<String> = self.panes[p].vars.keys().cloned().collect();
        for k in keys {
            let val = if let Some(v) = s.get(&k) {
                v.clone()
            } else if let Some(v) = dr.get(&k).filter(|_| rad) {
                v.clone()
            } else if let Some(v) = dm.get(&k) {
                v.clone()
            } else if let Some(v) = kd.get(&k) {
                v.clone()
            } else if let Some(v) = dv.get(&k) {
                v.clone()
            } else {
                dr.get(&k).cloned().unwrap_or(Value::Null)
            };
            self.panes[p].put_var(&k, &val);
        }
        self.panes[p].populating = false;
        self.k_refresh_row(p);
        self.refresh_life(p);
        // metadata (threaded)
        if let Some(cached) = self.info_cache_get(&path) {
            self.set_info(p, cached);
        } else {
            self.panes[p].info = None;
            self.panes[p].model_info = "Reading model metadata…".into();
            self.spawn_info_read(p, path);
            self.update_estimate(p);
        }
    }

    /// Read the model's metadata on a worker thread; the result arrives in `poll_pane`.
    pub fn spawn_info_read(&mut self, p: usize, path: PathBuf) {
        let backend = self.panes[p].backend.clone();
        if backend == RADIANCE {
            let g = self.settings.g();
            super::radiance::set_tools(&gs(g, "radiance_bin", ""), &gs(g, "radiance_home", ""));
        }
        self.panes[p].info_reading = Some((path.clone(), Instant::now()));
        self.panes[p].info_done = None;
        let tx = self.panes[p].msg_tx.clone();
        std::thread::spawn(move || {
            let r = Info::load_safe(&path, &backend);
            let _ = tx.send(PaneMsg::Info(path, r));
        });
    }

    /// True while the selected model's metadata is still being read in the background.
    pub fn info_pending(&self, p: usize) -> bool {
        let Some(path) = self.model_path(p) else { return false };
        self.panes[p].info.is_none()
            && self.panes[p].info_reading.as_ref().map(|(rp, _)| rp == &path).unwrap_or(false)
    }

    /// Called from the 1.5 s tick: re-request metadata that never arrived.
    pub fn check_info_read(&mut self, p: usize) {
        let Some(path) = self.model_path(p) else { return };
        if self.panes[p].info.is_some() || self.panes[p].info_done.as_ref() == Some(&path) {
            return;
        }
        let stale = match &self.panes[p].info_reading {
            None => true,
            Some((rp, _)) => rp != &path,
        };
        if stale {
            eprintln!("llm-duo: {} pane: metadata for {} did not arrive, reading it again", self.panes[p].side, path.display());
            self.spawn_info_read(p, path);
        }
    }

    pub fn set_info(&mut self, p: usize, info: Info) {
        if !info.fits(&self.panes[p].backend) {
            return; // metadata from before a backend switch
        }
        self.panes[p].model_info = info.summary();
        self.panes[p].info = Some(info.clone());
        match &info {
            Info::V(v) => self.v_apply_info(p, v.clone()),
            Info::R(r) => self.r_apply_info(p, r.clone()),
            Info::G(g) => {
                self.panes[p].ngl_max = 999.max(g.n_layer as i64 + 1);
                let c = g.mtp_candidates.clone();
                self.apply_mtp_candidates(p, c);
            }
        }
        self.update_estimate(p);
        if self.panes[p].autoload {
            self.schedule(Timer::AutoLoad(p), 300);
        }
        if std::mem::take(&mut self.panes[p].fit_after_info) && self.panes[p].proc.is_none() {
            self.fit_context(p);
        }
    }

    pub fn apply_mtp_candidates(&mut self, p: usize, mut cands: Vec<MtpCand>) {
        let kb = self.panes[p].is_kobold();
        if kb && !self.supports(p, "--draftmodel") {
            cands.clear(); // (very old KoboldCpp) no draft model support
        }
        let mut builtin = self.panes[p].info_g().map(|i| i.nextn > 0).unwrap_or(false);
        if kb && !self.supports(p, "--usemtp") {
            builtin = false; // KoboldCpp before 1.115: no built-in MTP heads
        }
        let best_rel = cands.iter().find(|c| c.ok == Some(true)).map(|c| c.rel.clone());
        let mut r2l = HashMap::new();
        let mut l2r = HashMap::new();
        for c in &cands {
            let label = match c.ok {
                Some(true) => {
                    let tag = if Some(&c.rel) == best_rel.as_ref() { "★ BEST " } else { "✓ MATCH" };
                    format!("{tag}  {:.2} GiB   {}", c.size / GIB, c.rel)
                }
                Some(false) => {
                    let why = if c.note.contains("size") { "wrong size " } else { "wrong model" };
                    format!("✗ {why}  {:.2} GiB   {}", c.size / GIB, c.rel)
                }
                None => format!("? unknown    {:.2} GiB   {}", c.size / GIB, c.rel),
            };
            r2l.insert(c.rel.clone(), label.clone());
            l2r.insert(label, c.rel.clone());
        }
        let mut values = vec!["(off)".to_string()];
        if builtin {
            values.push("(built-in)".into());
        }
        let mut labels = values.clone();
        for c in &cands {
            values.push(c.rel.clone());
            labels.push(r2l[&c.rel].clone());
        }
        let supported = if kb { !cands.is_empty() || builtin } else { let f = self.mtp_flag(p); self.supports(p, f) };
        {
            let pane = &mut self.panes[p];
            pane.mtp_cands = cands.clone();
            pane.mtp_r2l = r2l;
            pane.mtp_l2r = l2r;
            pane.mtp_values = labels;
            pane.mtp_enabled = supported && (!cands.is_empty() || builtin);
        }
        let saved = self.panes[p].mtp_saved.clone().and_then(|v| v.as_str().map(String::from));
        let choice = if saved.as_deref() == Some("(built-in)") && builtin {
            "(built-in)".to_string()
        } else if cands.is_empty() {
            "(off)".to_string()
        } else if saved.is_none() {
            // never configured: auto-pick a matching drafter
            let best = &cands[0];
            if best.ok == Some(true) && supported { best.rel.clone() } else { "(off)".into() }
        } else if values.contains(saved.as_ref().unwrap()) {
            saved.clone().unwrap()
        } else {
            "(off)".to_string()
        };
        let label = self.panes[p].mtp_r2l.get(&choice).cloned().unwrap_or(choice.clone());
        self.panes[p].populating = true;
        self.panes[p].put_var("mtp", &Value::from(label));
        self.panes[p].populating = false;
        self.update_mtp_note(p);
        if saved.is_none() && choice != "(off)" {
            self.on_option_change(p); // persist the auto choice
        }
    }

    pub fn update_mtp_note(&mut self, p: usize) {
        if self.panes[p].is_rad() {
            return;
        }
        if self.panes[p].is_vllm() {
            self.v_mtp_note(p);
            return;
        }
        if self.panes[p].is_kobold() && self.k_mtp_note(p) {
            return;
        }
        if !self.panes[p].is_kobold() && !self.supports(p, "--spec-type") {
            let pane = &mut self.panes[p];
            pane.mtp_note = if !pane.mtp_cands.is_empty() {
                "MTP needs a newer llama.cpp build (no --spec-type in llama-server --help)".into()
            } else {
                String::new()
            };
            return;
        }
        let pane = &mut self.panes[p];
        let sel = pane.mtp_sel();
        if sel == "(built-in)" {
            pane.mtp_note =
                "model's own MTP head (no extra file)  •  adds its layer + its KV cache (~1 GiB at long context)".into();
            return;
        }
        if pane.mtp_cands.is_empty() {
            let builtin = pane.info_g().map(|i| i.nextn > 0).unwrap_or(false);
            pane.mtp_note = if builtin { "MTP off  •  this model has a built-in MTP head".into() } else { String::new() };
            return;
        }
        if let Some(c) = pane.mtp_cands.iter().find(|c| c.rel == sel) {
            pane.mtp_note = format!("{}  •  {:.2} GiB", c.note, c.size / GIB);
        } else {
            let ok: Vec<String> = pane
                .mtp_cands
                .iter()
                .filter(|c| c.ok == Some(true))
                .map(|c| std::path::Path::new(&c.rel).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
                .collect();
            pane.mtp_note = format!("MTP off  •  {} drafter file(s) found, {} matching", pane.mtp_cands.len(), ok.len())
                + &if ok.is_empty() { String::new() } else { format!(": {}", ok.join(", ")) };
        }
    }

    /// Extra weight files loaded on the GPU: mmproj + MTP drafter.
    pub fn mmproj_bytes(&self, p: usize, s: &Opts) -> f64 {
        let mut total = self.panes[p].mtp_bytes(s);
        let mm = gs(s, "mmproj", "(none)");
        if mm != "(none)" {
            if let Some(mp) = self.model_path(p) {
                total += file_size(&mp.parent().unwrap_or(&mp).join(mm));
            }
        }
        total
    }

    // ================================================================ stats ==
    pub fn stats_ids(&self, p: usize) -> Option<(String, String)> {
        let rel = &self.panes[p].model_var;
        if rel.is_empty() || !(self.panes[p].device.is_some() || self.dual_active(p)) {
            return None;
        }
        Some((self.stats_key(p), rel.clone()))
    }

    pub fn refresh_life(&mut self, p: usize) {
        let Some(ids) = self.stats_ids(p) else {
            self.panes[p].set_status("life", "—");
            return;
        };
        let st = self.settings.get_stats(&ids.0, &ids.1);
        let mut txt = stats_summary(st.as_ref());
        if let Some(st) = &st {
            if self.panes[p].proc.is_none() && gs(st, "fp", "") != stats_fingerprint(&self.current_options(p)) {
                txt = format!("settings changed - these stats reset when you Load:  {txt}");
            }
        }
        self.panes[p].set_status("life", txt);
    }

    pub fn stats_begin_load(&mut self, p: usize, opts: &Opts) {
        let ids = self.stats_ids(p);
        self.panes[p].stats_ids_loaded = ids.clone();
        self.panes[p].req = Map::new();
        let Some(ids) = ids else { return };
        let mut st = match self.settings.get_stats(&ids.0, &ids.1) {
            Some(st) if gs(&st, "fp", "") == stats_fingerprint(opts) => st,
            old => {
                // new setup (or settings changed): start over, but keep the old totals in history.jsonl
                if let Some(old) = old {
                    if gi(&old, "loads", 0) > 0 || gi(&old, "requests", 0) > 0 {
                        self.history_event(p, "retired", json!({"stats": old}));
                    }
                }
                super::mtp::new_stats(opts)
            }
        };
        st.insert("loads".into(), (gi(&st, "loads", 0) + 1).into());
        st.insert("last".into(), now_iso().into());
        self.settings.set_stats(&ids.0, &ids.1, st);
        self.refresh_life(p);
        let pane = &mut self.panes[p];
        pane.dash_reqs.clear();
        pane.dash_metrics.clear();
        pane.dash_active = None;
        pane.dash_progress = None;
        pane.dash_cancelled.clear();
        pane.dash_gen = None;
        pane.dash_slot_ctx = None;
        pane.dash_load_t = Some(super::telemetry::unix_now());
        pane.dash_ready_t = None;
        self.history_event(p, "load", json!({"opts": opts}));
    }

    pub fn show_cache(&mut self, p: usize, cached: i64, total: i64) {
        let pct = 100.0 * cached as f64 / total as f64;
        let mut txt = format!(
            "{:.0}% reused  ({} of {} prompt tokens from cache, {} processed)",
            pct,
            fmt_thousands(cached),
            fmt_thousands(total),
            fmt_thousands(total - cached)
        );
        if let Some(ids) = &self.panes[p].stats_ids_loaded {
            if let Some(st) = self.settings.get_stats(&ids.0, &ids.1) {
                if gi(&st, "cache_total", 0) != 0 {
                    let ct = gi(&st, "cache_tokens", 0) + cached;
                    let tt = gi(&st, "cache_total", 0) + total;
                    txt += &format!("   •   this setup overall {:.0}%", 100.0 * ct as f64 / tt as f64);
                }
            }
        }
        if pct < 50.0 && total > 8192 {
            txt += "   ⚠ most of the prompt was re-read";
        }
        self.panes[p].set_status("cache", txt);
    }

    pub fn stats_add_f(&mut self, p: usize, field: &str, v: f64) {
        if let Some(ids) = self.panes[p].stats_ids_loaded.clone() {
            if let Some(mut st) = self.settings.get_stats(&ids.0, &ids.1) {
                let cur = gf(&st, field, 0.0);
                st.insert(field.into(), json!(((cur + v) * 10.0).round() / 10.0));
                self.settings.set_stats(&ids.0, &ids.1, st);
            }
        }
    }

    pub fn stats_bump(&mut self, p: usize, field: &str) {
        if let Some(ids) = self.panes[p].stats_ids_loaded.clone() {
            if let Some(mut st) = self.settings.get_stats(&ids.0, &ids.1) {
                st.insert(field.into(), (gi(&st, field, 0) + 1).into());
                self.settings.set_stats(&ids.0, &ids.1, st);
                self.refresh_life(p);
            }
        }
    }

    /// Collect llama-server's end-of-request timing lines into one record.
    pub fn stats_line(&mut self, p: usize, line: &str) {
        self.dash_line(p, line);
        let re = super::kobold::lre();
        if let Some(m) = re.prompt.captures(line) {
            let mut r = Map::new();
            r.insert("p_ms".into(), json!(m[1].parse::<f64>().unwrap_or(0.0)));
            r.insert("p_tok".into(), json!(m[2].parse::<i64>().unwrap_or(0)));
            self.panes[p].req = r;
            return;
        }
        if let Some((ms, tok, _)) = super::kobold::eval_match(line) {
            self.panes[p].req.insert("g_ms".into(), json!(ms));
            self.panes[p].req.insert("g_tok".into(), json!(tok));
            return;
        }
        if let Some(m) = re.accept.captures(line) {
            if let (Some(a), Some(g)) = (m.get(2), m.get(3)) {
                self.panes[p].req.insert("acc".into(), json!(a.as_str().parse::<i64>().unwrap_or(0)));
                self.panes[p].req.insert("gen".into(), json!(g.as_str().parse::<i64>().unwrap_or(0)));
                return;
            }
        }
        if let Some(m) = re.req_done.captures(line) {
            if let Some(n) = m.get(1) {
                self.panes[p].req.insert("n_end".into(), json!(n.as_str().parse::<i64>().unwrap_or(0)));
            }
            let r = std::mem::take(&mut self.panes[p].req);
            // "release: id 0 | task 771 | stop processing ..."
            let task = super::dash::task_id(line);
            let cancelled = task.map(|t| self.panes[p].dash_cancelled.contains(&t)).unwrap_or(false);
            self.panes[p].dash_gen = None;
            if r.contains_key("g_tok") {
                // finished with its timing lines: a real request, even if some other task was cancelled
                self.stats_record(p, &r);
            } else if self.panes[p].dash_active.is_some() || cancelled {
                // released without timing lines: the client went away / cancelled it
                self.stats_bump(p, "aborts");
                self.dash_request(p, &r, None, true);
            }
            if let Some(t) = task {
                self.panes[p].dash_cancelled.retain(|c| *c != t);
            }
        }
    }

    pub fn stats_record(&mut self, p: usize, r: &Map<String, Value>) {
        // prompt-cache reuse for this request: total prompt = context at the end - generated
        let mut cache = None;
        if r.contains_key("n_end") && r.contains_key("g_tok") {
            let total = gi(r, "n_end", 0) - gi(r, "g_tok", 0);
            if total > 0 {
                let cached = (total - gi(r, "p_tok", 0)).max(0);
                cache = Some((cached, total));
                self.show_cache(p, cached, total);
            }
        }
        if r.contains_key("g_tok") {
            self.dash_request(p, r, cache, false);
        }
        let Some(ids) = self.panes[p].stats_ids_loaded.clone() else { return };
        let Some(mut st) = self.settings.get_stats(&ids.0, &ids.1) else { return };
        if !r.contains_key("g_tok") {
            return;
        }
        let add_i = |st: &mut Opts, k: &str, v: i64| {
            let cur = gi(st, k, 0);
            st.insert(k.into(), (cur + v).into());
        };
        let add_f = |st: &mut Opts, k: &str, v: f64| {
            let cur = gf(st, k, 0.0);
            st.insert(k.into(), json!(cur + v));
        };
        if let Some((c, t)) = cache {
            add_i(&mut st, "cache_tokens", c);
            add_i(&mut st, "cache_total", t);
        }
        add_i(&mut st, "requests", 1);
        add_i(&mut st, "prompt_tokens", gi(r, "p_tok", 0));
        add_f(&mut st, "prompt_ms", gf(r, "p_ms", 0.0));
        if gi(r, "p_tok", 0) >= 1024 {
            // prompt speed only from real (uncached) reads
            add_i(&mut st, "big_prompt_tokens", gi(r, "p_tok", 0));
            add_f(&mut st, "big_prompt_ms", gf(r, "p_ms", 0.0));
        }
        let g_tok = gi(r, "g_tok", 0);
        let g_ms = gf(r, "g_ms", 0.0);
        add_i(&mut st, "gen_tokens", g_tok);
        add_f(&mut st, "gen_ms", g_ms);
        if g_tok >= 64 && g_ms > 0.0 {
            // ignore tiny replies for min/max
            let tps = g_tok as f64 / (g_ms / 1000.0);
            let mn = st.get("gen_tps_min").and_then(|v| v.as_f64());
            let mx = st.get("gen_tps_max").and_then(|v| v.as_f64());
            st.insert("gen_tps_min".into(), json!(mn.map(|m| m.min(tps)).unwrap_or(tps)));
            st.insert("gen_tps_max".into(), json!(mx.map(|m| m.max(tps)).unwrap_or(tps)));
        }
        add_i(&mut st, "draft_acc", gi(r, "acc", 0));
        add_i(&mut st, "draft_gen", gi(r, "gen", 0));
        st.insert("last".into(), now_iso().into());
        self.settings.set_stats(&ids.0, &ids.1, st);
        self.refresh_life(p);
    }

    // ============================================================ estimates ==
    pub fn update_estimate(&mut self, p: usize) {
        if self.panes[p].is_rad() {
            self.r_update_estimate(p);
            return;
        }
        if self.panes[p].is_vllm() {
            self.v_update_estimate(p);
            return;
        }
        let Some(info) = self.panes[p].info_g() else {
            self.panes[p].est_text = if self.info_pending(p) { "Estimate: reading model metadata…" } else { "Estimate: select a model" }.into();
            self.panes[p].est_pct = 0.0;
            return;
        };
        if self.dual_active(p) {
            self.update_estimate_dual(p);
            return;
        }
        let mut s = self.current_options(p);
        let mtp_bytes = self.panes[p].mtp_bytes(&s);
        let mm_bytes = self.mmproj_bytes(p, &s) - mtp_bytes;
        let (adj, calib) = self.calib_adjust(p);
        let oh = gf(self.settings.g(), "overhead_mib", 300.0);
        let mut auto_note = String::new();
        if gi(&s, "ngl", 999) < 0 && self.panes[p].device.is_some() {
            let n_eff = info.n_layer.saturating_sub(info.nextn) as i64;
            let budget = self.free_budget(p);
            match self.auto_layers(&info, &s, mm_bytes + mtp_bytes, oh, adj, budget) {
                None => {
                    auto_note = "auto: not even 0 GPU layers fit at this context - lower context".into();
                    s.insert("ngl".into(), 0.into());
                }
                Some(k) => {
                    s.insert("ngl".into(), k.into());
                    auto_note = format!(
                        "auto: ≈ {}/{} layers on GPU{} ({} decides at load)",
                        k.min(n_eff),
                        n_eff,
                        if k > n_eff { " + output" } else { "" },
                        if self.panes[p].is_kobold() { "KoboldCpp" } else { "llama.cpp" }
                    );
                }
            }
        }
        let e = est_simple(&info, &s, mm_bytes + mtp_bytes, oh, adj);
        let total_dev = self.panes[p].device.as_ref().map(|d| d.total_mib as f64 * MIB).unwrap_or(0.0);
        let mut txt = format!("Estimated VRAM ≈ {:.2} GiB", e.total / GIB)
            + &if total_dev > 0.0 { format!(" of {:.1} GiB", total_dev / GIB) } else { String::new() }
            + &format!("\nweights {:.2}  •  KV {:.2}  •  ", e.weights / GIB, e.kv / GIB)
            + &if e.state != 0.0 { format!("state {:.2}  •  ", e.state / GIB) } else { String::new() }
            + &format!("compute {:.2}", e.compute / GIB)
            + &if mm_bytes != 0.0 { format!("  •  mmproj {:.2}", mm_bytes / GIB) } else { String::new() }
            + &if mtp_bytes != 0.0 { format!("  •  MTP {:.2}", mtp_bytes / GIB) } else { String::new() }
            + if gs(&s, "mtp", "") == "(built-in)" { "  •  incl. MTP KV copy" } else { "" }
            + &format!("  •  overhead {:.2}", e.overhead / GIB);
        let mut warn = vec![];
        if gs(&s, "kv", "f16") != "f16" && gs(&s, "fa", "on") == "off" {
            warn.push("quantized KV cache needs flash attention on/auto".to_string());
        }
        if !auto_note.is_empty() {
            txt += &format!("\n{auto_note}");
        }
        txt += &self.moe_tips(p, &s);
        if info.n_ctx_train != 0 && gi(&s, "ctx", 0) > info.n_ctx_train {
            warn.push(format!("ctx exceeds training ctx ({})", fmt_thousands(info.n_ctx_train)));
        }
        if let Some(c) = &calib {
            let pctx = c.get("params").and_then(|v| v.as_object()).map(|o| gi(o, "ctx", 0)).unwrap_or(0);
            txt += &format!("\n✓ calibrated from a real load at ctx {} (adjusted {:+.2} GiB)", fmt_thousands(pctx), adj / GIB);
            if let Some(b) = c.get("_borrowed").and_then(|v| v.as_str()) {
                txt += &format!(
                    " - borrowed from {} (same architecture)",
                    std::path::Path::new(b).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                );
            }
        } else if info.swa_guessed {
            warn.push(format!("SWA layout guessed ({}); load once to calibrate", info.swa_note));
        }
        if !warn.is_empty() {
            txt += &format!("\n⚠ {}", warn.join("; "));
        }
        let pct = if total_dev > 0.0 { 100.0 * e.total / total_dev } else { 0.0 };
        let pane = &mut self.panes[p];
        pane.est_text = txt;
        pane.est_pct = pct.min(100.0);
        pane.est_color = if pct < 85.0 { "#2e9e44" } else if pct < 98.0 { "#d08a00" } else { "#d0342c" }.into();
        pane.est_over = pct > 100.5;
    }

    /// This model's calibration, or else the newest one recorded on the same card(s) for a model
    /// with the same architecture. Returns (record, source_info, borrowed_from).
    pub fn calib_lookup(&mut self, p: usize, dev_key: &str) -> CalibHit {
        let rel = self.panes[p].model_var.clone();
        if let Some(c) = self.settings.get_calib(dev_key, &rel) {
            return (Some(c), self.panes[p].info.clone(), None);
        }
        let Some(info) = self.panes[p].info.clone() else { return (None, None, None) };
        let ck = (dev_key.to_string(), rel.clone());
        if let Some(hit) = self.panes[p].borrow_cache.get(&ck) {
            return hit.clone();
        }
        let mine = info.sig();
        let mut best: CalibHit = (None, None, None);
        let root = self.models_root(p);
        let backend = self.panes[p].backend.clone();
        for (key, rec) in self.settings.calib_all() {
            let (dk, orel) = match key.split_once('|') {
                Some(x) => x,
                None => continue,
            };
            if dk != dev_key || orel == rel {
                continue;
            }
            let Some(rec) = rec.as_object().cloned() else { continue };
            // an empty signature (written before the metadata had loaded) counts as unknown
            let sig = rec.get("sig").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from);
            if let Some(s) = &sig {
                if *s != mine {
                    continue;
                }
            }
            let path = root.join(orel);
            let src = match self.info_cache_get(&path) {
                Some(s) => s,
                None => match Info::load(&path, &backend) {
                    Ok(s) => {
                        self.info_cache_put(&path, s.clone());
                        s
                    }
                    Err(_) => continue,
                },
            };
            let sig = match sig {
                Some(s) => s,
                None => {
                    let s = src.sig();
                    self.settings.calib_set_sig(&key, &s);
                    self.settings.save();
                    s
                }
            };
            let newer = best.0.as_ref().map(|b| gs(&rec, "when", "") > gs(b, "when", "")).unwrap_or(true);
            if sig == mine && newer {
                best = (Some(rec), Some(src), Some(orel.to_string()));
            }
        }
        self.panes[p].borrow_cache.insert(ck, best.clone());
        best
    }

    /// Fixed correction (bytes) learned from a real load of this model on this GPU.
    ///
    /// The per-token KV math comes straight from the GGUF metadata, so the remaining error is
    /// mostly constant (compute buffers, runtime overhead). It is applied as an offset rather
    /// than a KV multiplier, so changing the context size still scales correctly.
    pub fn calib_adjust(&mut self, p: usize) -> (f64, Option<Opts>) {
        if self.panes[p].is_vllm() || self.panes[p].is_rad() {
            return (0.0, None);
        }
        if self.panes[p].info.is_none() || self.panes[p].device.is_none() || self.dual_active(p) {
            return (0.0, None);
        }
        let name = self.panes[p].device.as_ref().unwrap().name.clone();
        let key = self.calib_key(p, &name);
        let (c, src, borrowed) = self.calib_lookup(p, &key);
        let Some(c) = c else { return (0.0, None) };
        let Some(measured) = c.get("measured").and_then(|v| v.as_f64()) else { return (0.0, None) };
        let Some(src) = src.and_then(|s| s.g().cloned()) else { return (0.0, None) };
        let mut prm = default_model();
        if let Some(Value::Object(o)) = c.get("params") {
            for (k, v) in o {
                prm.insert(k.clone(), v.clone());
            }
        }
        if gi(&prm, "ngl", 0) < 0 {
            return (0.0, None); // ignore any calibration recorded from an auto-layer load
        }
        let oh = gf(self.settings.g(), "overhead_mib", 300.0);
        let e = est_simple(&src, &prm, gf(&c, "mm", 0.0), oh, 0.0);
        let adj = calib_clamp(measured - e.total, 4.0 * GIB);
        let mut c2 = c.clone();
        c2.insert("_borrowed".into(), borrowed.map(Value::from).unwrap_or(Value::Null));
        (adj, Some(c2))
    }

    pub fn moe_tips(&mut self, p: usize, s: &Opts) -> String {
        let Some(info) = self.panes[p].info_g() else { return String::new() };
        if info.n_expert == 0 || gi(s, "n_cpu_moe", 0) <= 0 {
            return String::new();
        }
        let mut tips = vec![];
        if gi(s, "ubatch", 512) < 1024 {
            tips.push("UBatch 1024 (~2x faster prompts: experts are streamed half as often)".to_string());
        }
        let mtp = gs(s, "mtp", "");
        let f = self.mtp_flag(p);
        if (mtp.is_empty() || mtp == "(off)") && info.nextn > 0 && self.supports(p, f) {
            tips.push("MTP (built-in), Draft max 3 (~1.5-2x faster generation)".into());
        }
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        let phys = (cpus / 2).max(1) as i64;
        if gi(s, "threads", 0) > phys {
            tips.push(format!("Threads {phys} (physical cores) - hyperthreads measured no gain"));
        }
        if tips.is_empty() { String::new() } else { format!("\n💡 Experts in system RAM - try: {}", tips.join("; ")) }
    }

    // ============================================================ dual mode ==
    pub fn refresh_dual_ui(&mut self, p: usize) {
        if self.panes[p].side != "left" {
            return;
        }
        if self.dual_active(p) {
            let devs = self.dual_devices();
            self.panes[p].show_dual = true;
            self.panes[p].dual_mode_values = if self.panes[p].is_rad() {
                vec!["tensor".into()]
            } else if self.panes[p].is_vllm() {
                VLLM_SPLIT_MODES.iter().map(|s| s.to_string()).collect()
            } else {
                self.split_modes_for(p)
            };
            self.panes[p].gpu_name = devs.iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(" + ") + "   [dual]";
            self.panes[p].dual_note = "Dual GPU mode: one model across both cards".into();
        } else {
            self.panes[p].show_dual = false;
            if let Some(d) = &self.panes[p].device {
                self.panes[p].gpu_name = format!("{}   [{}]", d.name, d.id);
            }
            self.panes[p].dual_note.clear();
        }
    }

    pub fn dual_first_values(&self) -> Vec<String> {
        self.dual_devices().iter().map(|d| dev_label(d, 0)).collect()
    }

    pub fn dual_budgets(&self, p: usize) -> HashMap<String, f64> {
        let margin = gf(self.settings.g(), "fit_margin_mib", 1536.0) * MIB;
        let mut out = HashMap::new();
        let bases: HashMap<String, Option<f64>> = if self.panes[p].proc.is_some() {
            self.panes[p].mon.iter().map(|m| (m.dev.id.clone(), m.base)).collect()
        } else {
            HashMap::new()
        };
        for d in self.dual_devices() {
            let (used, total, _) = gpu_stats(Some(&d));
            let total = total.filter(|t| *t != 0.0).unwrap_or(d.total_mib as f64 * MIB);
            let used = match bases.get(&d.id) {
                Some(b) => b.unwrap_or(0.0),
                None => used.unwrap_or(0.0),
            };
            out.insert(d.id.clone(), total - used - margin);
        }
        out
    }

    pub fn dual_plan(&mut self, p: usize, s: &Opts) -> Option<DualPlan> {
        let info = self.panes[p].info_g()?;
        let mut s = s.clone();
        let n_eff = info.n_layer.saturating_sub(info.nextn) as i64;
        let ngl = gi(&s, "ngl", 999);
        if ngl < 0 || ngl < n_eff + 1 {
            s.insert("ngl".into(), (n_eff + 1).into()); // dual mode keeps every layer on a GPU
        }
        let mtp_b = self.panes[p].mtp_bytes(&s);
        let mm_b = self.mmproj_bytes(p, &s) - mtp_b;
        let manual = Manual { mode: gs(&s, "dual_mode", ""), ts: gs(&s, "dual_ts", ""), first: gs(&s, "dual_first", "") };
        let strat = gs(&s, "dual_strategy", "Auto Balance");
        let adjust = self.dual_adjust(p);
        let budgets = self.dual_budgets(p);
        let oh = gf(self.settings.g(), "overhead_mib", 300.0);
        let devs = self.dual_devices();
        if devs.len() != 2 {
            return None;
        }
        Some(plan_dual(
            &info,
            &s,
            &devs,
            &budgets,
            &strat,
            (mm_b, mtp_b),
            oh,
            if strat == "Manual" { Some(&manual) } else { None },
            &adjust,
        ))
    }

    /// Per-card corrections learned from earlier clean dual-mode loads of this model.
    pub fn dual_adjust(&mut self, p: usize) -> HashMap<String, f64> {
        let base = self.dev_base(p);
        let key = self.calib_key(p, &base);
        let (c, src, borrowed) = self.calib_lookup(p, &key);
        self.panes[p].dual_borrowed = borrowed;
        let mut out = HashMap::new();
        let Some(c) = c else { return out };
        let Some(cards) = c.get("cards").and_then(|v| v.as_object()).cloned() else { return out };
        if self.panes[p].info.is_none() {
            return out;
        }
        let Some(src) = src.and_then(|s| s.g().cloned()) else { return out };
        let mut prm = default_model();
        let params = c.get("params").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        for (k, v) in &params {
            prm.insert(k.clone(), v.clone());
        }
        if !params.contains_key("dual_moe_spread") {
            prm.insert("dual_moe_spread".into(), false.into()); // recorded before spread placement existed
        }
        let oh = gf(self.settings.g(), "overhead_mib", 300.0);
        let devs: HashMap<String, Device> = self.dual_devices().into_iter().map(|d| (d.id.clone(), d)).collect();
        let side: Vec<f64> = c
            .get("side")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
            .unwrap_or_else(|| vec![0.0, 0.0]);
        let side = (side.first().cloned().unwrap_or(0.0), side.get(1).cloned().unwrap_or(0.0));
        let order: Vec<String> = c
            .get("order")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let card = |id: &str| cards.get(id).and_then(|v| v.as_f64()).unwrap_or(0.0);
        if gs(&c, "mode", "") == "none" {
            let Some(dev_id) = order.first() else { return out };
            let mut p2 = prm.clone();
            p2.insert("ngl".into(), 999.into());
            let e = est_simple(&src, &p2, side.0 + side.1, oh, 0.0);
            out.insert(format!("{dev_id}:single"), calib_clamp(card(dev_id) - e.total, 3.0 * GIB));
            return out;
        }
        let ord: Vec<Device> = order.iter().filter_map(|i| devs.get(i).cloned()).collect();
        if ord.len() != order.len() || ord.len() < 2 {
            return HashMap::new();
        }
        let k = gi(&c, "k", 0).max(0) as usize;
        let e = dual_estimate(&src, &prm, &ord, k, side, oh, &HashMap::new());
        for (i, d) in ord.iter().enumerate().take(2) {
            out.insert(d.id.clone(), calib_clamp(card(&d.id) - e[i].total, 6.0 * GIB));
        }
        out
    }

    pub fn compat_lines(&mut self, p: usize) -> Vec<String> {
        let devs = self.dual_devices();
        let links: Vec<(Option<u32>, Option<u32>)> = devs.iter().map(pcie_link).collect();
        let mut out = vec![devs
            .iter()
            .zip(&links)
            .map(|(d, (g, w))| {
                format!("{}: ", d.short_name())
                    + &match g {
                        Some(g) => format!("PCIe {g}.0 x{}", w.map(|x| x.to_string()).unwrap_or("?".into())),
                        None => "PCIe ?".into(),
                    }
            })
            .collect::<Vec<_>>()
            .join(", ")];
        let backends: BTreeSet<&String> = devs.iter().map(|d| &d.backend).collect();
        if backends.len() > 1 {
            out.push("⚠ different backends - both cards must use the same llama.cpp backend".into());
        }
        let mut t: Vec<i64> = devs.iter().map(|d| d.total_mib).collect();
        t.sort();
        if t.len() == 2 && t[1] as f64 > t[0] as f64 * 1.25 {
            out.push(format!(
                "Mismatched VRAM ({:.0} vs {:.0} GiB): split weighted per card; larger card gets the output layer, logits and MTP",
                t[0] as f64 / 1024.0,
                t[1] as f64 / 1024.0
            ));
        }
        let slow = links.iter().any(|(g, w)| matches!(g, Some(g) if *g < 4 || w.unwrap_or(16) < 16));
        let modes = self.split_modes_for(p);
        let mut mtxt =
            "split modes in this build: ".to_string() + &modes.iter().filter(|m| *m != "none").cloned().collect::<Vec<_>>().join(", ");
        if modes.iter().any(|m| m == "tensor") {
            mtxt += " (tensor = experimental";
            mtxt += if slow { ", not advised on this PCIe link)" } else { ")" };
        }
        out.push(mtxt);
        out
    }

    pub fn update_estimate_dual(&mut self, p: usize) {
        let s = self.current_options(p);
        let Some(plan) = self.dual_plan(p, &s) else { return };
        let info = self.panes[p].info_g().unwrap();
        self.panes[p].last_plan = Some(plan.clone());
        let mut lines = vec![];
        let mut worst: f64 = 0.0;
        let n_eff = info.n_layer.saturating_sub(info.nextn);
        let bud = self.dual_budgets(p);
        let margin = gf(self.settings.g(), "fit_margin_mib", 1536.0) * MIB;
        let mut over_card = false;
        let mut in_margin = false;
        let mut flag = |dev_id: &str, total: f64| -> &'static str {
            let b = bud.get(dev_id).cloned().unwrap_or(0.0);
            if total > b + margin {
                over_card = true;
                return "  ⛔ exceeds the card";
            }
            if total > b {
                in_margin = true;
                return "  ⚠ inside safety margin";
            }
            ""
        };
        let mut head;
        if plan.mode == "none" {
            let d = plan.single.clone().unwrap();
            let e = plan.est[1].clone().or(plan.est[0].clone()).unwrap_or_default();
            let pct = 100.0 * e.total / (d.total_mib as f64 * MIB);
            worst = pct;
            head = format!("Plan: single GPU on {} - {}", d.name, plan.note);
            lines.push(
                format!("{}: ≈ {:.2} / {:.1} GiB ({:.0}%)", d.name, e.total / GIB, d.total_mib as f64 / 1024.0, pct)
                    + flag(&d.id, e.total),
            );
        } else {
            head = format!("Plan: {} split ({}) - {}", plan.mode, plan.ts.clone().unwrap_or_default(), plan.note);
            if let (Some(cs), "layer") = (&plan.cpu_set, plan.mode.as_str()) {
                let k = plan.k.unwrap_or(0);
                let ca = cs.iter().filter(|i| **i < k).count();
                head += &format!(
                    "\nCPU MoE experts: {} layers on {} side, {} on {} side",
                    ca,
                    plan.order[0].short_name(),
                    cs.len() - ca,
                    plan.order[1].short_name()
                );
            }
            for (i, d) in plan.order.iter().enumerate() {
                let e = plan.est[i].clone().unwrap_or_default();
                let pct = 100.0 * e.total / (d.total_mib as f64 * MIB);
                worst = worst.max(pct);
                let lay = if plan.mode == "layer" {
                    match plan.k {
                        Some(k) => {
                            if i == 0 {
                                format!("layers 0-{}", k as i64 - 1)
                            } else {
                                format!("layers {}-{} + output", k, n_eff as i64 - 1)
                            }
                        }
                        None => String::new(),
                    }
                } else {
                    "share of every layer".into()
                };
                lines.push(
                    format!(
                        "{}: {}  ≈ {:.2} / {:.1} GiB ({:.0}%; KV {:.2}, compute {:.2})",
                        d.name,
                        lay,
                        e.total / GIB,
                        d.total_mib as f64 / 1024.0,
                        pct,
                        e.kv / GIB,
                        e.compute / GIB
                    ) + flag(&d.id, e.total),
                );
            }
        }
        let mut warn = vec![];
        let ngl = gi(&s, "ngl", 999);
        if ngl >= 0 && ngl < n_eff as i64 + 1 {
            warn.push("GPU layers ignored in dual mode (all layers go on the GPUs)".to_string());
        }
        if over_card {
            warn.push("won't fit in VRAM - lower context, raise CPU MoE layers, or press Fit".into());
        } else if in_margin {
            warn.push(
                "fits, but eats into the safety margin (Settings → Fit margin) - the driver may evict memory to system RAM under load; +1 CPU MoE layer or Fit gives headroom"
                    .into(),
            );
        }
        if info.n_ctx_train != 0 && gi(&s, "ctx", 0) > info.n_ctx_train {
            warn.push(format!("ctx exceeds training ctx ({})", fmt_thousands(info.n_ctx_train)));
        }
        if plan.mode == "tensor" || plan.mode == "row" {
            warn.push(format!("{} split estimate is rough (experimental mode)", plan.mode));
        }
        let mut txt = format!("Estimated VRAM per card\n{}", lines.join("\n"));
        txt += &self.moe_tips(p, &s);
        let adj = self.dual_adjust(p);
        if !adj.is_empty() {
            let mut vals: Vec<(&String, &f64)> = adj.iter().collect();
            vals.sort_by(|a, b| a.0.cmp(b.0));
            txt += &format!(
                "\n✓ calibrated from a real dual load ({})",
                vals.iter().map(|(_, v)| format!("{:+.2} GiB", *v / GIB)).collect::<Vec<_>>().join(", ")
            );
            if let Some(b) = &self.panes[p].dual_borrowed {
                txt += &format!(
                    " - borrowed from {} (same architecture)",
                    std::path::Path::new(b).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                );
            }
        }
        if !warn.is_empty() {
            txt += &format!("\n⚠ {}", warn.join("; "));
        }
        let compat = self.compat_lines(p);
        let pane = &mut self.panes[p];
        pane.est_text = txt;
        pane.dual_info = format!("{head}\n{}", compat.join("\n"));
        pane.est_pct = worst.min(100.0);
        pane.est_over = over_card;
        pane.est_color = if over_card || worst >= 98.0 {
            "#d0342c"
        } else if in_margin || worst >= 85.0 {
            "#d08a00"
        } else {
            "#2e9e44"
        }
        .into();
    }

    pub fn fit_dual(&mut self, p: usize) {
        let s = self.current_options(p);
        let info = self.panes[p].info_g().unwrap();
        let cap = if info.n_ctx_train != 0 { info.n_ctx_train } else { 131072 };
        let cap = (cap / 1024 * 1024).max(1024);
        let ok = |app: &mut App, ctx: i64| {
            let mut s2 = s.clone();
            s2.insert("ctx".into(), ctx.into());
            app.dual_plan(p, &s2).map(|pl| pl.fits).unwrap_or(false)
        };
        if !ok(self, 1024) {
            self.msg(
                "warning",
                "Even 1024 context doesn't fit across both GPUs with these settings. Try CPU MoE layers, a smaller quant or q8_0/q4_0 KV cache.",
            );
            return;
        }
        let (mut lo, mut hi) = (1i64, cap / 1024);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if ok(self, mid * 1024) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let prev_ctx = gi(&s, "ctx", 8192);
        self.set_var_notify(p, "ctx", Value::from((lo * 1024).to_string()));
        let mut s2 = s.clone();
        s2.insert("ctx".into(), (lo * 1024).into());
        let plan = self.dual_plan(p, &s2).unwrap();
        self.set_top_status(
            format!("Dual Fit ({}): context {}", gs(&s, "dual_strategy", ""), fmt_thousands(lo * 1024))
                + if lo * 1024 == cap { " (model's max)" } else { "" }
                + &if plan.mode != "none" {
                    format!(", {} split {}", plan.mode, plan.ts.clone().unwrap_or_default())
                } else {
                    format!(", single GPU {}", plan.single.as_ref().map(|d| d.name.clone()).unwrap_or_default())
                },
        );
        if lo * 1024 < cap && lo * 1024 < prev_ctx {
            // Fit lowered your context: say why
            self.explain_fit(p, &s, lo * 1024);
        }
    }

    /// Show why Fit stopped where it did: the limiting card and every input.
    pub fn explain_fit(&mut self, p: usize, s: &Opts, ctx: i64) {
        let margin = gf(self.settings.g(), "fit_margin_mib", 1536.0) * MIB;
        let adj = self.dual_adjust(p);
        let mut s2 = s.clone();
        s2.insert("ctx".into(), (ctx + 1024).into());
        let Some(nxt) = self.dual_plan(p, &s2) else { return };
        let bud = self.dual_budgets(p);
        let mut lines = vec![format!("Fit chose {} context. What it used:", fmt_thousands(ctx)), String::new()];
        let pairs: Vec<(Device, Est)> = if nxt.mode != "none" {
            nxt.order.iter().cloned().zip(nxt.est.iter().map(|e| e.clone().unwrap_or_default())).collect()
        } else {
            vec![(nxt.single.clone().unwrap(), nxt.est[1].clone().or(nxt.est[0].clone()).unwrap_or_default())]
        };
        for (d, e) in pairs {
            let (used, total, _) = gpu_stats(Some(&d));
            let total = total.filter(|t| *t != 0.0).unwrap_or(d.total_mib as f64 * MIB);
            let in_use = used.unwrap_or(0.0);
            let b = bud.get(&d.id).cloned().unwrap_or(0.0);
            let limiting = e.total > b;
            lines.push(format!("{}{}", d.name, if limiting { "   ← LIMITING CARD" } else { "" }));
            lines.push(format!(
                "   total {:.1}  −  already in use {:.2}  −  margin {:.2}  =  {:.2} GiB usable",
                total / GIB,
                in_use / GIB,
                margin / GIB,
                b / GIB
            ));
            lines.push(
                format!(
                    "   model needs at {}: {:.2} GiB  (KV {:.2}, compute {:.2})",
                    fmt_thousands(ctx + 1024),
                    e.total / GIB,
                    e.kv / GIB,
                    e.compute / GIB
                ) + &adj.get(&d.id).map(|a| format!("  incl. calibration {:+.2}", a / GIB)).unwrap_or_default(),
            );
            lines.push(String::new());
        }
        let per_tok = self.kv_per_token(p, s);
        lines.extend([
            "If 'already in use' is higher than normal (browser, desktop, a leftover".to_string(),
            "process), close it and press Fit again.".into(),
            format!("Each GiB is worth about {:.0}k tokens of context for this model.", 1024.0 * MIB / per_tok.max(1.0) / 1000.0),
            "A context above Fit's choice can still load fine; it uses the safety".into(),
            "margin, and the launcher warns you if the driver ever evicts memory.".into(),
        ]);
        self.set_why(format!("{APP_NAME}: why Fit chose {}", fmt_thousands(ctx)), lines.join("\n"));
    }

    pub fn kv_per_token(&self, p: usize, s: &Opts) -> f64 {
        let Some(info) = self.panes[p].info_g() else { return 1.0 };
        let mut a = s.clone();
        a.insert("ctx".into(), 1024.into());
        let mut b = s.clone();
        b.insert("ctx".into(), 2048.into());
        let e1 = est_simple(&info, &a, 0.0, 0.0, 0.0).kv;
        let e2 = est_simple(&info, &b, 0.0, 0.0, 0.0).kv;
        (e2 - e1).max(1.0) / 1024.0
    }

    /// VRAM available to this pane's model, minus the Fit margin.
    pub fn free_budget(&self, p: usize) -> f64 {
        let dev = self.panes[p].device.as_ref();
        let (used, total, _) = gpu_stats(dev);
        let total = total.unwrap_or(dev.map(|d| d.total_mib as f64 * MIB).unwrap_or(0.0));
        let mut used = used.unwrap_or(0.0);
        if self.panes[p].proc.is_some() {
            if let Some(b) = self.panes[p].vram_base {
                used = b; // don't count our own running model
            }
        }
        total - used - gf(self.settings.g(), "fit_margin_mib", 768.0) * MIB
    }

    /// Mimic llama.cpp's auto-fit: most GPU layers that fit at this context (None = none).
    pub fn auto_layers(&self, info: &ModelInfo, s: &Opts, mm: f64, oh: f64, adj: f64, budget: f64) -> Option<i64> {
        let n_eff = info.n_layer.saturating_sub(info.nextn) as i64;
        let (mut lo, mut hi) = (-1i64, n_eff + 1);
        while lo < hi {
            let mid = floordiv(lo + hi + 1, 2);
            let mut s2 = s.clone();
            s2.insert("ngl".into(), mid.into());
            if est_simple(info, &s2, mm, oh, adj).total <= budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo < 0 { None } else { Some(lo) }
    }

    pub fn fit_context(&mut self, p: usize) {
        if self.panes[p].proc.is_none() && self.info_pending(p) {
            self.panes[p].fit_after_info = true;
            self.set_top_status("Reading model metadata… Fit will run as soon as it's done.");
            return;
        }
        self.panes[p].fit_after_info = false;
        if self.panes[p].is_rad() {
            self.r_fit(p);
            return;
        }
        if self.panes[p].is_vllm() {
            self.v_fit(p);
            return;
        }
        if self.dual_active(p) && self.panes[p].proc.is_none() && self.panes[p].info_g().is_some() {
            self.fit_dual(p);
            return;
        }
        if self.panes[p].proc.is_some() {
            self.msg("info", "Stop the model first - Fit measures free VRAM on this GPU.");
            return;
        }
        let (Some(info), true) = (self.panes[p].info_g(), self.panes[p].device.is_some()) else {
            self.msg("info", "Select a model (and assign a GPU) first.");
            return;
        };
        let budget = self.free_budget(p);
        let mut s = self.current_options(p);
        let auto = gi(&s, "ngl", 999) < 0;
        let n_eff = info.n_layer.saturating_sub(info.nextn) as i64;
        if auto {
            s.insert("ngl".into(), (n_eff + 1).into()); // Fit = largest context that still keeps every layer on GPU
        }
        let mm = self.mmproj_bytes(p, &s);
        let oh = gf(self.settings.g(), "overhead_mib", 300.0);
        let (adj, _) = self.calib_adjust(p);
        let est_at = |s: &Opts, ctx: i64| {
            let mut s2 = s.clone();
            s2.insert("ctx".into(), ctx.into());
            est_simple(&info, &s2, mm, oh, adj).total
        };
        let fits = |ctx: i64| est_at(&s, ctx) <= budget;
        let cap = if info.n_ctx_train != 0 { info.n_ctx_train } else { 131072 };
        let cap = (cap / 1024 * 1024).max(1024);
        if !fits(1024) && auto {
            let cur = self.current_options(p);
            let k = self.auto_layers(&info, &cur, mm, oh, adj, budget);
            let layers = match k {
                Some(k) => format!("≈ {} of {} layers", k.min(n_eff), n_eff),
                None => "no layers".into(),
            };
            self.msg(
                "info",
                &format!(
                    "This model can't fit entirely on this GPU at any context size.\n\nWith GPU layers on auto (-1), context is your choice and llama.cpp\nputs as many layers on the GPU as fit around it:\nat the current {} context, {} would go on the GPU.\n\nMore context leaves room for fewer layers, which means slower generation.\nThe estimate line updates the layer count as you change context.",
                    fmt_thousands(gi(&cur, "ctx", 0)),
                    layers
                ),
            );
            return;
        }
        if !fits(1024) {
            let base = est_at(&s, 0);
            self.msg(
                "warning",
                &format!(
                    "Even 1024 context doesn't fit.\n\nFree VRAM (after margin): {:.2} GiB\nModel without context needs ≈ {:.2} GiB\n\nTry a smaller quant, fewer GPU layers, or CPU MoE layers.",
                    budget / GIB,
                    base / GIB
                ),
            );
            return;
        }
        let best_ctx = |s3: &Opts| -> i64 {
            if est_at(s3, 1024) > budget {
                return 0;
            }
            let (mut lo, mut hi) = (1i64, cap / 1024);
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if est_at(s3, mid * 1024) <= budget {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo * 1024
        };
        let lo = best_ctx(&s) / 1024;
        // Too little room for useful context: for MoE models, offer moving experts to RAM.
        if lo * 1024 < 16384 && info.n_expert != 0 && !auto {
            let mut options: Vec<(i64, i64)> = vec![];
            let ncm0 = gi(&s, "n_cpu_moe", 0);
            for ncm in (ncm0 + 1)..=(n_eff.min(ncm0 + 16)) {
                let mut s3 = s.clone();
                s3.insert("n_cpu_moe".into(), ncm.into());
                let c = best_ctx(&s3);
                if c > lo * 1024 && options.last().map(|o| c > o.1).unwrap_or(true) {
                    options.push((ncm, c));
                }
                if c >= cap.min(131072) {
                    break;
                }
            }
            let target = options.iter().find(|o| o.1 >= 32768).cloned().or(options.last().cloned());
            if let Some(target) = target {
                let table: Vec<String> = options
                    .iter()
                    .take(8)
                    .map(|(n, c)| format!("   CPU MoE layers {:>2}  →  context {}", n, fmt_thousands(*c)))
                    .collect();
                let text = format!(
                    "With everything on the GPU, only {} context fits ({:.2} GiB free after the margin).\n\nMoving some MoE expert layers to system RAM frees VRAM with only a\nsmall speed cost (only a few experts run per token):\n\n{}\n\nUse CPU MoE layers {} with context {}?",
                    fmt_thousands(lo * 1024),
                    budget / GIB,
                    table.join("\n"),
                    target.0,
                    fmt_thousands(target.1)
                );
                // The Python app asked synchronously; here the answer arrives later.
                self.ask(&text, "yesno", Pending::FitMoe { p, ncm: target.0, ctx: target.1, budget, fallback: lo * 1024, auto, cap });
                return;
            }
        }
        self.fit_apply_plain(p, lo * 1024, budget, auto, cap);
    }

    pub fn fit_apply_plain(&mut self, p: usize, ctx: i64, budget: f64, auto: bool, cap: i64) {
        self.set_var_notify(p, "ctx", Value::from(ctx.to_string()));
        let mut note = if ctx == cap { " (model's max)".to_string() } else { String::new() };
        if auto {
            note += " with all layers on GPU";
        }
        let side = cap_side(self.panes[p].side);
        self.set_top_status(format!(
            "{side} pane: Fit chose context {}{note} for {:.2} GiB free",
            fmt_thousands(ctx),
            budget / GIB
        ));
    }

    pub fn fit_apply_moe(&mut self, p: usize, ncm: i64, ctx: i64, budget: f64) {
        self.set_var_notify(p, "n_cpu_moe", Value::from(ncm.to_string()));
        self.set_var_notify(p, "ctx", Value::from(ctx.to_string()));
        let side = cap_side(self.panes[p].side);
        self.set_top_status(format!(
            "{side} pane: Fit chose CPU MoE layers {ncm}, context {} for {:.2} GiB free",
            fmt_thousands(ctx),
            budget / GIB
        ));
    }

    pub fn show_metadata(&mut self, p: usize) {
        if self.info_pending(p) {
            self.msg("info", "Still reading this model's metadata - try again in a moment.");
            return;
        }
        let Some(info) = self.panes[p].info.clone() else {
            self.msg("info", "Select a model first.");
            return;
        };
        let title = format!(
            "{}: {}",
            if self.panes[p].is_rad() {
                "Radiance model"
            } else if self.panes[p].is_vllm() {
                "Model config"
            } else {
                "GGUF metadata"
            },
            info.path().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        );
        self.report(&title, &info.metadata_text(), None);
    }
}

pub fn cap_side(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[allow(dead_code)]
fn _unused(_: &Est) {}
#[allow(dead_code)]
fn _unused2() -> Option<Est> {
    let _ = estimate_vram;
    None
}
