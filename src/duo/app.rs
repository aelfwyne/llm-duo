//! The application controller: settings, GPUs, both panes, dual mode, alerts, the
//! settings dialog and quantize tool back ends, and the JSON interface the QML UI talks to.

use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use super::consts::*;
use super::devices::{attach_sysfs, detect_devices, dev_label, Device};
use super::estimate::split_modes_supported;
use super::gguf::shard_re;
use super::kobold::{kobold_help_flags, map_vulkan, vulkan_gpus, VkGpu};
use super::mtp::is_helper_name;
use super::pane::{cap_side, Info, Pane};
use super::quantize::Quant;
use super::settings::Settings;
use super::util::*;
use super::vllm::{detect_devices_torch, scan_vllm_models, vllm_venv};

pub type Job = Box<dyn FnOnce(&mut App) + Send>;

#[derive(Clone, Debug, PartialEq)]
pub enum Timer {
    Save(usize),
    AutoLoad(usize),
    Calib(usize),
    Chime(usize),
}

pub enum Pending {
    FitMoe { p: usize, ncm: i64, ctx: i64, budget: f64, fallback: i64, auto: bool, cap: i64 },
    QuantReplace,
    QuantDeleteUnfinished(PathBuf),
    QuantCloseRunning,
    QuitModels,
    QuitQuant,
    ConvReplace,
    LoadAnyway(usize),
    ConvDeleteUnfinished(PathBuf),
    ConvCloseRunning,
}

pub struct App {
    pub settings: Settings,
    pub devices: Vec<Device>,
    pub server_help: String,
    pub split_modes: Vec<String>,
    info_cache: HashMap<(String, u64), Info>,
    pub models_by: HashMap<String, Vec<String>>,
    pub panes: [Pane; 2],
    pub job_tx: Sender<Job>,
    job_rx: Receiver<Job>,
    k_fetching: HashMap<String, Vec<usize>>,
    vk: Option<HashMap<String, i64>>,
    vk_how: HashMap<String, String>,
    pub vk_list: Vec<VkGpu>,
    pub vk_error: String,
    llama_help_cache: HashMap<String, ((String, f64, u64), String)>,
    pub top_status: String,
    why: Option<(String, String)>,
    why_status: String,
    alerts: Vec<(String, String, Option<PathBuf>)>,
    pub title: String,
    pub events: Vec<Value>,
    pending: HashMap<u64, Pending>,
    next_q: u64,
    timers: Vec<(Instant, Timer)>,
    pub quant: Quant,
    quant_busy: bool,
    pub conv: super::rad_convert::Conv,
    pub started: bool,
    detecting: bool,
    /// dashboard: GPU sensor samples and limits per device id
    pub gpu_hist: HashMap<String, std::collections::VecDeque<super::telemetry::GpuSample>>,
    pub gpu_limits: HashMap<String, super::telemetry::GpuLimits>,
}

impl App {
    pub fn new() -> Self {
        let settings = Settings::load();
        let lb = gs(settings.g(), "left_backend", LLAMA);
        let rb = gs(settings.g(), "right_backend", LLAMA);
        let fix = |b: String| if BACKENDS.contains(&b.as_str()) { b } else { LLAMA.to_string() };
        let (tx, rx) = channel();
        let mut models_by = HashMap::new();
        for b in BACKENDS {
            models_by.insert(b.to_string(), vec![]);
        }
        let mut app = App {
            settings,
            devices: vec![],
            server_help: String::new(),
            split_modes: vec!["none".into(), "layer".into()],
            info_cache: HashMap::new(),
            models_by,
            panes: [Pane::new("left", 0, fix(lb)), Pane::new("right", 1, fix(rb))],
            job_tx: tx,
            job_rx: rx,
            k_fetching: HashMap::new(),
            vk: None,
            vk_how: HashMap::new(),
            vk_list: vec![],
            vk_error: String::new(),
            llama_help_cache: HashMap::new(),
            top_status: String::new(),
            why: None,
            why_status: String::new(),
            alerts: vec![],
            title: APP_NAME.into(),
            events: vec![],
            pending: HashMap::new(),
            next_q: 1,
            timers: vec![],
            quant: Quant::new(),
            quant_busy: false,
            conv: super::rad_convert::Conv::new(),
            started: false,
            detecting: false,
            gpu_hist: HashMap::new(),
            gpu_limits: HashMap::new(),
        };
        for p in 0..2 {
            app.apply_backend_ui(p);
        }
        app
    }

    // ================================================================ events ==
    pub fn push_event(&mut self, v: Value) {
        self.events.push(v);
    }
    pub fn msg(&mut self, kind: &str, text: &str) {
        self.push_event(json!({"type": "msg", "kind": kind, "title": APP_NAME, "text": text}));
    }
    pub fn report(&mut self, title: &str, text: &str, log_path: Option<&Path>) {
        self.push_event(json!({"type": "report", "title": title, "text": text,
                               "logPath": log_path.map(|p| p.display().to_string()), "logDir": log_dir().display().to_string()}));
    }
    pub fn clipboard(&mut self, text: &str) {
        self.push_event(json!({"type": "clipboard", "text": text}));
    }
    pub fn ask(&mut self, text: &str, buttons: &str, action: Pending) {
        let id = self.next_q;
        self.next_q += 1;
        self.pending.insert(id, action);
        self.push_event(json!({"type": "question", "id": id, "title": APP_NAME, "text": text, "buttons": buttons}));
    }

    pub fn set_top_status(&mut self, s: impl Into<String>) {
        self.top_status = s.into();
        // hide the Why? button once the status message it explains is replaced
        if self.why.is_some() && self.top_status != self.why_status {
            self.why = None;
        }
    }

    pub fn set_why(&mut self, title: String, text: String) {
        self.why = Some((title, text));
        self.why_status = self.top_status.clone();
    }

    /// Record an error report without opening a window (no focus stealing).
    pub fn raise_alert(&mut self, title: &str, text: &str, path: Option<PathBuf>, short: &str) {
        self.alerts.push((title.into(), text.into(), path.clone()));
        self.title = format!("⚠ {APP_NAME} - {short}");
        self.set_top_status(format!("⚠ {short} - click 'View error'{}", if path.is_some() { " (log saved)" } else { "" }));
    }

    fn show_alerts(&mut self) {
        let alerts = std::mem::take(&mut self.alerts);
        self.title = APP_NAME.into();
        for (t, x, p) in alerts {
            self.report(&t, &x, p.as_deref());
        }
    }

    // ================================================================ timers ==
    pub fn schedule(&mut self, t: Timer, ms: u64) {
        self.timers.retain(|(_, x)| *x != t);
        self.timers.push((Instant::now() + Duration::from_millis(ms), t));
    }
    pub fn cancel_timer(&mut self, t: &Timer) {
        self.timers.retain(|(_, x)| x != t);
    }
    pub fn has_timer(&self, t: &Timer) -> bool {
        self.timers.iter().any(|(_, x)| x == t)
    }
    fn run_timers(&mut self) {
        let now = Instant::now();
        let due: Vec<Timer> = self.timers.iter().filter(|(t, _)| *t <= now).map(|(_, x)| x.clone()).collect();
        self.timers.retain(|(t, _)| *t > now);
        for t in due {
            match t {
                Timer::Save(p) => self.save_options(p),
                Timer::AutoLoad(p) => self.do_autoload(p),
                Timer::Calib(p) => self.record_calibration(p),
                Timer::Chime(p) => self.maybe_chime(p),
            }
        }
    }

    // ============================================================= startup ==
    pub fn startup(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        self.start_detect(true, Box::new(|app: &mut App| app.startup_continue()));
    }

    fn start_detect(&mut self, show_errors: bool, then: Job) {
        self.set_top_status("Detecting GPUs…");
        self.detecting = true;
        let g = self.settings.g().clone();
        let vk_bin = self.llama_bin("Vulkan");
        let tx = self.job_tx.clone();
        std::thread::spawn(move || {
            let (devs, err) = detect_devices_blocking(&g, &vk_bin);
            let _ = tx.send(Box::new(move |app: &mut App| {
                app.detecting = false;
                app.devices = devs;
                app.vk = None;
                app.load_server_help();
                let n = app.devices.len();
                app.set_top_status(format!("{n} GPU(s) detected{}", if n > 0 { "" } else { " — check Settings" }));
                if !err.is_empty() && show_errors {
                    app.report(&format!("{APP_NAME}: GPU detection"), &err, None);
                }
                then(app);
            }));
        });
    }

    fn startup_continue(&mut self) {
        let ids: Vec<String> = self.devices.iter().map(|d| d.id.clone()).collect();
        {
            let g = self.settings.g_mut();
            let left = gs(g, "left_device", "");
            if !ids.contains(&left) {
                g.insert("left_device".into(), ids.first().cloned().unwrap_or_default().into());
            }
            let left = gs(g, "left_device", "");
            let right = gs(g, "right_device", "");
            if !ids.contains(&right) || right == left {
                g.insert("right_device".into(), ids.iter().find(|i| **i != left).cloned().unwrap_or_default().into());
            }
            if gi(g, "fit_margin_mib", 1536) == 768 {
                g.insert("fit_margin_mib".into(), 1536.into()); // old default left cards ~95% full
            }
        }
        self.settings.save();
        self.apply_settings();
        self.prefetch_kobold();
        let g = self.settings.g().clone();
        let models_ok = Path::new(&gs(&g, "models_dir", "")).is_dir();
        let llama_ok = models_ok && KOBOLD_BUILDS.iter().any(|b| Path::new(&self.llama_bin(b)).is_file());
        let kobold_ok = models_ok
            && KOBOLD_BUILDS.iter().any(|b| {
                let e = self.kobold_bin(b);
                !e.is_empty() && Path::new(&e).is_file()
            });
        let vllm_ok = expanduser(&gs(&g, "vllm_bin", "")).is_file() && Path::new(&gs(&g, "vllm_models_dir", "")).is_dir();
        if !(llama_ok || kobold_ok || vllm_ok) {
            self.push_event(json!({"type": "openSettings"}));
            return;
        }
        self.restore_last_models();
    }

    fn restore_last_models(&mut self) {
        let mut loading = vec![];
        let auto = gb(self.settings.g(), "auto_load", false);
        for p in 0..2 {
            if p == 1 && self.dual_on() {
                continue;
            }
            let key = self.panes[p].last_key();
            let last = self.settings.g().get("last").and_then(|v| v.get(&key)).cloned();
            let Some(last) = last else { continue };
            let model = last.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if self.panes[p].device.is_none() || !self.panes[p].models.contains(&model) {
                continue;
            }
            self.panes[p].model_var = model;
            self.on_model_selected(p);
            if auto && truthy(last.get("running")) {
                self.request_autoload(p);
                loading.push(self.panes[p].side);
            }
        }
        if !loading.is_empty() {
            self.set_top_status(format!("Auto-loading last used model(s): {}", loading.join(", ")));
        }
    }

    fn load_server_help(&mut self) {
        let h = self.llama_help("ROCm");
        self.server_help = if h.is_empty() { self.llama_help("Vulkan") } else { h };
        self.split_modes = split_modes_supported(&self.server_help);
    }

    pub fn apply_settings(&mut self) {
        {
            let g = self.settings.g();
            super::radiance::set_tools(&gs(g, "radiance_bin", ""), &gs(g, "radiance_home", ""));
        }
        let by_id: HashMap<String, Device> = self.devices.iter().map(|d| (d.id.clone(), d.clone())).collect();
        self.rescan_models();
        for p in 0..2 {
            if self.panes[p].proc.is_none() {
                let side = self.panes[p].side;
                let id = gs(self.settings.g(), &format!("{side}_device"), "");
                self.set_device(p, by_id.get(&id).cloned());
            }
        }
        self.apply_dual();
    }

    // ============================================================ binaries ==
    pub fn llama_bin(&self, build: &str) -> String {
        let p = gs(self.settings.g(), if build == "ROCm" { "server_bin" } else { "server_bin_vulkan" }, "");
        if p.trim().is_empty() { String::new() } else { expanduser(&p).to_string_lossy().into_owned() }
    }

    /// llama-server --help text for one build (cached per binary file).
    pub fn llama_help(&mut self, build: &str) -> String {
        let exe = self.llama_bin(build);
        let Some((mt, sz)) = file_key(Path::new(&exe)) else { return String::new() };
        let key = (exe.clone(), mt, sz);
        if let Some((k, t)) = self.llama_help_cache.get(build) {
            if *k == key {
                return t.clone();
            }
        }
        let text = match run_capture(&[exe, "--help".into()], Duration::from_secs(30), None) {
            Ok(o) => o.stdout + &o.stderr,
            Err(_) => String::new(),
        };
        self.llama_help_cache.insert(build.to_string(), (key, text.clone()));
        text
    }

    pub fn kobold_bin(&self, build: &str) -> String {
        let p = gs(self.settings.g(), if build == "ROCm" { "kobold_rocm_bin" } else { "kobold_vulkan_bin" }, "");
        if p.trim().is_empty() { String::new() } else { expanduser(&p).to_string_lossy().into_owned() }
    }

    /// Flags this build's --help lists (cached per binary file), or None if not read yet.
    pub fn kobold_flags(&self, build: &str) -> Option<BTreeSet<String>> {
        let exe = self.kobold_bin(build);
        let c = self.settings.g().get("kobold_flags")?.get(build)?.as_object()?.clone();
        if exe.is_empty() {
            return None;
        }
        let (mt, sz) = file_key(Path::new(&exe))?;
        if gs(&c, "path", "") != exe || gf(&c, "mtime", -1.0) != mt || gi(&c, "size", -1) as u64 != sz {
            return None;
        }
        Some(
            c.get("flags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        )
    }

    /// Read '<binary> --help' in the background; pane `cb` gets k_after_flags when done.
    pub fn fetch_kobold_flags(&mut self, build: &str, cb: Option<usize>) {
        let exe = self.kobold_bin(build);
        if let Some(w) = self.k_fetching.get_mut(build) {
            if let Some(p) = cb {
                w.push(p);
            }
            return;
        }
        if exe.is_empty() || !Path::new(&exe).is_file() {
            if let Some(p) = cb {
                let e = format!("KoboldCpp {build} binary not found: {}", if exe.is_empty() { "(not set)" } else { &exe });
                self.k_after_flags(p, false, &e);
            }
            return;
        }
        self.k_fetching.insert(build.to_string(), cb.into_iter().collect());
        let tx = self.job_tx.clone();
        let b = build.to_string();
        std::thread::spawn(move || {
            let r = kobold_help_flags(&exe, 180);
            let _ = tx.send(Box::new(move |app: &mut App| app.kobold_flags_done(&b, &exe, r)));
        });
    }

    fn kobold_flags_done(&mut self, build: &str, exe: &str, r: Result<Vec<String>, String>) {
        if let Ok(flags) = &r {
            if let Some((mt, sz)) = file_key(Path::new(exe)) {
                let g = self.settings.g_mut();
                if !g.get("kobold_flags").map(|v| v.is_object()).unwrap_or(false) {
                    g.insert("kobold_flags".into(), json!({}));
                }
                g.get_mut("kobold_flags")
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert(build.into(), json!({"path": exe, "mtime": mt, "size": sz, "flags": flags}));
                self.settings.save();
            }
        }
        let err = r.as_ref().err().cloned().unwrap_or_default();
        for p in self.k_fetching.remove(build).unwrap_or_default() {
            self.k_after_flags(p, r.is_ok(), &err);
        }
        for p in 0..2 {
            if self.panes[p].is_kobold() {
                self.panes[p].k_last_build = None; // re-check MTP / split options for this build
                self.k_refresh_row(p);
                self.update_estimate(p);
            }
        }
        if r.is_err() {
            let first = err.lines().next().unwrap_or("could not read --help").to_string();
            self.set_top_status(format!("KoboldCpp {build}: {first}"));
        }
    }

    pub fn prefetch_kobold(&mut self) {
        for b in KOBOLD_BUILDS {
            let exe = self.kobold_bin(b);
            if !exe.is_empty() && Path::new(&exe).is_file() && access_x(&exe) && self.kobold_flags(b).is_none() {
                self.fetch_kobold_flags(b, None);
            }
        }
    }

    pub fn vulkan_index(&mut self, dev: &Device) -> Option<i64> {
        if self.vk.is_none() {
            let (vk, err) = vulkan_gpus();
            let (m, how) = if vk.is_empty() { (HashMap::new(), HashMap::new()) } else { map_vulkan(&self.devices, &vk) };
            self.vk_list = vk;
            self.vk = Some(m);
            self.vk_how = how;
            self.vk_error = err;
        }
        self.vk.as_ref().unwrap().get(&dev.id).cloned()
    }

    pub fn vulkan_map_text(&mut self) -> String {
        if self.devices.is_empty() {
            return "no GPUs detected".into();
        }
        let devs = self.devices.clone();
        let mut parts = vec![];
        for d in &devs {
            let i = self.vulkan_index(d);
            let nm = d.short_name();
            parts.push(match i {
                Some(i) => format!("{} {nm} → Vulkan {i} ({})", d.id, self.vk_how.get(&d.id).cloned().unwrap_or_default()),
                None => format!("{} {nm} → ?", d.id),
            });
        }
        parts.join("   •   ") + &if self.vk_error.is_empty() { String::new() } else { format!("\n⚠ {}", self.vk_error) }
    }

    // ================================================================ dual ==
    pub fn dual_devices(&self) -> Vec<Device> {
        match (&self.panes[0].device, &self.panes[1].device) {
            (Some(a), Some(b)) if a.id != b.id => vec![a.clone(), b.clone()],
            _ => vec![],
        }
    }

    pub fn dual_on(&self) -> bool {
        gb(self.settings.g(), "dual", false) && self.dual_devices().len() == 2
    }

    fn on_dual_toggle(&mut self, want: bool) {
        let problem = if want && self.panes[1].proc.is_some() {
            Some("Stop the right pane's model first - Dual GPU mode needs both cards.")
        } else if self.panes[0].proc.is_some() {
            Some("Stop the left pane's model before switching Dual GPU mode.")
        } else if want && self.dual_devices().len() != 2 {
            Some("Dual GPU mode needs two different GPUs assigned to the panes (see Settings).")
        } else {
            None
        };
        if let Some(pr) = problem {
            self.msg("info", pr);
            return;
        }
        self.settings.g_mut().insert("dual".into(), want.into());
        self.settings.save();
        self.apply_dual();
        self.set_top_status(if want { "Dual GPU mode on - left pane spans both GPUs" } else { "Dual GPU mode off" });
    }

    pub fn apply_dual(&mut self) {
        let on = self.dual_on();
        self.set_pane_disabled(1, on, if on { "In use by Dual GPU mode (left pane)" } else { "" });
        self.refresh_dual_ui(0);
        self.update_gpu_stats(0);
        if self.panes[0].proc.is_none() {
            self.on_model_selected(0); // reload options saved for this mode
        }
    }

    // ============================================================== models ==
    pub fn models_for(&self, backend: &str) -> Vec<String> {
        self.models_by.get(backend).cloned().unwrap_or_default()
    }

    pub fn rescan_models(&mut self) {
        let g = self.settings.g().clone();
        self.models_by.insert(VLLM.into(), scan_vllm_models(&gs(&g, "vllm_models_dir", "")));
        self.models_by.insert(RADIANCE.into(), super::radiance::scan_rad_models(&gs(&g, "radiance_models_dir", "")));
        let root = PathBuf::from(gs(&g, "models_dir", ""));
        let mut models = vec![];
        if root.is_dir() {
            let mut found = vec![];
            walk_gguf(&root, &mut found, 0);
            for p in found {
                let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if name.to_lowercase().contains("mmproj") || is_helper_name(&name) {
                    continue;
                }
                if let Some(m) = shard_re().captures(&name) {
                    if m[1].parse::<i64>().unwrap_or(1) != 1 {
                        continue;
                    }
                }
                if let Ok(r) = p.strip_prefix(&root) {
                    models.push(r.to_string_lossy().into_owned());
                }
            }
        }
        models.sort_by_key(|a| a.to_lowercase());
        self.models_by.insert(LLAMA.into(), models.clone());
        self.models_by.insert(KOBOLD.into(), models); // KoboldCpp runs the same GGUF files
        for p in 0..2 {
            let m = self.models_for(&self.panes[p].backend.clone());
            self.set_models(p, m);
        }
    }

    pub fn info_cache_get(&self, path: &Path) -> Option<Info> {
        let (mt, _) = file_key(path)?;
        self.info_cache.get(&(path.to_string_lossy().into_owned(), mt.to_bits())).cloned()
    }
    pub fn info_cache_put(&mut self, path: &Path, info: Info) {
        if let Some((mt, _)) = file_key(path) {
            self.info_cache.insert((path.to_string_lossy().into_owned(), mt.to_bits()), info);
        }
    }

    pub fn refresh_quant_btn_visible(&self) -> bool {
        self.quant_busy || self.panes.iter().any(|p| p.backend == VLLM)
    }
    pub fn quant_running(&mut self, running: bool) {
        self.quant_busy = running;
    }

    // ========================================================== main loop ==
    /// Called every 100 ms by the UI.
    pub fn poll(&mut self) {
        while let Ok(job) = self.job_rx.try_recv() {
            job(self);
        }
        for p in 0..2 {
            self.poll_pane(p);
        }
        self.quant_poll();
        self.conv_poll();
        self.run_timers();
    }

    /// Called every 1.5 s by the UI.
    pub fn tick(&mut self) {
        if !self.started || self.detecting {
            return;
        }
        for p in 0..2 {
            self.tick_pane(p);
        }
        self.sample_gpus();
    }

    // =============================================================== close ==
    fn request_close(&mut self) {
        if self.panes.iter().any(|p| p.proc.is_some()) {
            self.ask("Stop running models and quit?", "yesno", Pending::QuitModels);
            return;
        }
        self.close_stage2();
    }

    fn close_stage2(&mut self) {
        if self.quant.proc.is_some() || self.conv.proc.is_some() {
            let what = if self.quant.proc.is_some() { "An FP8 conversion" } else { "A Radiance conversion" };
            self.ask(&format!("{what} is still running. Stop it and quit?"), "yesno", Pending::QuitQuant);
            return;
        }
        self.finish_close();
    }

    fn finish_close(&mut self) {
        if self.quant.proc.is_some() {
            self.quant_stop();
        }
        if self.conv.proc.is_some() {
            self.conv_stop();
        }
        self.r_probe_kill_all();
        for p in 0..2 {
            if self.panes[p].proc.is_some() {
                self.stop(p, true, false); // keep "running" so auto-load restores it
            }
        }
        for p in 0..2 {
            if self.has_timer(&super::app::Timer::Save(p)) {
                self.save_options(p);
            }
        }
        self.push_event(json!({"type": "quit"}));
    }

    fn answer(&mut self, id: u64, ans: &str) {
        let Some(action) = self.pending.remove(&id) else { return };
        let yes = ans == "yes";
        match action {
            Pending::FitMoe { p, ncm, ctx, budget, fallback, auto, cap } => {
                if yes {
                    self.fit_apply_moe(p, ncm, ctx, budget);
                } else {
                    self.fit_apply_plain(p, fallback, budget, auto, cap);
                }
            }
            Pending::LoadAnyway(p) => {
                if yes {
                    self.panes[p].load_confirmed = true;
                    self.load(p);
                }
            }
            Pending::ConvReplace => {
                if yes {
                    self.conv_start(true);
                }
            }
            Pending::ConvDeleteUnfinished(f) => {
                if yes {
                    let _ = std::fs::remove_file(f);
                }
            }
            Pending::ConvCloseRunning => match ans {
                "yes" => {
                    self.conv_stop();
                    self.conv.visible = false;
                }
                "no" => self.conv.visible = false,
                _ => {}
            },
            Pending::QuantReplace => {
                if yes {
                    self.quant_start_confirmed(true);
                }
            }
            Pending::QuantDeleteUnfinished(dir) => {
                if yes {
                    let _ = std::fs::remove_dir_all(dir);
                }
            }
            Pending::QuantCloseRunning => match ans {
                "yes" => {
                    self.quant_stop();
                    self.quant.visible = false;
                }
                "no" => self.quant.visible = false,
                _ => {}
            },
            Pending::QuitModels => {
                if yes {
                    self.close_stage2();
                }
            }
            Pending::QuitQuant => {
                if yes {
                    self.finish_close();
                }
            }
        }
    }

    // ============================================================ settings ==
    fn settings_data(&mut self) -> Value {
        let g = self.settings.g().clone();
        let s = |k: &str| py_str(g.get(k).unwrap_or(&Value::from("")));
        let mut fields = Map::new();
        for k in [
            "models_dir", "server_bin", "server_bin_vulkan", "host", "left_port", "right_port", "overhead_mib",
            "fit_margin_mib", "chime_min_secs", "vllm_bin", "vllm_models_dir", "vllm_env", "vllm_left_port",
            "vllm_right_port", "kobold_rocm_bin", "kobold_vulkan_bin", "kobold_left_port", "kobold_right_port",
            "radiance_bin", "radiance_home", "radiance_models_dir", "radiance_recipes_dir", "radiance_env",
            "radiance_left_port", "radiance_right_port", "llama_api_key", "kobold_password", "vllm_api_key", "radiance_api_key",
        ] {
            fields.insert(k.into(), Value::from(s(k)));
        }
        let labels: Vec<String> = self.devices.iter().map(|d| dev_label(d, 1)).collect();
        let pick = |id: String| self.devices.iter().find(|d| d.id == id).map(|d| dev_label(d, 1)).unwrap_or_default();
        json!({
            "fields": fields,
            "autoLoad": gb(&g, "auto_load", false),
            "chime": gb(&g, "chime", false),
            "backends": [backend_label(&self.panes[0].backend), backend_label(&self.panes[1].backend)],
            "backendLocked": [self.panes[0].proc.is_some(), self.panes[1].proc.is_some()],
            "backendLabels": BACKENDS.iter().map(|b| backend_label(b)).collect::<Vec<_>>(),
            "experimentalLabel": backend_label(VLLM),
            "devices": labels,
            "leftDevice": pick(gs(&g, "left_device", "")),
            "rightDevice": pick(gs(&g, "right_device", "")),
            "showVk": !gs(&g, "kobold_vulkan_bin", "").trim().is_empty(),
            "help": help(),
        })
    }

    fn settings_save(&mut self, a: &Value) -> Value {
        let f = a.get("fields").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        let fs = |k: &str| f.get(k).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let ints = [
            "left_port", "right_port", "overhead_mib", "fit_margin_mib", "chime_min_secs", "vllm_left_port",
            "vllm_right_port", "kobold_left_port", "kobold_right_port", "radiance_left_port", "radiance_right_port",
        ];
        let mut nums = HashMap::new();
        for k in ints {
            match fs(k).parse::<i64>() {
                Ok(v) => {
                    nums.insert(k, v);
                }
                Err(_) => return json!({"error": "Ports, overhead and margin must be whole numbers."}),
            }
        }
        let dev_id = |label: &str| label.split(':').next().unwrap_or("").to_string();
        let new_left = dev_id(a.get("leftDevice").and_then(|v| v.as_str()).unwrap_or(""));
        let new_right = dev_id(a.get("rightDevice").and_then(|v| v.as_str()).unwrap_or(""));
        if !new_left.is_empty() && new_left == new_right {
            return json!({"error": "Left and right panes must use different GPUs."});
        }
        let g = self.settings.g().clone();
        let running = self.panes.iter().any(|p| p.proc.is_some());
        if running && (new_left != gs(&g, "left_device", "") || new_right != gs(&g, "right_device", "")) {
            return json!({"error": "Stop running models before changing GPU assignment."});
        }
        let bin_changed = fs("server_bin") != gs(&g, "server_bin", "") || fs("vllm_bin") != gs(&g, "vllm_bin", "");
        {
            let gm = self.settings.g_mut();
            for k in [
                "models_dir", "server_bin", "server_bin_vulkan", "vllm_bin", "vllm_models_dir", "vllm_env", "kobold_rocm_bin",
                "kobold_vulkan_bin", "radiance_bin", "radiance_home", "radiance_models_dir", "radiance_recipes_dir", "radiance_env",
                "llama_api_key", "kobold_password", "vllm_api_key", "radiance_api_key",
            ] {
                gm.insert(k.into(), Value::from(fs(k)));
            }
            let host = fs("host");
            gm.insert("host".into(), Value::from(if host.is_empty() { "127.0.0.1".to_string() } else { host }));
            for (k, v) in &nums {
                let v = if *k == "chime_min_secs" { (*v).max(0) } else { *v };
                gm.insert(k.to_string(), v.into());
            }
            gm.insert("auto_load".into(), truthy(a.get("autoLoad")).into());
            gm.insert("chime".into(), truthy(a.get("chime")).into());
            gm.insert("left_device".into(), new_left.into());
            gm.insert("right_device".into(), new_right.into());
        }
        self.settings.save();
        let backends: Vec<String> = a
            .get("backends")
            .and_then(|v| v.as_array())
            .map(|x| x.iter().map(|y| y.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default();
        let finish = move |app: &mut App| {
            app.apply_settings();
            for p in 0..2 {
                if let Some(lab) = backends.get(p) {
                    let b = BACKENDS.iter().find(|b| backend_label(b) == lab).map(|b| b.to_string()).unwrap_or(LLAMA.into());
                    app.set_backend(p, &b);
                }
                if !app.panes[p].is_vllm() {
                    app.panes[p].k_last_build = None;
                    app.k_refresh_row(p);
                }
            }
            app.prefetch_kobold();
        };
        if bin_changed && self.devices.is_empty() {
            self.start_detect(true, Box::new(finish));
        } else {
            finish(self);
        }
        json!({"ok": true})
    }

    fn settings_redetect(&mut self, server_bin: &str) {
        self.settings.g_mut().insert("server_bin".into(), Value::from(server_bin.trim()));
        self.start_detect(
            true,
            Box::new(|app: &mut App| {
                let labels: Vec<String> = app.devices.iter().map(|d| dev_label(d, 1)).collect();
                let g = app.settings.g().clone();
                let pick = |id: String| app.devices.iter().find(|d| d.id == id).map(|d| dev_label(d, 1)).unwrap_or_default();
                let ev = json!({"type": "settingsDevices", "devices": labels,
                                "leftDevice": pick(gs(&g, "left_device", "")),
                                "rightDevice": pick(gs(&g, "right_device", ""))});
                app.push_event(ev);
            }),
        );
    }

    // ================================================================= UI ==
    fn pane_ui(&mut self, p: usize) -> Value {
        let dual_active = self.dual_active(p);
        let modes = self.panes[p].dual_mode_values.clone();
        let first_vals = self.dual_first_values();
        let quant_btn = self.refresh_quant_btn_visible();
        let _ = quant_btn;
        let pane = &self.panes[p];
        let running = pane.proc.is_some();
        let vars: Map<String, Value> = pane.vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let state_color = match pane.state.as_str() {
            "Stopped" => "#888888",
            "Loading" | "Stopping" | "Checking plan" => "#d08a00",
            "Ready" => "#2e9e44",
            "Processing" => "#2a7fd4",
            _ => "#d0342c",
        };
        let manual = pane.var_s("dual_strategy") == "Manual";
        let status: Map<String, Value> = pane.status.iter().map(|(k, v)| (k.to_string(), Value::from(v.clone()))).collect();
        json!({
            "side": pane.side,
            "backend": pane.backend,
            "isVllm": pane.is_vllm(), "isKobold": pane.is_kobold(), "isLlama": pane.is_llama(), "isRad": pane.is_rad(),
            "probing": pane.r_probe.is_some(),
            "badge": pane.badge.0, "badgeColor": pane.badge.1,
            "gpuName": pane.gpu_name, "gpuStat": pane.gpu_stat, "gpuStat2": pane.gpu_stat2,
            "vramPct": pane.vram_pct, "vramPct2": pane.vram_pct2, "showMon2": dual_active,
            "dualNote": pane.dual_note,
            "models": pane.models, "model": pane.model_var, "modelInfo": pane.model_info,
            "running": running,
            "vars": vars,
            "mmprojValues": pane.mmproj_values, "mtpValues": pane.mtp_values, "mtpEnabled": pane.mtp_enabled,
            "vMtpValues": pane.v_mtp_values, "vMtpEnabled": pane.v_mtp_enabled,
            "vTextEnabled": pane.v_text_enabled, "vQuantEnabled": pane.v_quant_enabled,
            "nglMax": pane.ngl_max,
            "mtpNote": pane.mtp_note, "vMtpNote": pane.v_mtp_note,
            "kNote": pane.k_note.0, "kNoteColor": pane.k_note.1,
            "estText": pane.est_text, "estColor": pane.est_color, "estPct": pane.est_pct,
            "showDual": pane.show_dual && dual_active,
            "dualStrategies": if pane.is_rad() {
                super::radiance::RAD_STRATEGIES.to_vec()
            } else if pane.is_vllm() {
                VLLM_STRATEGIES.to_vec()
            } else {
                DUAL_STRATEGIES.to_vec()
            },
            "dualModes": modes, "dualFirstValues": first_vals,
            "dualManualEnabled": manual && !running, "dualInfo": pane.dual_info,
            "webEnabled": pane.web_enabled,
            "webLabel": if pane.is_vllm() {
                "API docs"
            } else if pane.is_kobold() {
                "Open KoboldAI Lite"
            } else if pane.is_rad() {
                "Open dashboard"
            } else {
                "Open web UI"
            },
            "state": pane.state, "stateColor": state_color,
            "status": status,
            "disabled": pane.pane_disabled,
        })
    }

    pub fn ui(&mut self) -> Value {
        let left = self.pane_ui(0);
        let right = self.pane_ui(1);
        json!({
            "app": {
                "title": self.title,
                "build": env!("LLM_DUO_BUILD"),
                "topStatus": self.top_status,
                "why": self.why.is_some(),
                "alerts": self.alerts.len(),
                "dualOn": self.dual_on(),
                "dualEnabled": self.panes[0].proc.is_none(),
                "quantVisible": self.refresh_quant_btn_visible(),
                "quantLabel": if self.quant_busy { "Quantize… (running)" } else { "Quantize…" },
                "convVisible": self.conv.proc.is_some() || self.panes.iter().any(|p| p.is_rad()),
                "convLabel": if self.conv.proc.is_some() { "Convert… (running)" } else { "Convert…" },
            },
            "panes": [left, right],
            "quant": self.quant_ui(),
            "convert": self.conv_ui(),
            "lists": {
                "kvTypes": KV_TYPES, "fa": ["on", "auto", "off"], "kbuilds": KOBOLD_BUILDS,
                "vKvTypes": VLLM_KV_TYPES, "vDtypes": VLLM_DTYPES, "vQuant": VLLM_ONLINE_QUANT_KEYS,
                "vReasoning": VLLM_REASONING,
                "rKv": super::radiance::RAD_KV_TYPES, "rSpec": super::radiance::RAD_SPEC,
                "rDraftWindow": super::radiance::RAD_DRAFT_WINDOW, "rVision": super::radiance::RAD_VISION,
                "rEmbed": super::radiance::RAD_EMBED, "rPlacement": super::radiance::RAD_PLACEMENT,
                "rReasoning": super::radiance::RAD_REASONING, "rTpWire": super::radiance::RAD_TP_WIRE,
            },
        })
    }

    // =========================================================== dispatch ==
    /// Single entry point for the UI: `target` is "app", "left", "right" or "quant".
    pub fn call(&mut self, target: &str, action: &str, arg: &Value) -> Value {
        let s = |k: &str| arg.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let p = match target {
            "left" => Some(0),
            "right" => Some(1),
            _ => None,
        };
        if let Some(p) = p {
            if self.panes[p].pane_disabled && !matches!(action, "copyLog" | "saveLog") {
                return Value::Null;
            }
            match action {
                "setVar" => {
                    let key = s("key");
                    let val = arg.get("value").cloned().unwrap_or(Value::Null);
                    self.set_var(p, &key, &val);
                }
                "selectModel" => self.select_model(p, &s("model")),
                "fit" => self.fit_context(p),
                "load" => self.load(p),
                "stop" => self.stop(p, false, true),
                "openWeb" => self.open_web(p),
                "copyCmd" => self.copy_cmd(p),
                "copyLog" => self.copy_log(p),
                "saveLog" => self.save_log(p),
                "showMetadata" => self.show_metadata(p),
                _ => {}
            }
            return Value::Null;
        }
        if target == "dash" {
            return match action {
                "live" => self.dash_live(arg),
                "history" => self.dash_history(arg),
                "delete" => self.dash_delete(arg),
                _ => Value::Null,
            };
        }
        if target == "convert" {
            let v = arg.get("value").map(py_str).unwrap_or_default();
            self.conv_call(action, &v);
            return Value::Null;
        }
        if target == "quant" {
            match action {
                "show" => self.quant_show(),
                "setSrc" => self.quant_set_src(&s("value")),
                "setOut" => self.quant_set_out(&s("value")),
                "setIgnore" => self.quant.ignore = s("value"),
                "start" => self.quant_start(),
                "stop" => self.quant_stop(),
                "hide" => self.quant.visible = false,
                "close" => self.quant_close(),
                _ => {}
            }
            return Value::Null;
        }
        match action {
            "startup" => self.startup(),
            "rescan" => self.rescan_models(),
            "setDual" => self.on_dual_toggle(truthy(arg.get("on"))),
            "showAlerts" => self.show_alerts(),
            "showWhy" => {
                if let Some((t, x)) = self.why.clone() {
                    self.report(&t, &x, None);
                }
            }
            "answer" => {
                let id = arg.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                self.answer(id, &s("answer"));
            }
            "requestClose" => self.request_close(),
            "settingsData" => return self.settings_data(),
            "settingsSave" => return self.settings_save(arg),
            "settingsRedetect" => self.settings_redetect(&s("serverBin")),
            "settingsVk" => {
                self.vk = None;
                return Value::from(self.vulkan_map_text());
            }
            "testChime" => super::proc::play_chime(),
            "openLogs" => {
                let d = log_dir();
                let _ = std::fs::create_dir_all(&d);
                open_url(&d.display().to_string());
            }
            "help" => return Value::from(help_text(&s("key"))),
            _ => {}
        }
        Value::Null
    }

    pub fn take_events(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.events)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

fn walk_gguf(d: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 16 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(d) else { return };
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        let ft = match e.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_gguf(&p, out, depth + 1);
        } else if p.extension().map(|x| x == "gguf").unwrap_or(false) {
            out.push(p);
        }
    }
}

/// GPU detection as in the Python app: llama-server --list-devices, else the vLLM venv's PyTorch.
fn detect_devices_blocking(g: &Opts, vk_bin: &str) -> (Vec<Device>, String) {
    let mut devs = vec![];
    let mut err = String::new();
    let mut det_bin = gs(g, "server_bin", "");
    if !Path::new(&det_bin).is_file() && !vk_bin.is_empty() && Path::new(vk_bin).is_file() {
        det_bin = vk_bin.to_string(); // no ROCm build: detect with the Vulkan one
    }
    let has_venv = vllm_venv(&gs(g, "vllm_bin", "")).is_some();
    if Path::new(&det_bin).is_file() || !has_venv {
        let r = detect_devices(&det_bin);
        devs = r.0;
        err = r.1;
    }
    if devs.is_empty() && has_venv {
        let (d, err2) = detect_devices_torch(g);
        devs = attach_sysfs(d);
        err = if devs.is_empty() { format!("{err}\n\n{err2}").trim().to_string() } else { String::new() };
    }
    (devs, err)
}

#[allow(dead_code)]
fn _cap(s: &str) -> String {
    cap_side(s)
}
