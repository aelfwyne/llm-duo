//! FP8 quantize tool: makes an FP8-dynamic (W8A8) copy of a vLLM model in the background by
//! running the bundled Python script with the vLLM venv's interpreter.

use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use super::app::{App, Pending};
use super::proc::{self, Proc, ProcMsg};
use super::util::*;
use super::vllm::{scan_vllm_models, vllm_process_env, vllm_venv, VllmModelInfo};

pub const FP8_TOOL_SRC: &str = include_str!("../../assets/fp8_dynamic_quant.py");

pub struct Quant {
    pub visible: bool,
    pub sources: Vec<String>,
    infos: HashMap<String, Arc<VllmModelInfo>>,
    pub src: String,
    pub out: String,
    pub ignore: String,
    pub info: String,
    pub status: String,
    pub progress: f64,
    pub proc: Option<Proc>,
    user_stopped: bool,
    t0: Option<Instant>,
    out_dir: Option<PathBuf>,
    pub lines: VecDeque<String>,
    tx: Sender<ProcMsg>,
    rx: Receiver<ProcMsg>,
}

impl Quant {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Quant {
            visible: false,
            sources: vec![],
            infos: HashMap::new(),
            src: String::new(),
            out: String::new(),
            ignore: String::new(),
            info: String::new(),
            status: "Idle".into(),
            progress: 0.0,
            proc: None,
            user_stopped: false,
            t0: None,
            out_dir: None,
            lines: VecDeque::new(),
            tx,
            rx,
        }
    }
}

impl Default for Quant {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    fn quant_root(&self) -> PathBuf {
        PathBuf::from(gs(self.settings.g(), "vllm_models_dir", ""))
    }

    pub fn quant_ui(&self) -> Value {
        let q = &self.quant;
        json!({
            "visible": q.visible, "sources": q.sources, "src": q.src, "out": q.out, "ignore": q.ignore,
            "info": q.info, "status": q.status, "progress": q.progress, "running": q.proc.is_some(),
        })
    }

    pub fn quant_show(&mut self) {
        self.quant.visible = true;
        if self.quant.proc.is_none() {
            self.quant_refresh_sources();
        }
    }

    pub fn quant_refresh_sources(&mut self) {
        let root = self.quant_root();
        let mut vals = vec![];
        let mut infos = HashMap::new();
        for rel in scan_vllm_models(&root.to_string_lossy()) {
            let path = root.join(&rel);
            let info = match self.info_cache_get(&path).and_then(|i| i.v().cloned()) {
                Some(i) => i,
                None => match VllmModelInfo::load(&path) {
                    Ok(i) => Arc::new(i),
                    Err(_) => continue,
                },
            };
            if !info.quant.is_empty() {
                continue;
            }
            infos.insert(rel.clone(), info);
            vals.push(rel);
        }
        self.quant.infos = infos;
        self.quant.sources = vals.clone();
        if !vals.contains(&self.quant.src) {
            let first = vals.first().cloned().unwrap_or_default();
            self.quant_set_src(&first);
        }
        if vals.is_empty() {
            self.quant.info = format!("No 16-bit model folders in {}", root.display());
        }
    }

    pub fn quant_set_src(&mut self, rel: &str) {
        self.quant.src = rel.to_string();
        if !rel.is_empty() && self.quant.proc.is_none() {
            let p = std::path::Path::new(rel);
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            self.quant.out = p.with_file_name(format!("{name}-FP8-Dynamic")).to_string_lossy().into_owned();
        }
        self.quant_update_info();
    }

    pub fn quant_set_out(&mut self, v: &str) {
        self.quant.out = v.to_string();
        self.quant_update_info();
    }

    fn quant_update_info(&mut self) {
        let Some(info) = self.quant.infos.get(&self.quant.src).cloned() else { return };
        let lin: f64 = info.layer_bytes.iter().sum::<f64>() + info.mtp_bytes;
        let est = info.total_bytes - lin / 2.0;
        let root = self.quant_root();
        let free = disk_free(&root);
        let out = self.quant.out.trim();
        let out = if out.is_empty() { None } else { Some(root.join(out)) };
        let mins = (info.total_bytes + est) / (300.0 * MIB) / 60.0; // ~300 MB/s, read + write
        let mut txt = format!("{:.1} GiB  →  ≈ {:.1} GiB", info.total_bytes / GIB, est / GIB)
            + &free.map(|f| format!("   •   free on that drive: {:.0} GiB", f / GIB)).unwrap_or_default()
            + &format!(
                "   •   roughly {}-{} min (mostly disk speed)",
                py_round(mins * 0.7).max(1),
                py_round(mins * 1.5).max(2)
            );
        if let Some(f) = free {
            if f < est + GIB {
                txt += "\n⚠ not enough free space for the output";
            }
        }
        if let Some(o) = &out {
            if o.exists() && std::fs::read_dir(o).map(|mut r| r.next().is_some()).unwrap_or(false) {
                txt += "\n⚠ the output folder already exists - it will be replaced";
            }
        }
        if info.multimodal {
            txt += "\nvision model: the vision tower stays 16-bit";
        }
        self.quant.info = txt;
    }

    fn quant_python(&self) -> String {
        match vllm_venv(&gs(self.settings.g(), "vllm_bin", "")) {
            Some(v) => v.join("bin/python").to_string_lossy().into_owned(),
            None => "python3".into(),
        }
    }

    pub fn quant_start(&mut self) {
        if self.quant.proc.is_some() {
            return;
        }
        let (rel, name) = (self.quant.src.clone(), self.quant.out.trim().to_string());
        if rel.is_empty() || name.is_empty() {
            self.msg("info", "Pick a source model and an output folder name.");
            return;
        }
        let root = self.quant_root();
        let (src, dst) = (root.join(&rel), root.join(&name));
        let canon = |p: &PathBuf| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        if canon(&dst) == canon(&src) {
            self.msg("error", "The output folder must differ from the source.");
            return;
        }
        if dst.exists() && std::fs::read_dir(&dst).map(|mut r| r.next().is_some()).unwrap_or(false) {
            self.ask(&format!("{}\nalready exists. Replace it?", dst.display()), "yesno", Pending::QuantReplace);
            return;
        }
        self.quant_start_confirmed(false);
    }

    pub fn quant_start_confirmed(&mut self, replace: bool) {
        let root = self.quant_root();
        let (src, dst) = (root.join(&self.quant.src), root.join(self.quant.out.trim()));
        if replace {
            let _ = std::fs::remove_dir_all(&dst);
        }
        let tool = dirs().data.join("tools").join("fp8_dynamic_quant.py");
        if let Some(d) = tool.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Err(e) = std::fs::write(&tool, FP8_TOOL_SRC) {
            self.msg("error", &format!("Could not write the quantize tool:\n{e}"));
            return;
        }
        let mut cmd = vec![
            self.quant_python(),
            "-u".into(),
            tool.to_string_lossy().into_owned(),
            src.to_string_lossy().into_owned(),
            dst.to_string_lossy().into_owned(),
        ];
        match shlex_split(&self.quant.ignore) {
            Ok(pats) => {
                for p in pats {
                    cmd.push("--ignore".into());
                    cmd.push(p);
                }
            }
            Err(e) => {
                self.msg("error", &format!("Can't read 'Also keep 16-bit': {e}"));
                return;
            }
        }
        let g = self.settings.g().clone();
        let mut env: Vec<(String, String)> =
            if vllm_venv(&gs(&g, "vllm_bin", "")).is_some() { vllm_process_env(&g, &[]) } else { std::env::vars().collect() };
        env.retain(|e| e.0 != "HIP_VISIBLE_DEVICES");
        env.push(("HIP_VISIBLE_DEVICES".into(), String::new())); // CPU only
        self.push_event(json!({"type": "logClear", "side": "quant"}));
        self.quant.lines.clear();
        self.quant_log(&format!("$ {}", shlex_join(&cmd)));
        while self.quant.rx.try_recv().is_ok() {}
        match proc::spawn(&cmd, &env, false, self.quant.tx.clone()) {
            Ok(pr) => self.quant.proc = Some(pr),
            Err(e) => {
                self.quant_log(&format!("Failed to start: {e}"));
                self.quant.proc = None;
                return;
            }
        }
        self.quant.out_dir = Some(dst);
        self.quant.t0 = Some(Instant::now());
        self.quant.progress = 0.0;
        self.quant.status = "Starting…".into();
        self.quant_running(true);
    }

    fn quant_log(&mut self, line: &str) {
        if self.quant.lines.len() >= 400 {
            self.quant.lines.pop_front();
        }
        self.quant.lines.push_back(line.to_string());
        self.push_event(json!({"type": "log", "side": "quant", "text": line}));
    }

    pub fn quant_stop(&mut self) {
        let Some(pr) = &self.quant.proc else { return };
        self.quant.user_stopped = true;
        proc::term_group(pr.pid);
        self.quant.status = "Stopping…".into();
    }

    pub fn quant_close(&mut self) {
        if self.quant.proc.is_some() {
            self.ask(
                "A conversion is running.\n\nYes: stop it\nNo: keep it running in the background (reopen with Quantize… in the top bar)",
                "yesnocancel",
                Pending::QuantCloseRunning,
            );
            return;
        }
        self.quant.visible = false;
    }

    pub fn quant_poll(&mut self) {
        for _ in 0..500 {
            let Ok(msg) = self.quant.rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(val) => {
                    if let Some(rest) = val.strip_prefix("PROGRESS ") {
                        let nums: Vec<f64> = rest.split_whitespace().take(2).filter_map(|x| x.parse().ok()).collect();
                        if nums.len() < 2 {
                            continue;
                        }
                        let (done, total) = (nums[0], nums[1]);
                        let frac = done / total.max(1.0);
                        self.quant.progress = 1000.0 * frac;
                        let el = self.quant.t0.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                        let eta = if frac > 0.01 { Some(el / frac - el) } else { None };
                        self.quant.status = format!(
                            "{:.1} of {:.1} GiB  ({:.0}%)   •   {:.1} min",
                            done / GIB,
                            total / GIB,
                            100.0 * frac,
                            el / 60.0
                        ) + &eta.map(|e| format!(", about {:.0} min left", e / 60.0)).unwrap_or_default();
                    } else {
                        self.quant_log(&val);
                    }
                }
                ProcMsg::Exit(id, rc) => self.quant_on_exit(id, rc),
            }
        }
    }

    fn quant_on_exit(&mut self, id: u64, rc: i32) {
        if self.quant.proc.as_ref().map(|p| p.id) != Some(id) {
            return;
        }
        self.quant.proc = None;
        self.quant_running(false);
        let stopped = std::mem::take(&mut self.quant.user_stopped);
        let out_name = self.quant.out_dir.as_ref().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if rc == 0 {
            self.quant.progress = 1000.0;
            let mins = self.quant.t0.map(|t| t.elapsed().as_secs_f64() / 60.0).unwrap_or(0.0);
            self.quant.status = format!("Done in {mins:.1} min - '{out_name}' is now in the vLLM model lists");
            self.rescan_models();
            self.set_top_status(format!("Quantize: {out_name} is ready"));
            self.quant_refresh_sources();
            return;
        }
        self.quant.status = if stopped { "Stopped".into() } else { format!("Failed (exit code {rc}) - see the log") };
        self.set_top_status(format!("Quantize: {}", if stopped { "stopped" } else { "failed" }));
        if let Some(d) = self.quant.out_dir.clone() {
            if d.exists() {
                self.ask(&format!("Delete the unfinished output folder?\n{}", d.display()), "yesno", Pending::QuantDeleteUnfinished(d));
            }
        }
    }
}

pub fn disk_free(p: &std::path::Path) -> Option<f64> {
    let c = std::ffi::CString::new(p.to_string_lossy().as_bytes()).ok()?;
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut s) != 0 {
            return None;
        }
        Some(s.f_bavail as f64 * s.f_frsize as f64)
    }
}
