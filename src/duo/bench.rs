//! BetterBench (https://github.com/GGZ14/BetterBench): run its `betterbench run` benchmark
//! against a pane's running server in the background, with the last used options remembered.

use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use super::app::{App, Pending};
use super::proc::{self, Proc, ProcMsg};
use super::util::*;

pub const BB_URL: &str = "https://github.com/GGZ14/BetterBench";
/// The prompt categories of BetterBench's v1 corpus.
pub const BB_CATEGORIES: [&str; 8] = ["chat", "code", "file_edit", "json", "math", "prose", "reasoning", "summarization"];

pub struct Bench {
    pub visible: bool,
    /// the pane the window targets (and the run, while one is going)
    pub p: usize,
    pub proc: Option<Proc>,
    pub status: String,
    pub report: Option<String>,
    pub model: String,
    pub endpoint: String,
    /// why BetterBench can't be started (not installed …), shown in the window
    pub missing: String,
    /// waiting for the /v1/models lookup before spawning
    pub starting: bool,
    user_stopped: bool,
    t0: Option<Instant>,
    last_line: String,
    pub lines: VecDeque<String>,
    tx: Sender<ProcMsg>,
    rx: Receiver<ProcMsg>,
}

impl Bench {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Bench {
            visible: false,
            p: 0,
            proc: None,
            status: "Idle".into(),
            report: None,
            model: String::new(),
            endpoint: String::new(),
            missing: String::new(),
            starting: false,
            user_stopped: false,
            t0: None,
            last_line: String::new(),
            lines: VecDeque::new(),
            tx,
            rx,
        }
    }
    pub fn busy(&self) -> bool {
        self.proc.is_some() || self.starting
    }
}

impl Default for Bench {
    fn default() -> Self {
        Self::new()
    }
}

/// The window's options as BetterBench itself defaults them.
pub fn bench_defaults() -> Map<String, Value> {
    let v = json!({
        "command": "betterbench",
        "quick": false, "passes": 20, "warmup": 3,
        "decode": true, "prefill": true, "concurrency": true,
        "categories": BB_CATEGORIES,
        "greedy": false, "seed": "", "maxModelLen": "", "config": "",
        "name": "", "notes": "", "duoNotes": true,
        "html": true, "openReport": true, "updateCheck": true, "extra": "",
    });
    v.as_object().unwrap().clone()
}

/// Find the executable `word` names: a path, a name on PATH, or (for apps started from the
/// desktop, whose PATH often lacks it) ~/.local/bin, where pip and pipx put commands.
pub fn resolve_exe(word: &str) -> Option<PathBuf> {
    if word.contains('/') {
        let p = expanduser(word);
        return is_executable(&p).then_some(p);
    }
    which(word).or_else(|| {
        let p = home().join(".local/bin").join(word);
        is_executable(&p).then_some(p)
    })
}

fn not_installed(word: &str) -> String {
    format!(
        "BetterBench was not found ('{word}' is not an executable on PATH or in ~/.local/bin).\n\n\
         Install it from {BB_URL}, for example:\n    pipx install git+{BB_URL}\n\
         then set 'BetterBench command' here if it lives somewhere else."
    )
}

/// `http://host:port/v1` from a pane's endpoint status, with a wildcard bind host made local.
fn local_endpoint(url: &str) -> Option<(String, i64)> {
    let rest = url.strip_prefix("http://")?;
    let hp = rest.split('/').next()?;
    let (host, port) = hp.rsplit_once(':')?;
    let port: i64 = port.parse().ok()?;
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let host = if host.is_empty() || host == "0.0.0.0" || host == "::" { "127.0.0.1" } else { host };
    Some((host.to_string(), port))
}

/// The command line for `opts` (everything but the endpoint's key, which goes in the environment).
pub fn build_bench_cmd(exe: &str, rest: &[String], o: &Map<String, Value>, endpoint: &str, model: &str, duo_notes: &[(String, String)]) -> Result<Vec<String>, String> {
    let b = |k: &str| gb(o, k, false);
    let s = |k: &str| gs(o, k, "").trim().to_string();
    let mut cmd: Vec<String> = vec![exe.to_string()];
    cmd.extend(rest.iter().cloned());
    cmd.extend(["run".into(), "--endpoint".into(), endpoint.into(), "--model".into(), model.into()]);
    if b("quick") {
        cmd.push("--quick".into());
    } else {
        let passes = gi(o, "passes", 20);
        if passes < 1 {
            return Err("Passes must be at least 1.".into());
        }
        cmd.extend(["--passes".into(), passes.to_string(), "--warmup".into(), gi(o, "warmup", 3).max(0).to_string()]);
    }
    let phases: Vec<&str> = ["decode", "prefill", "concurrency"].into_iter().filter(|k| b(k)).collect();
    if phases.is_empty() {
        return Err("Pick at least one phase (decode, prefill or concurrency).".into());
    }
    if phases.len() < 3 {
        cmd.extend(phases.iter().map(|p| format!("--{p}")));
    }
    if b("decode") || b("concurrency") {
        let cats: Vec<String> =
            o.get("categories").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        if cats.is_empty() {
            return Err("Pick at least one prompt category (the decode and concurrency phases need prompts).".into());
        }
        if !BB_CATEGORIES.iter().all(|c| cats.iter().any(|x| x == c)) {
            cmd.push("--categories".into());
            cmd.extend(cats);
        }
    }
    if b("greedy") {
        cmd.push("--greedy".into());
    }
    let seed = s("seed");
    if !seed.is_empty() {
        seed.parse::<i64>().map_err(|_| format!("Seed must be a whole number, not '{seed}'."))?;
        cmd.extend(["--seed".into(), seed]);
    }
    let mml = s("maxModelLen");
    if !mml.is_empty() {
        match mml.parse::<i64>() {
            Ok(n) if n > 0 => cmd.extend(["--max-model-len".into(), n.to_string()]),
            _ => return Err(format!("Max model length must be a positive number of tokens, not '{mml}'.")),
        }
    }
    let cfg = s("config");
    if !cfg.is_empty() {
        let p = expanduser(&cfg);
        if !p.is_file() {
            return Err(format!("Config file not found:\n{}", p.display()));
        }
        cmd.extend(["--config".into(), p.to_string_lossy().into_owned()]);
    }
    let name = s("name");
    if !name.is_empty() {
        cmd.extend(["--name".into(), name]);
    }
    if b("duoNotes") {
        for (k, v) in duo_notes {
            if !v.is_empty() {
                cmd.extend(["--note".into(), format!("{k}={v}")]);
            }
        }
    }
    for n in shlex_split(&s("notes")).map_err(|e| format!("Can't read Notes: {e}"))? {
        if !n.contains('=') || n.starts_with('=') {
            return Err(format!("Each note must be KEY=VALUE, not '{n}'."));
        }
        cmd.extend(["--note".into(), n]);
    }
    if !b("html") {
        cmd.push("--no-html".into());
    }
    if !b("updateCheck") {
        cmd.push("--no-update-check".into());
    }
    cmd.extend(shlex_split(&s("extra")).map_err(|e| format!("Can't read Extra args: {e}"))?);
    Ok(cmd)
}

impl App {
    /// Saved options merged over the defaults.
    pub fn bench_opts(&self) -> Map<String, Value> {
        let mut o = bench_defaults();
        if let Some(saved) = self.settings.g().get("betterbench").and_then(|v| v.as_object()) {
            for (k, v) in saved {
                if o.contains_key(k) {
                    o.insert(k.clone(), v.clone());
                }
            }
        }
        o
    }

    pub fn bench_ui(&self) -> Value {
        let b = &self.bench;
        let pane = &self.panes[b.p];
        json!({
            "visible": b.visible, "side": pane.side, "running": b.busy(),
            "ready": pane.web_enabled, "status": b.status, "report": b.report,
            "model": if b.busy() { b.model.clone() } else { pane.model_var.clone() },
            "backend": pane.backend, "gpu": pane.gpu_name,
            "endpoint": if b.busy() { b.endpoint.clone() } else { pane.status.get("url").cloned().unwrap_or_default() },
            "missing": b.missing, "url": BB_URL,
            "opts": self.bench_opts(), "defaults": bench_defaults(), "categories": BB_CATEGORIES,
        })
    }

    pub fn bench_here(&self, p: usize) -> bool {
        self.bench.busy() && self.bench.p == p
    }

    fn bench_check_installed(&mut self, o: &Map<String, Value>) -> Option<(String, Vec<String>)> {
        let words = match shlex_split(&gs(o, "command", "betterbench")) {
            Ok(w) if !w.is_empty() => w,
            _ => vec!["betterbench".into()],
        };
        match resolve_exe(&words[0]) {
            Some(exe) => {
                self.bench.missing.clear();
                Some((exe.to_string_lossy().into_owned(), words[1..].to_vec()))
            }
            None => {
                self.bench.missing = not_installed(&words[0]);
                None
            }
        }
    }

    /// The pane's BetterBench… button.
    pub fn bench_show(&mut self, p: usize) {
        if !self.bench.busy() {
            self.bench.p = p;
        }
        self.bench.visible = true;
        let o = self.bench_opts();
        self.bench_check_installed(&o);
    }

    /// Start: `arg` holds the window's options; they become the remembered ones.
    pub fn bench_start(&mut self, arg: &Value) {
        if self.bench.busy() {
            return;
        }
        let mut o = bench_defaults();
        if let Some(a) = arg.get("opts").and_then(|v| v.as_object()) {
            for (k, v) in a {
                if o.contains_key(k) {
                    o.insert(k.clone(), v.clone());
                }
            }
        }
        self.settings.g_mut().insert("betterbench".into(), Value::Object(o.clone()));
        self.settings.save();
        let Some((exe, rest)) = self.bench_check_installed(&o) else {
            let m = self.bench.missing.clone();
            self.msg("error", &m);
            return;
        };
        let p = self.bench.p;
        if !self.panes[p].web_enabled {
            self.msg("info", &format!("The {} pane has no model ready to benchmark. Load one and wait for 'Ready'.", self.panes[p].side));
            return;
        }
        let url = self.panes[p].status.get("url").cloned().unwrap_or_default();
        let Some((host, port)) = local_endpoint(&url) else {
            self.msg("error", &format!("Can't tell this pane's endpoint from '{url}'."));
            return;
        };
        let endpoint = format!("http://{}:{port}/v1", if host.contains(':') { format!("[{host}]") } else { host.clone() });
        // the served model name: what /v1/models reports (Extra args may set an alias), else the file's name
        let fallback = {
            let mv = &self.panes[p].model_var;
            let n = std::path::Path::new(mv).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| mv.clone());
            n.strip_suffix(".rad").map(String::from).unwrap_or(n)
        };
        let token = self.poll_token(p);
        let notes = vec![
            ("backend".to_string(), self.panes[p].backend.clone()),
            ("gpu".to_string(), self.panes[p].gpu_name.clone()),
            ("model_file".to_string(), self.panes[p].model_var.clone()),
            ("launcher".to_string(), format!("LLM Duo {}", env!("CARGO_PKG_VERSION"))),
        ];
        self.bench.starting = true;
        self.bench.status = "Asking the server for its model name…".into();
        let tx = self.job_tx.clone();
        std::thread::spawn(move || {
            let model = super::http::get_auth(&host, port, "/v1/models", Duration::from_secs(5), token.as_deref())
                .ok()
                .filter(|r| r.status == 200)
                .and_then(|r| serde_json::from_str::<Value>(&r.body).ok())
                .and_then(|v| v.get("data")?.get(0)?.get("id")?.as_str().map(String::from))
                .unwrap_or(fallback);
            let _ = tx.send(Box::new(move |app: &mut App| {
                app.bench.starting = false;
                app.bench_launch(p, &exe, &rest, &o, &endpoint, &model, &notes, token);
            }));
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn bench_launch(&mut self, p: usize, exe: &str, rest: &[String], o: &Map<String, Value>, endpoint: &str, model: &str, notes: &[(String, String)], token: Option<String>) {
        if !self.panes[p].web_enabled {
            self.bench.status = "Idle".into();
            self.msg("info", "The model stopped before the benchmark could start.");
            return;
        }
        let cmd = match build_bench_cmd(exe, rest, o, endpoint, model, notes) {
            Ok(c) => c,
            Err(e) => {
                self.bench.status = "Idle".into();
                self.msg("error", &e);
                return;
            }
        };
        let mut env: Vec<(String, String)> = std::env::vars().filter(|e| e.0 != "PYTHONUNBUFFERED").collect();
        env.push(("PYTHONUNBUFFERED".into(), "1".into()));
        if let Some(t) = token.filter(|t| !t.is_empty()) {
            // in the environment, not on the command line where ps shows it
            env.retain(|e| e.0 != "BETTERBENCH_API_KEY");
            env.push(("BETTERBENCH_API_KEY".into(), t));
        }
        self.push_event(json!({"type": "logClear", "side": "bench"}));
        self.bench.lines.clear();
        self.bench_log(&format!("$ {}", shlex_join(&cmd)));
        while self.bench.rx.try_recv().is_ok() {}
        match proc::spawn(&cmd, &env, false, self.bench.tx.clone()) {
            Ok(pr) => self.bench.proc = Some(pr),
            Err(e) => {
                self.bench_log(&format!("Failed to start: {e}"));
                self.bench.status = "Failed to start - see the log".into();
                return;
            }
        }
        self.bench.p = p;
        self.bench.model = model.to_string();
        self.bench.endpoint = endpoint.to_string();
        self.bench.report = None;
        self.bench.last_line.clear();
        self.bench.user_stopped = false;
        self.bench.t0 = Some(Instant::now());
        self.bench.status = "Running…".into();
        self.set_top_status(format!("BetterBench: benchmarking {model} ({} pane)", self.panes[p].side));
    }

    fn bench_log(&mut self, line: &str) {
        if self.bench.lines.len() >= 2000 {
            self.bench.lines.pop_front();
        }
        self.bench.lines.push_back(line.to_string());
        self.push_event(json!({"type": "log", "side": "bench", "text": line}));
    }

    pub fn bench_stop(&mut self) {
        let Some(pr) = &self.bench.proc else { return };
        self.bench.user_stopped = true;
        proc::term_group(pr.pid);
        self.bench.status = "Stopping…".into();
    }

    pub fn bench_close(&mut self) {
        if self.bench.busy() {
            self.ask(
                "A benchmark is running.\n\nYes: stop it\nNo: keep it running in the background (reopen with the pane's BetterBench… button)",
                "yesnocancel",
                Pending::BenchCloseRunning,
            );
            return;
        }
        self.bench.visible = false;
    }

    pub fn bench_open_report(&mut self) {
        if let Some(r) = self.bench.report.clone() {
            open_url(&r);
        }
    }

    pub fn bench_open_runs(&mut self) {
        let home = std::env::var("BETTERBENCH_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".betterbench"));
        let runs = home.join("runs");
        open_url(&(if runs.is_dir() { runs } else { home }).display().to_string());
    }

    pub fn bench_poll(&mut self) {
        for _ in 0..500 {
            let Ok(msg) = self.bench.rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(val) => {
                    let t = val.trim();
                    if let Some(u) = t.strip_prefix("file://") {
                        if u.ends_with(".html") {
                            self.bench.report = Some(t.to_string());
                        }
                    }
                    if !t.is_empty() {
                        self.bench.last_line = t.chars().take(160).collect();
                    }
                    self.bench_log(&val);
                }
                ProcMsg::Exit(id, rc) => self.bench_on_exit(id, rc),
            }
        }
        if self.bench.proc.is_some() && !self.bench.user_stopped {
            let el = self.bench.t0.map(|t| t.elapsed().as_secs()).unwrap_or(0);
            self.bench.status = format!("Running {}:{:02}:{:02}   •   {}", el / 3600, el / 60 % 60, el % 60, self.bench.last_line);
        }
    }

    fn bench_on_exit(&mut self, id: u64, rc: i32) {
        if self.bench.proc.as_ref().map(|p| p.id) != Some(id) {
            return;
        }
        self.bench.proc = None;
        let stopped = std::mem::take(&mut self.bench.user_stopped);
        let mins = self.bench.t0.map(|t| t.elapsed().as_secs_f64() / 60.0).unwrap_or(0.0);
        if rc == 0 {
            self.bench.status = format!("Done in {mins:.1} min") + if self.bench.report.is_some() { " - report written" } else { "" };
            self.set_top_status(format!("BetterBench: {} done", self.bench.model));
            if gb(&self.bench_opts(), "openReport", true) {
                self.bench_open_report();
            }
            return;
        }
        self.bench.status = if stopped { "Stopped".into() } else { format!("Failed (exit code {rc}) - see the log") };
        self.set_top_status(format!("BetterBench: {}", if stopped { "stopped" } else { "failed" }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint() {
        assert_eq!(local_endpoint("http://0.0.0.0:8080/v1"), Some(("127.0.0.1".into(), 8080)));
        assert_eq!(local_endpoint("http://192.168.1.5:5001/v1"), Some(("192.168.1.5".into(), 5001)));
        assert_eq!(local_endpoint(""), None);
    }

    #[test]
    fn command() {
        let mut o = bench_defaults();
        let notes = vec![("backend".to_string(), "llama.cpp".to_string()), ("gpu".to_string(), String::new())];
        let c = build_bench_cmd("bb", &[], &o, "http://127.0.0.1:8080/v1", "m", &notes).unwrap();
        assert_eq!(
            c,
            ["bb", "run", "--endpoint", "http://127.0.0.1:8080/v1", "--model", "m", "--passes", "20", "--warmup", "3", "--note", "backend=llama.cpp"]
        );
        o.insert("quick".into(), json!(true));
        o.insert("concurrency".into(), json!(false));
        o.insert("categories".into(), json!(["code", "math"]));
        o.insert("html".into(), json!(false));
        o.insert("duoNotes".into(), json!(false));
        o.insert("notes".into(), json!("quant=q4 'tp=2'"));
        let c = build_bench_cmd("bb", &[], &o, "e", "m", &notes).unwrap();
        assert_eq!(
            c,
            ["bb", "run", "--endpoint", "e", "--model", "m", "--quick", "--decode", "--prefill", "--categories", "code", "math", "--note", "quant=q4", "--note", "tp=2", "--no-html"]
        );
        o.insert("decode".into(), json!(false));
        o.insert("prefill".into(), json!(false));
        assert!(build_bench_cmd("bb", &[], &o, "e", "m", &notes).is_err());
        o.insert("prefill".into(), json!(true));
        o.insert("categories".into(), json!([]));
        assert!(build_bench_cmd("bb", &[], &o, "e", "m", &notes).is_ok()); // prefill needs no corpus
        o.insert("notes".into(), json!("nokey"));
        assert!(build_bench_cmd("bb", &[], &o, "e", "m", &notes).is_err());
    }
}
