//! Fleet mode, the App side: pinning a Radiance pane's configuration, the fleet's settings and
//! groups, building and checking the fleet YAML, running `radiance --config`, polling
//! `/admin/fleet`, hot reload and the per-model admin actions. The text work is in `fleet.rs`.

use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use super::app::{App, Pending};
use super::fleet::*;
use super::proc::{self, Proc, ProcMsg};
use super::radiance::{
    card_weight_bytes, config_path, gen_batch_problem, gen_step, rad_arch, ngram_on_card, rad_failure_hint, rad_yaml, uses_config, write_private, yaml_log_lines, RAD_KEY_ENV,
};
use super::settings::Settings;
use super::util::*;

/// Environment variables that would hide cards from a fleet that is meant to use both.
const CARD_MASKS: [&str; 3] = ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"];
const MIB: f64 = 1024.0 * 1024.0;

pub struct FleetRun {
    /// Fleet mode is shown instead of the panes
    pub on: bool,
    pub proc: Option<Proc>,
    tx: Sender<ProcMsg>,
    rx: Receiver<ProcMsg>,
    pub lines: VecDeque<String>,
    pub status: String,
    /// `/admin/fleet` answered since the start
    pub ready: bool,
    pub rows: Vec<Value>,
    pub summary: String,
    /// the file the running process was started with or last reloaded from
    pub running_yaml: String,
    /// pins or settings changed while it runs: Apply
    pub changed: bool,
    /// start again once the stopping process has exited (Apply with a restart)
    restart: bool,
    user_stopped: bool,
    t0: Option<Instant>,
    polling: bool,
    last_poll: Option<Instant>,
    /// host / port / key of the running process, for the admin routes
    host: String,
    port: i64,
    token: Option<String>,
}

impl FleetRun {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        FleetRun {
            on: false,
            proc: None,
            tx,
            rx,
            lines: VecDeque::new(),
            status: "Stopped".into(),
            ready: false,
            rows: vec![],
            summary: String::new(),
            running_yaml: String::new(),
            changed: false,
            restart: false,
            user_stopped: false,
            t0: None,
            polling: false,
            last_poll: None,
            host: String::new(),
            port: 0,
            token: None,
        }
    }
}

impl Default for FleetRun {
    fn default() -> Self {
        Self::new()
    }
}

fn obj_of(v: Option<&Value>) -> Map<String, Value> {
    v.and_then(|x| x.as_object()).cloned().unwrap_or_default()
}

/// Bytes of a model: the .rad file, or a checkpoint folder's weight files.
fn model_bytes(p: &Path) -> f64 {
    if p.is_dir() {
        super::radiance::dir_bytes(p)
    } else {
        std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0)
    }
}

/// MemTotal from /proc/meminfo, in MiB.
fn ram_mib() -> i64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemTotal:")).map(String::from))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|k| k.parse::<i64>().ok()))
        .map(|kib| kib / 1024)
        .unwrap_or(0)
}

/// What the pin's flags pin in host memory: prefix cache + host pool, MiB.
fn pin_host_mib(flags: &[String]) -> i64 {
    ["--prefix-cache-mib", "--host-pool-mib"]
        .iter()
        .filter_map(|f| flag_value(flags, f).and_then(|v| v.parse::<i64>().ok()))
        .sum()
}

fn pin_flags_of(pin: &Value) -> Vec<String> {
    pin.get("flags").and_then(|f| f.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn local_host(h: &str) -> String {
    match h {
        "0.0.0.0" | "" => "127.0.0.1".into(),
        "::" => "::1".into(),
        h => h.to_string(),
    }
}

impl App {
    // ============================================================ storage ==
    fn fleet_pins(&self) -> Map<String, Value> {
        obj_of(self.settings.fleet().get("pins"))
    }
    fn fleet_server(&self) -> Map<String, Value> {
        let mut s = obj_of(self.settings.fleet().get("server"));
        for (k, v) in [("port", json!(8080)), ("host_mib", json!(0)), ("disk_tier", json!("")), ("load_timeout", json!("")), ("warm_up", json!(false))] {
            s.entry(k).or_insert(v);
        }
        s
    }
    fn fleet_groups(&self) -> Vec<Value> {
        self.settings.fleet().get("groups").and_then(|g| g.as_array()).cloned().unwrap_or_default()
    }
    fn fleet_save(&mut self, key: &str, v: Value) {
        self.settings.fleet_mut().insert(key.into(), v);
        self.settings.save();
        if self.fleet.proc.is_some() {
            self.fleet.changed = true;
        }
    }

    // ========================================================== the mode ==
    pub fn fleet_set_mode(&mut self, on: bool) {
        if on == self.fleet.on {
            return;
        }
        if on {
            let bin = expanduser(&gs(self.settings.g(), "radiance_bin", "radiance"));
            if !uses_config(&bin) {
                self.msg(
                    "info",
                    &format!("Fleet mode needs Radiance 2.0 or later (one process serving several models from a YAML file).\n\nThe radiance binary in Settings ({}) is older, or could not be asked.", bin.display()),
                );
                return;
            }
            let running: Vec<String> = self.panes.iter().filter(|p| p.proc.is_some() || p.r_probe.is_some()).map(|p| cap_first(&p.side)).collect();
            if !running.is_empty() {
                self.msg(
                    "info",
                    &format!("The fleet uses both cards. Stop the {} pane{} first.", running.join(" and "), if running.len() > 1 { "s" } else { "" }),
                );
                return;
            }
        } else if self.fleet.proc.is_some() {
            self.msg("info", "The fleet is running. Stop it before going back to the panes.");
            return;
        }
        self.fleet.on = on;
    }

    // ================================================================ pins ==
    /// "Pin to fleet" on a Radiance pane: freeze its configuration as a fleet model.
    pub fn fleet_pin_from_pane(&mut self, p: usize, name_override: Option<String>) {
        if !self.panes[p].is_rad() {
            return;
        }
        if !self.r_config_mode() {
            self.msg("info", "Pinning needs Radiance 2.0 or later: the fleet is a 2.0 feature.");
            return;
        }
        if self.panes[p].device.is_none() || self.panes[p].model_var.is_empty() {
            self.msg("info", "Select a model on this pane first.");
            return;
        }
        self.flush_save(p);
        let (cmd, opts) = match self.r_build_command(p, None) {
            Ok(x) => x,
            Err(e) => {
                self.msg("error", &format!("Could not build this pane's command:\n{e}"));
                return;
            }
        };
        let flags = pin_flags(&cmd);
        let path = self.model_path(p).map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        // without a name: the existing pin of this model on these card(s), even if it was renamed
        let default_name = pin_name_for(&path);
        let name = name_override
            .unwrap_or_else(|| pin_target_name(&self.fleet_pins(), &self.dev_key(p), &self.panes[p].model_var, &default_name));
        // Radiance's own parser, now: a flag it refuses is refused at pin time with its message
        let (env, _) = self.fleet_env();
        let mut line = vec![cmd[0].clone()];
        line.extend(flags.iter().cloned());
        line.extend(["--served-model-name".to_string(), name.clone()]);
        let mut parts = match rad_yaml(&line, &env) {
            Ok(y) => split_rad_yaml(&y),
            Err(e) => {
                self.msg("error", &format!("rad-yaml refused this pane's settings, so they can't be pinned:\n\n{e}"));
                return;
            }
        };
        if flag_value(&cmd, "--api-key").is_some() {
            parts.dropped.push("--api-key from Extra args (the fleet takes the key from Settings, through the environment)".into());
        }
        let tp = pin_tp(&flags);
        // the claim, estimated: weights split over the cards + one context of KV at the rate the
        // pane's last load of this model measured + its activation arena
        let pane = &self.panes[p];
        let ctx = gi(&opts, "ctx", 0) as f64;
        let same = pane.loaded.as_ref().map(|l| l.0 == pane.model_var && gs(&l.1, "r_kv", "") == gs(&opts, "r_kv", "")).unwrap_or(false);
        let kv_one = match (same, pane.r_kv_tokens, pane.r_kv_bytes) {
            (true, Some(t), Some(b)) if t > 0 => Some(b / t as f64 * ctx),
            _ => None,
        };
        // what stays off the cards doesn't count: the host pool (the model's total, shared by the
        // ranks) and an n-gram table read from RAM or disk
        let total = model_bytes(Path::new(&path));
        let table = pane.info_r().map(|i| i.ngram_bytes).unwrap_or(0.0);
        let card_min = self.r_devs(p).iter().map(App::v_total).fold(f64::INFINITY, f64::min);
        let head = gi(&opts, "r_headroom", 96) as f64 * MIB;
        let on_card = ngram_on_card(&gs(&opts, "r_ngram_place", "(default)"), total, table, tp as f64, head, card_min);
        let host_pool = gi(&opts, "r_host_pool", 0) as f64 * MIB;
        let weights = card_weight_bytes(total, table, on_card, host_pool, tp as f64);
        let claim = kv_one.map(|kv| weights + kv + pane.r_arena.unwrap_or(0.0)).unwrap_or(0.0);
        let host_mib = pin_host_mib(&flags);
        let pin = json!({
            "flags": flags,
            "enabled": true,
            "src": {
                "dev_key": self.dev_key(p),
                "model": self.panes[p].model_var,
                "side": self.panes[p].side,
                "tp": tp,
                "opts": opts.clone(),
                "claim_bytes": claim,
                "pinned_at": now_iso(),
            },
            "fleet": {"aliases": "", "priority": "", "pin": false, "ttl": "", "park_ram": "allowed", "park_restore": "auto",
                      "kv_min_tokens": "", "kv_max_tokens": "", "experts_max_mib": ""},
        });
        // tell up front what the pin means
        let mut t = format!(
            "{name}: {} at {} tokens of context, {} cache.\n\n",
            if tp > 1 { format!("tensor parallel over {tp} cards (tp {tp}): it takes both cards whenever it is loaded") } else { "one card (tp 1): Radiance picks which card when it loads".into() },
            fmt_thousands(ctx as i64),
            gs(&opts, "r_kv", "i8")
        );
        t += &if claim > 0.0 {
            format!(
                "In a fleet it claims ≈ {:.1} GiB a card (estimate: weights{}, less the host pool and an n-gram table kept off the cards, + one full-length context of KV at the rate this pane measured + activations). That much must be free beside the models already loaded, or Radiance parks or unloads others to make room.\n",
                claim / super::fleet::GIB,
                if tp > 1 { " split over the cards" } else { "" }
            )
        } else {
            "Its claim in a fleet (weights + one full-length context of KV + activations) is not measured yet: Load it on this pane once and re-pin to see the figure.\n".into()
        };
        if host_mib > 0 {
            t += &format!("It pins {:.1} GiB of host RAM (prefix cache / host pool), out of the fleet's shared host budget.\n", host_mib as f64 / 1024.0);
        }
        if !parts.dropped.is_empty() {
            t += &format!(
                "\nThese settings are process-wide in a fleet and are not part of the pin (set them in the Fleet view where it offers them):\n  {}\n",
                parts.dropped.join("\n  ")
            );
        }
        let pins = self.fleet_pins();
        if pins.contains_key(&name) {
            let taken: Vec<String> = pins.keys().cloned().collect();
            let alt = unique_name(&name, &taken);
            self.ask(
                &if name != default_name {
                    format!("{t}\nThis model is already pinned as {name}. Replace that pin?\n(No: keep both, the new one as {alt})")
                } else {
                    format!("{t}\nA pin named {name} exists. Replace it?\n(No: keep both, the new one as {alt})")
                },
                "yesnocancel",
                Pending::FleetPin { name, pin },
            );
            return;
        }
        self.fleet_store_pin(&name, pin, false);
        self.msg("info", &format!("{t}\nPinned. Switch on Fleet mode to serve it with your other pins."));
    }

    /// `replace`: over an existing pin of that name (keeping its fleet keys and enabled state).
    pub fn fleet_store_pin(&mut self, name: &str, mut pin: Value, replace: bool) {
        let mut pins = self.fleet_pins();
        if replace {
            if let Some(old) = pins.get(name) {
                for k in ["fleet", "enabled"] {
                    if let Some(v) = old.get(k) {
                        pin[k] = v.clone();
                    }
                }
            }
        }
        pins.insert(name.to_string(), pin);
        self.fleet_save("pins", Value::Object(pins));
        self.set_top_status(format!("Pinned {name} to the fleet"));
    }

    pub fn fleet_answer_pin(&mut self, name: String, pin: Value, ans: &str) {
        match ans {
            "yes" => self.fleet_store_pin(&name, pin, true),
            "no" => {
                let taken: Vec<String> = self.fleet_pins().keys().cloned().collect();
                let alt = unique_name(&name, &taken);
                self.fleet_store_pin(&alt, pin, false);
            }
            _ => {}
        }
    }

    /// Re-pin from the pane that shows the same model on the same kind of card(s).
    fn fleet_repin(&mut self, name: &str) {
        let pins = self.fleet_pins();
        let Some(pin) = pins.get(name) else { return };
        let (dev_key, model) = (
            pin.pointer("/src/dev_key").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            pin.pointer("/src/model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        );
        let found = (0..2).find(|&p| self.panes[p].is_rad() && self.panes[p].model_var == model && self.dev_key(p) == dev_key);
        let Some(p) = found else {
            let dual = dev_key.contains("DUAL:");
            self.msg(
                "info",
                &format!(
                    "To re-pin {name}, select {model} on a Radiance pane{}, then come back here.\n\n(Switch Fleet mode off to see the panes.)",
                    if dual { " with Dual GPU mode on" } else { " with Dual GPU mode off" }
                ),
            );
            return;
        };
        self.fleet_pin_from_pane(p, Some(name.to_string()));
    }

    /// The pane's saved options for the pin's model and card(s) differ from the pinned ones.
    fn fleet_pin_differs(&self, pin: &Value) -> bool {
        let (Some(dev), Some(model), Some(opts)) = (
            pin.pointer("/src/dev_key").and_then(|v| v.as_str()),
            pin.pointer("/src/model").and_then(|v| v.as_str()),
            pin.pointer("/src/opts").and_then(|v| v.as_object()),
        ) else {
            return false;
        };
        let Some(saved) = self.settings.data.get("models").and_then(|m| m.get(&Settings::model_key(dev, model))).and_then(|v| v.as_object()) else {
            return false;
        };
        opts.iter().any(|(k, v)| saved.get(k).map(|s| py_str(s) != py_str(v)).unwrap_or(false))
    }

    // ============================================================ the file ==
    /// Environment for radiance and rad-yaml: this process's, without card masks (the fleet
    /// uses both cards), with RADIANCE_HOME, Settings' Radiance environment and the API key.
    /// Returns the environment and notes for the log.
    fn fleet_env(&self) -> (Vec<(String, String)>, Vec<String>) {
        let g = self.settings.g().clone();
        let mut notes = vec![];
        let mut env: Vec<(String, String)> = std::env::vars().collect();
        for m in CARD_MASKS {
            if env.iter().any(|e| e.0 == m) {
                notes.push(format!("[LLM Duo] {m} from LLM Duo's own environment is not passed on: the fleet uses every card"));
                env.retain(|e| e.0 != m);
            }
        }
        let mut set = |k: &str, v: String| {
            env.retain(|e| e.0 != k);
            env.push((k.to_string(), v));
        };
        let home = gs(&g, "radiance_home", "");
        if !home.trim().is_empty() {
            set("RADIANCE_HOME", expanduser(home.trim()).to_string_lossy().into_owned());
        }
        let models = gs(&g, "radiance_models_dir", "");
        for part in shlex_split(&gs(&g, "radiance_env", "")).unwrap_or_default() {
            if let Some((k, v)) = part.split_once('=') {
                if !k.is_empty() && !CARD_MASKS.contains(&k) {
                    set(k, v.replace("{models}", &models));
                }
            }
        }
        if let Some(key) = self.fleet_key() {
            set(RAD_KEY_ENV, key);
        }
        (env, notes)
    }

    fn fleet_key(&self) -> Option<String> {
        let k = gs(self.settings.g(), "radiance_api_key", "").trim().to_string();
        if k.is_empty() { None } else { Some(k) }
    }

    /// The enabled pins, sorted by name.
    fn fleet_enabled(&self) -> Vec<(String, Value)> {
        let mut v: Vec<(String, Value)> = self.fleet_pins().into_iter().filter(|(_, p)| p.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// The fleet YAML from the enabled pins (each through rad-yaml).
    pub fn fleet_build(&self) -> Result<String, String> {
        let bin = expanduser(&gs(self.settings.g(), "radiance_bin", "radiance")).to_string_lossy().into_owned();
        let (env, _) = self.fleet_env();
        let mut models = vec![];
        for (name, pin) in self.fleet_enabled() {
            let mut line = vec![bin.clone()];
            line.extend(pin_flags_of(&pin));
            line.extend(["--served-model-name".to_string(), name.clone()]);
            let y = rad_yaml(&line, &env).map_err(|e| format!("{name}: rad-yaml refused its settings:\n{e}"))?;
            let parts = split_rad_yaml(&y);
            models.push(FleetModel { name: name.clone(), body: parts.body, keys: fleet_key_lines(&obj_of(pin.get("fleet"))) });
        }
        let host = gs(self.settings.g(), "host", "127.0.0.1");
        let key_env = self.fleet_key().map(|_| RAD_KEY_ENV);
        Ok(fleet_yaml(&self.fleet_server(), &host, key_env, &models, &self.fleet_groups()))
    }

    fn fleet_checks(&self) -> (Vec<String>, Vec<String>) {
        let checks: Vec<PinCheck> = self
            .fleet_enabled()
            .into_iter()
            .map(|(name, pin)| {
                let flags = pin_flags_of(&pin);
                let f = obj_of(pin.get("fleet"));
                let model = flag_value(&flags, "--model").unwrap_or_default();
                PinCheck {
                    aliases: alias_list(&gs(&f, "aliases", "")),
                    bad_aliases: bad_aliases(&gs(&f, "aliases", "")),
                    tp: pin_tp(&flags),
                    // a Hugging Face repo id is fetched by Radiance, not a path here
                    model_exists: !model.starts_with('/') || Path::new(&model).exists(),
                    claim: pin.pointer("/src/claim_bytes").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    host_mib: pin_host_mib(&flags),
                    priority: gs(&f, "priority", "").trim().parse().ok(),
                    // a generator: does its Batch tokens hold the largest request? (None: Radiance
                    // sizes it, the flag not given)
                    gen_note: if model.starts_with('/') {
                        rad_arch(Path::new(&model)).and_then(|a| gen_step(&a)).and_then(|g| {
                            let b = flag_value(&flags, "--max-num-batched-tokens")?.parse::<i64>().ok()?;
                            // only what is refused at the plugin's default (a guidance-only note stays on the pane)
                            gen_batch_problem(g, b, false).filter(|(red, _)| *red).map(|(_, t)| t)
                        })
                    } else {
                        None
                    },
                    name,
                }
            })
            .collect();
        let card = self.devices.iter().map(|d| d.total_mib).filter(|m| *m > 0).min().unwrap_or(0) as f64 * MIB;
        let budget = match gi(&self.fleet_server(), "host_mib", 0) {
            n if n > 0 => n,
            _ => ram_mib() * 3 / 4,
        };
        check_pins(&checks, self.devices.len().max(1), card, budget)
    }

    // ============================================================= actions ==
    /// Check: the pre-Load checks, then `radiance --config F --check` (parses only: no GPU).
    fn fleet_check(&mut self) {
        let (err, warn) = self.fleet_checks();
        let yaml = match self.fleet_build() {
            Ok(y) => y,
            Err(e) => {
                self.msg("error", &e);
                return;
            }
        };
        let path = config_path("fleet-check");
        if let Err(e) = write_private(&path, &yaml) {
            self.msg("error", &e);
            return;
        }
        let bin = expanduser(&gs(self.settings.g(), "radiance_bin", "radiance")).to_string_lossy().into_owned();
        let (env, _) = self.fleet_env();
        let cmd = vec![bin, "--config".into(), path.to_string_lossy().into_owned(), "--check".into()];
        let tx = self.job_tx.clone();
        self.fleet.status = "Checking…".into();
        std::thread::spawn(move || {
            let r = super::radiance::run_capture(&cmd, &env, Duration::from_secs(30));
            let _ = tx.send(Box::new(move |app: &mut App| {
                app.fleet.status = if app.fleet.proc.is_some() { "Running".into() } else { "Stopped".into() };
                let mut t = String::new();
                match r {
                    Ok((true, out)) => t += &format!("Radiance read the file and accepts it:\n{}\n", out.trim()),
                    Ok((false, out)) => t += &format!("Radiance refuses the file:\n{}\n", out.trim()),
                    Err(e) => t += &format!("Could not run radiance --check: {e}\n"),
                }
                t += "\n(--check only parses and validates; it does not check that the models fit on the cards.)\n";
                if !err.is_empty() {
                    t += &format!("\nErrors (Load refuses these):\n• {}\n", err.join("\n• "));
                }
                if !warn.is_empty() {
                    t += &format!("\nWarnings:\n• {}\n", warn.join("\n• "));
                }
                app.report("Fleet check", &t, None);
            }));
        });
    }

    pub fn fleet_load(&mut self, confirmed: bool) {
        if self.fleet.proc.is_some() {
            return;
        }
        let g = self.settings.g().clone();
        let bin = expanduser(&gs(&g, "radiance_bin", ""));
        if !bin.is_file() {
            self.msg("error", &format!("Radiance binary not found:\n{}\n\nSet it in Settings.", bin.display()));
            return;
        }
        let running: Vec<String> = self.panes.iter().filter(|p| p.proc.is_some() || p.r_probe.is_some()).map(|p| cap_first(&p.side)).collect();
        if !running.is_empty() {
            self.msg("error", &format!("The fleet uses both cards, and the {} pane still runs a server. Stop it first.", running.join(" and ")));
            return;
        }
        let (err, warn) = self.fleet_checks();
        if !err.is_empty() {
            self.msg("error", &format!("The fleet can't start:\n\n• {}", err.join("\n• ")));
            return;
        }
        if !warn.is_empty() && !confirmed {
            self.ask(&format!("• {}\n\nStart the fleet anyway?", warn.join("\n\n• ")), "yesno", Pending::FleetLoadAnyway);
            return;
        }
        let yaml = match self.fleet_build() {
            Ok(y) => y,
            Err(e) => {
                self.msg("error", &e);
                return;
            }
        };
        let path = config_path(FLEET_TAG);
        if let Err(e) = write_private(&path, &yaml) {
            self.msg("error", &format!("Could not write the fleet's configuration:\n{e}"));
            return;
        }
        let (env, notes) = self.fleet_env();
        let cmd = vec![bin.to_string_lossy().into_owned(), "--config".into(), path.to_string_lossy().into_owned()];
        self.push_event(json!({"type": "logClear", "side": "fleet"}));
        self.fleet.lines.clear();
        let shown: Vec<(String, String)> = env.iter().filter(|e| e.0 == "RADIANCE_HOME").cloned().collect();
        self.fleet_log(&format!("$ {} {}", env_text(&shown), shlex_join(&cmd)));
        for l in yaml_log_lines(&yaml) {
            self.fleet_log(&l);
        }
        for n in notes {
            self.fleet_log(&n);
        }
        while self.fleet.rx.try_recv().is_ok() {}
        match proc::spawn_server(&cmd, &env, false, self.fleet.tx.clone()) {
            Ok(pr) => self.fleet.proc = Some(pr),
            Err(e) => {
                self.fleet_log(&format!("Failed to start: {e}"));
                self.fleet.status = "Failed to start - see the log".into();
                return;
            }
        }
        let srv = self.fleet_server();
        self.fleet.host = local_host(&gs(&g, "host", "127.0.0.1"));
        self.fleet.port = port_of(&srv);
        self.fleet.token = self.fleet_key();
        self.fleet.running_yaml = yaml;
        self.fleet.changed = false;
        self.fleet.ready = false;
        self.fleet.rows.clear();
        self.fleet.summary.clear();
        self.fleet.user_stopped = false;
        self.fleet.t0 = Some(Instant::now());
        self.fleet.status = "Starting…".into();
        self.set_top_status(format!("Fleet: starting {} model(s) on port {}", self.fleet_enabled().len(), self.fleet.port));
    }

    pub fn fleet_stop(&mut self, wait: bool) {
        let Some(pr) = &self.fleet.proc else { return };
        self.fleet.user_stopped = true;
        self.fleet.status = "Stopping…".into();
        proc::stop_group(pr.pid, pr.child.clone(), 20, wait);
    }

    /// Apply edits to the running fleet: hot reload, or a restart when a section reload can't
    /// change is different.
    fn fleet_apply(&mut self, restart_ok: bool) {
        if self.fleet.proc.is_none() {
            self.msg("info", "The fleet isn't running: your changes are saved and are used at the next Load.");
            return;
        }
        let yaml = match self.fleet_build() {
            Ok(y) => y,
            Err(e) => {
                self.msg("error", &e);
                return;
            }
        };
        let need = restart_sections(&self.fleet.running_yaml, &yaml);
        if !need.is_empty() {
            if !restart_ok {
                self.ask(
                    &format!(
                        "The {} section changed, which Radiance only reads at start. Restart the fleet now? (Every model reloads.)",
                        need.join(", ")
                    ),
                    "yesno",
                    Pending::FleetRestart,
                );
                return;
            }
            self.fleet.restart = true;
            self.fleet_stop(false);
            return;
        }
        if yaml == self.fleet.running_yaml {
            self.fleet.changed = false;
            self.set_top_status("Fleet: nothing to apply");
            return;
        }
        let (err, _) = self.fleet_checks();
        if !err.is_empty() {
            self.msg("error", &format!("Not applied:\n\n• {}", err.join("\n• ")));
            return;
        }
        if let Err(e) = write_private(&config_path(FLEET_TAG), &yaml) {
            self.msg("error", &e);
            return;
        }
        self.fleet_log("[LLM Duo] applying the changed configuration (POST /admin/reload): changed models reload, the others keep serving");
        for l in yaml_log_lines(&yaml) {
            self.fleet_log(&l);
        }
        let (host, port, token) = (self.fleet.host.clone(), self.fleet.port, self.fleet.token.clone());
        let tx = self.job_tx.clone();
        self.fleet.status = "Reloading…".into();
        std::thread::spawn(move || {
            // a reload drains and unloads the changed models first
            let r = super::http::post_auth(&host, port, "/admin/reload", Duration::from_secs(300), token.as_deref());
            let _ = tx.send(Box::new(move |app: &mut App| {
                app.fleet.status = "Running".into();
                match r {
                    Ok(resp) if resp.status == 200 => {
                        app.fleet.running_yaml = yaml;
                        app.fleet.changed = false;
                        app.fleet_log("[LLM Duo] reload applied");
                        app.set_top_status("Fleet: changes applied");
                    }
                    Ok(resp) => {
                        let m = format!("Radiance refused the reload (HTTP {}):\n{}", resp.status, resp.body.trim());
                        app.fleet_log(&format!("[LLM Duo] {}", m.replace('\n', " ")));
                        app.msg("warning", &m);
                    }
                    Err(e) => app.msg("warning", &format!("The reload request failed: {e}")),
                }
            }));
        });
    }

    pub fn fleet_apply_restart(&mut self) {
        self.fleet_apply(true);
    }

    /// load / park / unload / pin / unpin one model through /admin/models/NAME/ACTION.
    fn fleet_model_action(&mut self, name: &str, action: &str) {
        if self.fleet.proc.is_none() || !["load", "park", "unload", "pin", "unpin"].contains(&action) {
            return;
        }
        let path = format!("/admin/models/{}/{action}{}", name, if action == "load" { "?wait=0" } else { "" });
        let (host, port, token) = (self.fleet.host.clone(), self.fleet.port, self.fleet.token.clone());
        let (tx, name, action) = (self.job_tx.clone(), name.to_string(), action.to_string());
        self.fleet_log(&format!("[LLM Duo] POST {path}"));
        std::thread::spawn(move || {
            let r = super::http::post_auth(&host, port, &path, Duration::from_secs(180), token.as_deref());
            let _ = tx.send(Box::new(move |app: &mut App| match r {
                Ok(resp) if resp.status < 300 => app.fleet_log(&format!("[LLM Duo] {action} {name}: {}", resp.body.trim())),
                Ok(resp) => {
                    app.fleet_log(&format!("[LLM Duo] {action} {name} refused (HTTP {}): {}", resp.status, resp.body.trim()));
                    app.set_top_status(format!("Fleet: {action} {name} refused - see the log"));
                }
                Err(e) => app.fleet_log(&format!("[LLM Duo] {action} {name} failed: {e}")),
            }));
        });
    }

    // =============================================================== loop ==
    pub fn fleet_poll(&mut self) {
        for _ in 0..2000 {
            let Ok(msg) = self.fleet.rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(l) => self.fleet_log(&l),
                ProcMsg::Exit(id, rc) => self.fleet_on_exit(id, rc),
            }
        }
        if self.fleet.proc.is_none() || self.fleet.polling || self.fleet.user_stopped {
            return;
        }
        if self.fleet.last_poll.map(|t| t.elapsed() < Duration::from_secs(1)).unwrap_or(false) {
            return;
        }
        self.fleet.polling = true;
        self.fleet.last_poll = Some(Instant::now());
        let id = self.fleet.proc.as_ref().map(|p| p.id).unwrap_or(0);
        let (host, port, token) = (self.fleet.host.clone(), self.fleet.port, self.fleet.token.clone());
        let tx = self.job_tx.clone();
        std::thread::spawn(move || {
            let r = super::http::get_auth(&host, port, "/admin/fleet", Duration::from_secs(2), token.as_deref());
            let _ = tx.send(Box::new(move |app: &mut App| {
                app.fleet.polling = false;
                if app.fleet.proc.as_ref().map(|p| p.id) != Some(id) {
                    return;
                }
                let Ok(resp) = r else { return };
                if resp.status != 200 {
                    if resp.status == 401 || resp.status == 403 {
                        app.fleet.summary = format!("/admin/fleet answered {}: check the API key in Settings", resp.status);
                    }
                    return;
                }
                let Ok(v) = serde_json::from_str::<Value>(&resp.body) else { return };
                if !app.fleet.ready {
                    app.fleet.ready = true;
                    app.fleet.status = "Running".into();
                    let secs = app.fleet.t0.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                    app.set_top_status(format!("Fleet: listening on port {} after {secs:.0} s", app.fleet.port));
                }
                app.fleet.rows = status_rows(&v);
                app.fleet.summary = status_summary(&v);
            }));
        });
    }

    fn fleet_on_exit(&mut self, id: u64, rc: i32) {
        if self.fleet.proc.as_ref().map(|p| p.id) != Some(id) {
            return;
        }
        self.fleet.proc = None;
        self.fleet.ready = false;
        self.fleet.rows.clear();
        self.fleet.summary.clear();
        self.fleet.changed = false;
        let stopped = std::mem::take(&mut self.fleet.user_stopped);
        self.fleet_log(&format!("[LLM Duo] radiance exited (code {rc})"));
        if std::mem::take(&mut self.fleet.restart) {
            self.fleet.status = "Stopped".into();
            self.fleet_load(true);
            return;
        }
        if stopped || rc == 0 {
            self.fleet.status = "Stopped".into();
            self.set_top_status("Fleet stopped");
            return;
        }
        self.fleet.status = format!("Exited (code {rc}) - see the log");
        let lines: Vec<String> = self.fleet.lines.iter().cloned().collect();
        let hint = rad_failure_hint(&lines);
        let tail: Vec<String> = lines.iter().rev().take(15).rev().cloned().collect();
        self.msg(
            "warning",
            &format!(
                "The fleet stopped (exit code {rc}).\n\n{}Last lines:\n{}",
                if hint.is_empty() { String::new() } else { format!("Likely cause: {hint}\n\n") },
                tail.join("\n")
            ),
        );
    }

    fn fleet_log(&mut self, line: &str) {
        if self.fleet.lines.len() >= 5000 {
            self.fleet.lines.pop_front();
        }
        self.fleet.lines.push_back(line.to_string());
        self.push_event(json!({"type": "log", "side": "fleet", "text": line}));
    }

    // ============================================================ editing ==
    fn fleet_set_pin(&mut self, name: &str, key: &str, value: &Value) {
        let mut pins = self.fleet_pins();
        if key == "name" {
            let new = py_str(value).trim().to_string();
            if new == name {
                return;
            }
            if let Some(why) = name_problem(&new) {
                self.msg("info", &format!("Not renamed: {why}"));
                return;
            }
            if pins.contains_key(&new) {
                self.msg("info", &format!("A pin named {new} exists already."));
                return;
            }
            let Some(p) = pins.remove(name) else { return };
            pins.insert(new.clone(), p);
            let groups: Vec<Value> = self
                .fleet_groups()
                .into_iter()
                .map(|mut g| {
                    if let Some(Value::Array(m)) = g.get_mut("members") {
                        for x in m.iter_mut() {
                            if x.as_str() == Some(name) {
                                *x = Value::from(new.clone());
                            }
                        }
                    }
                    g
                })
                .collect();
            self.fleet_save("groups", Value::Array(groups));
            self.fleet_save("pins", Value::Object(pins));
            return;
        }
        let Some(p) = pins.get_mut(name) else { return };
        if key == "enabled" {
            p["enabled"] = Value::Bool(truthy(Some(value)));
        } else {
            if p.get("fleet").map(|f| !f.is_object()).unwrap_or(true) {
                p["fleet"] = json!({});
            }
            p["fleet"][key] = value.clone();
        }
        self.fleet_save("pins", Value::Object(pins));
    }

    fn fleet_unpin(&mut self, name: &str) {
        let mut pins = self.fleet_pins();
        if pins.remove(name).is_some() {
            self.fleet_save("pins", Value::Object(pins));
        }
    }

    fn fleet_set_server(&mut self, key: &str, value: &Value) {
        let mut s = self.fleet_server();
        s.insert(key.to_string(), value.clone());
        self.fleet_save("server", Value::Object(s));
    }

    fn fleet_add_group(&mut self, kind: &str, members: &str) {
        let m = alias_list(members);
        let pins = self.fleet_pins();
        let unknown: Vec<&String> = m.iter().filter(|x| !pins.contains_key(*x)).collect();
        if m.len() < 2 || !unknown.is_empty() || !["together", "exclusive"].contains(&kind) {
            self.msg(
                "info",
                &if unknown.is_empty() {
                    "A group needs at least two pinned model names, separated by commas.".to_string()
                } else {
                    format!("Not pinned: {}", unknown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
                },
            );
            return;
        }
        let mut g = self.fleet_groups();
        g.push(json!({"kind": kind, "members": m}));
        self.fleet_save("groups", Value::Array(g));
    }

    fn fleet_del_group(&mut self, i: usize) {
        let mut g = self.fleet_groups();
        if i < g.len() {
            g.remove(i);
            self.fleet_save("groups", Value::Array(g));
        }
    }

    // ================================================================= UI ==
    pub fn fleet_ui(&self) -> Value {
        let mut pins: Vec<Value> = self
            .fleet_pins()
            .into_iter()
            .map(|(name, pin)| {
                let flags = pin_flags_of(&pin);
                let tp = pin_tp(&flags);
                let model = flag_value(&flags, "--model").unwrap_or_default();
                let claim = pin.pointer("/src/claim_bytes").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let host = pin_host_mib(&flags);
                let ctx = flag_value(&flags, "--max-model-len").unwrap_or_default();
                let mut info = vec![
                    format!("ctx {}", ctx.parse::<i64>().map(fmt_thousands).unwrap_or(ctx)),
                    flag_value(&flags, "--kv-cache-dtype").unwrap_or_else(|| "bf16".into()),
                    format!("seqs {}", flag_value(&flags, "--max-num-seqs").unwrap_or_else(|| "?".into())),
                ];
                info.push(if claim > 0.0 { format!("claim ≈ {:.1} GiB/card", claim / super::fleet::GIB) } else { "claim not measured".into() });
                if host > 0 {
                    info.push(format!("host {:.1} GiB", host as f64 / 1024.0));
                }
                json!({
                    "name": name,
                    "model": Path::new(&model).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(model.clone()),
                    "missing": model.starts_with('/') && !Path::new(&model).exists(),
                    "tp": tp,
                    "info": info.join(" · "),
                    "enabled": pin.get("enabled").and_then(|e| e.as_bool()).unwrap_or(true),
                    "differs": self.fleet_pin_differs(&pin),
                    "pinnedAt": pin.pointer("/src/pinned_at").and_then(|v| v.as_str()).unwrap_or(""),
                    "aliasError": bad_aliases(&obj_of(pin.get("fleet")).get("aliases").map(py_str).unwrap_or_default()).join("\n"),
                    "fleet": obj_of(pin.get("fleet")),
                })
            })
            .collect();
        pins.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        let groups: Vec<Value> = self
            .fleet_groups()
            .iter()
            .filter_map(group_of)
            .map(|(k, m)| json!({"kind": k, "members": m.join(", ")}))
            .collect();
        let running = self.fleet.proc.is_some();
        json!({
            "on": self.fleet.on,
            "running": running,
            "ready": self.fleet.ready,
            "status": self.fleet.status,
            "summary": self.fleet.summary,
            "changed": self.fleet.changed,
            "pins": pins,
            "server": self.fleet_server(),
            "groups": groups,
            "rows": self.fleet.rows,
            "endpoint": if running { format!("http://{}:{}/v1", self.fleet.host, self.fleet.port) } else { String::new() },
            "yamlPath": config_path(FLEET_TAG).to_string_lossy(),
            "pinnable": self.panes.iter().any(|p| p.is_rad()) && self.r_config_mode(),
        })
    }

    pub fn fleet_call(&mut self, action: &str, arg: &Value) -> Value {
        let s = |k: &str| arg.get(k).map(py_str).unwrap_or_default();
        let val = arg.get("value").cloned().unwrap_or(Value::Null);
        match action {
            "setMode" => self.fleet_set_mode(truthy(arg.get("on"))),
            "setPin" => self.fleet_set_pin(&s("name"), &s("key"), &val),
            "unpin" => self.fleet_unpin(&s("name")),
            "repin" => self.fleet_repin(&s("name")),
            "setServer" => self.fleet_set_server(&s("key"), &val),
            "addGroup" => self.fleet_add_group(&s("kind"), &s("members")),
            "delGroup" => self.fleet_del_group(arg.get("index").and_then(|i| i.as_u64()).unwrap_or(u64::MAX) as usize),
            "check" => self.fleet_check(),
            "load" => self.fleet_load(false),
            "stop" => self.fleet_stop(false),
            "apply" => self.fleet_apply(false),
            "model" => self.fleet_model_action(&s("name"), &s("action")),
            "openDash" => {
                if self.fleet.proc.is_some() {
                    let h = if self.fleet.host.contains(':') { format!("[{}]", self.fleet.host) } else { self.fleet.host.clone() };
                    open_url(&format!("http://{h}:{}/dashboard", self.fleet.port));
                }
            }
            "copyYaml" => match self.fleet_build() {
                Ok(y) => {
                    self.clipboard(&y);
                    self.set_top_status("Fleet YAML copied");
                }
                Err(e) => self.msg("error", &e),
            },
            "copyLog" => {
                let t = self.fleet.lines.iter().cloned().collect::<Vec<_>>().join("\n");
                self.clipboard(&t);
            }
            _ => {}
        }
        Value::Null
    }
}

fn cap_first(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

