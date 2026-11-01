//! Running a pane's server: command lines, launch, stop, log parsing, HTTP polling, spill
//! detection, calibration and crash reports.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use super::app::{App, Timer};
use super::consts::*;
use super::devices::{device_env, gpu_gtt, gpu_stats, Device};
use super::estimate::est_simple;
use super::kobold::{eval_match, kobold_failure_hint, kre, lre};
use super::mtp::make_relaxed_template;
use super::pane::{cap_side, Metrics, Mon, PaneMsg};
use super::proc::{self, ProcMsg};
use super::util::*;
use super::vllm::{parse_vllm_metrics, vllm_failure_hint, vllm_process_env, vre};

impl App {
    // =========================================================== command ==
    /// The server command line, with the API key / password from Settings added in the flag the
    /// backend understands (unless Extra args already sets one).
    pub fn build_command(&mut self, p: usize) -> Result<(Vec<String>, Opts), String> {
        let (mut cmd, s) = self.build_command_base(p)?;
        if let Some((flag, key)) = self.server_secret(p) {
            let given = cmd.iter().any(|a| a == "--api-key-env" || SECRET_FLAGS.iter().any(|f| a == f || a.starts_with(&format!("{f}="))));
            if !given {
                if self.r_key_in_env(p) {
                    // on the command line every process on the machine could read it in ps
                    cmd.extend(["--api-key-env".to_string(), super::radiance::RAD_KEY_ENV.to_string()]);
                } else {
                    cmd.extend([flag.to_string(), key]);
                }
            }
        }
        Ok((cmd, s))
    }

    /// (flag, value) of this pane's backend secret, when one is set in Settings.
    pub fn server_secret(&self, p: usize) -> Option<(&'static str, String)> {
        let g = self.settings.g();
        let pane = &self.panes[p];
        let (flag, key) = if pane.is_kobold() {
            ("--password", "kobold_password")
        } else if pane.is_vllm() {
            ("--api-key", "vllm_api_key")
        } else if pane.is_rad() {
            ("--api-key", "radiance_api_key")
        } else {
            ("--api-key", "llama_api_key")
        };
        let v = gs(g, key, "").trim().to_string();
        if v.is_empty() { None } else { Some((flag, v)) }
    }

    /// The token LLM Duo's own polling sends: the Settings secret, or one given in Extra args.
    pub fn poll_token(&self, p: usize) -> Option<String> {
        let cmd = &self.panes[p].cmd;
        for (i, a) in cmd.iter().enumerate() {
            if a == "--api-key-env" {
                let name = cmd.get(i + 1)?;
                if let Some(v) = self.panes[p].env_extra.iter().find(|e| &e.0 == name) {
                    return Some(v.1.clone());
                }
                return std::env::var(name).ok();
            }
            for f in SECRET_FLAGS {
                if a == f {
                    return cmd.get(i + 1).cloned();
                }
                if let Some(v) = a.strip_prefix(&format!("{f}=")) {
                    return Some(v.to_string());
                }
            }
        }
        self.server_secret(p).map(|x| x.1)
    }

    fn build_command_base(&mut self, p: usize) -> Result<(Vec<String>, Opts), String> {
        if self.panes[p].is_rad() {
            return self.r_build_command(p, None);
        }
        if self.panes[p].is_vllm() {
            return self.v_build_command(p);
        }
        if self.panes[p].is_kobold() {
            return self.k_build_command(p);
        }
        let s = self.current_options(p);
        let path = self.model_path(p).ok_or("no model selected")?;
        let g = self.settings.g().clone();
        let build = gs(&s, "kbuild", "ROCm");
        let vk = build == "Vulkan";
        let kv = gs(&s, "kv", "f16");
        let mut cmd: Vec<String> = vec![
            self.llama_bin(&build),
            "-m".into(),
            path.to_string_lossy().into_owned(),
            "-c".into(),
            gi(&s, "ctx", 8192).to_string(),
            "-fa".into(),
            gs(&s, "fa", "on"),
            "-ctk".into(),
            kv.clone(),
            "-ctv".into(),
            kv,
            "-np".into(),
            gi(&s, "parallel", 1).to_string(),
            "-b".into(),
            gi(&s, "batch", 2048).to_string(),
            "-ub".into(),
            gi(&s, "ubatch", 512).to_string(),
            "--host".into(),
            gs(&g, "host", "127.0.0.1"),
            "--port".into(),
            gi(&s, "port", 8080).to_string(),
            "--metrics".into(),
        ];
        let mut dual_plan = None;
        if self.dual_active(p) {
            dual_plan = self.dual_plan(p, &s);
            self.panes[p].last_plan = dual_plan.clone();
            cmd.extend(["-ngl".into(), "999".into()]);
            // Vulkan: GGML_VK_VISIBLE_DEVICES lists the cards in plan order -> Vulkan0, Vulkan1
            if let Some(pl) = &dual_plan {
                if pl.mode == "none" {
                    let dev = if vk { "Vulkan0".to_string() } else { pl.single.as_ref().map(|d| d.id.clone()).unwrap_or_default() };
                    cmd.extend(["-sm".into(), "none".into(), "-dev".into(), dev]);
                } else {
                    let devs: Vec<String> = pl
                        .order
                        .iter()
                        .enumerate()
                        .map(|(i, d)| if vk { format!("Vulkan{i}") } else { d.id.clone() })
                        .collect();
                    cmd.extend([
                        "-sm".into(),
                        pl.mode.clone(),
                        "-dev".into(),
                        devs.join(","),
                        "-ts".into(),
                        pl.ts.clone().unwrap_or_else(|| "1,1".into()),
                        "-mg".into(),
                        "1".into(),
                    ]);
                }
            }
        } else if gi(&s, "ngl", 999) >= 0 {
            // -1 = omit, so llama.cpp auto-fits layers to free VRAM
            cmd.extend(["-ngl".into(), gi(&s, "ngl", 999).to_string()]);
        }
        if gi(&s, "threads", 0) > 0 {
            cmd.extend(["-t".into(), gi(&s, "threads", 0).to_string()]);
        }
        let ncm = gi(&s, "n_cpu_moe", 0);
        if let Some(cs) = dual_plan.as_ref().and_then(|pl| pl.cpu_set.clone()).filter(|c| !c.is_empty()) {
            let layers = cs.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("|");
            cmd.extend(["-ot".into(), format!("blk\\.({layers})\\.ffn_(up|down|gate|gate_up)_exps\\.=CPU")]);
        } else if ncm > 0 {
            cmd.extend(["--n-cpu-moe".into(), ncm.to_string()]);
        }
        if gb(&s, "jinja", true) {
            cmd.push("--jinja".into());
            if gb(&s, "relax_tpl", false) {
                let tpl = self.panes[p].info.as_ref().and_then(|i| i.chat_template());
                let (tpl_path, n_fixed) = make_relaxed_template(tpl, &path)?;
                cmd.extend(["--chat-template-file".into(), tpl_path.to_string_lossy().into_owned()]);
                self.set_top_status(format!(
                    "{} pane: relaxed template {}",
                    cap_side(self.panes[p].side),
                    if n_fixed > 0 {
                        format!("({n_fixed} system-message check(s) relaxed)")
                    } else {
                        "(no system-message check found; template unchanged)".into()
                    }
                ));
            }
        }
        let mm = gs(&s, "mmproj", "(none)");
        if mm != "(none)" {
            cmd.extend(["--mmproj".into(), path.parent().unwrap().join(&mm).to_string_lossy().into_owned()]);
        }
        let mtp = gs(&s, "mtp", "(off)");
        if !mtp.is_empty() && mtp != "(off)" {
            if mtp != "(built-in)" {
                cmd.extend(["--model-draft".into(), path.parent().unwrap().join(&mtp).to_string_lossy().into_owned()]);
            }
            cmd.extend([
                "--spec-type".into(),
                "draft-mtp".into(),
                "--spec-draft-n-max".into(),
                gi(&s, "mtp_n", 4).to_string(),
            ]);
            let own = gs(&s, "extra", "").contains("--spec-draft-device"); // Extra args decide
            if let (Some(pl), false) = (&dual_plan, own) {
                if self.supports(p, "--spec-draft-device") {
                    let devs = Self::vk_order(pl, &s);
                    let chosen = gs(&s, "mtp_dev", "(auto)");
                    let tgt = devs
                        .iter()
                        .position(|d| d.id == chosen)
                        .unwrap_or(if pl.mode == "none" { 0 } else { devs.len() - 1 });
                    let dev = if vk { format!("Vulkan{tgt}") } else { devs[tgt].id.clone() };
                    cmd.extend(["--spec-draft-device".into(), dev]);
                }
            }
        }
        let extra = gs(&s, "extra", "");
        if !extra.trim().is_empty() {
            cmd.extend(shlex_split(&extra)?);
        }
        Ok((cmd, s))
    }

    /// The cards a Dual GPU plan makes visible, in Vulkan numbering order: the split's order, or
    /// the one card of a single-GPU plan plus the other card when the drafter was put there.
    pub fn vk_order(pl: &super::estimate::DualPlan, s: &Opts) -> Vec<Device> {
        if pl.mode != "none" {
            return pl.order.clone();
        }
        let mut v: Vec<Device> = pl.single.clone().into_iter().collect();
        let chosen = gs(s, "mtp_dev", "(auto)");
        let mtp_on = !matches!(gs(s, "mtp", "(off)").as_str(), "(off)" | "");
        if mtp_on && v.first().map(|d| d.id != chosen).unwrap_or(false) {
            if let Some(o) = pl.order.iter().find(|d| d.id == chosen) {
                v.push(o.clone());
            }
        }
        v
    }

    pub fn run_env(&mut self, p: usize) -> Vec<(String, String)> {
        if self.panes[p].is_rad() {
            return self.r_env(p);
        }
        if self.panes[p].is_vllm() {
            return self.v_env(p);
        }
        if self.panes[p].is_kobold() {
            return self.k_env(p);
        }
        self.panes[p].k_env_notes.clear();
        if self.kbuild(p) == "Vulkan" {
            let plan = if self.dual_active(p) { self.panes[p].last_plan.clone() } else { None };
            let s = self.current_options(p);
            let devs: Vec<Device> = match plan {
                Some(pl) => Self::vk_order(&pl, &s),
                None => self.panes[p].device.clone().into_iter().collect(),
            };
            return self.vk_env(p, &devs);
        }
        if self.dual_active(p) {
            let mut env: Vec<(String, String)> = vec![];
            for d in self.dual_devices() {
                for (k, v) in device_env(&d) {
                    let prev = env.iter().find(|e| e.0 == k).map(|e| e.1.clone()).unwrap_or_default();
                    let mut set: Vec<String> = format!("{prev},{v}")
                        .trim_matches(',')
                        .split(',')
                        .map(String::from)
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    set.sort_by_key(|x| x.parse::<i64>().unwrap_or(0));
                    let joined = set.join(",");
                    if let Some(e) = env.iter_mut().find(|e| e.0 == k) {
                        e.1 = joined;
                    } else {
                        env.push((k, joined));
                    }
                }
            }
            return env;
        }
        self.panes[p].device.as_ref().map(device_env).unwrap_or_default()
    }

    pub fn copy_cmd(&mut self, p: usize) {
        if self.panes[p].model_var.is_empty() || self.panes[p].device.is_none() {
            return;
        }
        let (cmd, _) = match self.build_command(p) {
            Ok(x) => x,
            Err(e) => {
                self.msg("error", &format!("Could not build the command:\n{e}"));
                return;
            }
        };
        let env = self.run_env(p).iter().map(|(k, v)| format!("{k}={}", shlex_quote(v))).collect::<Vec<_>>().join(" ");
        let prefix = if self.panes[p].is_vllm() { self.v_copy_prefix() } else { String::new() };
        let text = prefix + &if env.is_empty() { String::new() } else { format!("{env} ") } + &shlex_join(&cmd);
        self.clipboard(&text);
    }

    // ============================================================ launch ==
    pub fn load(&mut self, p: usize) {
        if self.panes[p].proc.is_some() {
            return;
        }
        if self.panes[p].device.is_none() {
            self.msg("error", "No GPU assigned to this pane. Open Settings.");
            return;
        }
        if self.panes[p].model_var.is_empty() {
            self.msg("error", "Select a model first.");
            return;
        }
        let g = self.settings.g().clone();
        if self.panes[p].r_probe.is_some() {
            self.msg("info", "Fit is still asking Radiance for its memory plan - wait a moment (or press Fit again to cancel it).");
            return;
        }
        if self.panes[p].is_rad() && !expanduser(&gs(&g, "radiance_bin", "")).is_file() {
            self.msg(
                "error",
                &format!("Radiance binary not found:\n{}\n\nSet it in Settings.", gs(&g, "radiance_bin", "")),
            );
            return;
        }
        if self.panes[p].is_vllm() && !expanduser(&gs(&g, "vllm_bin", "")).is_file() {
            self.msg("error", &format!("vLLM binary not found:\n{}\n\nSet it in Settings.", gs(&g, "vllm_bin", "")));
            return;
        }
        if self.panes[p].is_llama() && !Path::new(&self.k_bin(p, None)).is_file() {
            self.msg(
                "error",
                &format!(
                    "llama-server {} binary not found:\n{}\n\nSet it in Settings, or switch this model's Build.",
                    self.kbuild(p),
                    self.k_bin(p, None)
                ),
            );
            return;
        }
        if self.panes[p].is_kobold() {
            let (b, exe) = (self.kbuild(p), self.k_bin(p, None));
            if exe.is_empty() || !Path::new(&exe).is_file() {
                self.msg(
                    "error",
                    &format!(
                        "KoboldCpp {b} binary not found:\n{}\n\nSet it in Settings, or switch this model's Build.",
                        if exe.is_empty() { "(not set)" } else { &exe }
                    ),
                );
                return;
            }
            if !access_x(&exe) {
                self.msg(
                    "error",
                    &format!("The KoboldCpp {b} binary isn't executable:\n{exe}\n\nRun:  chmod +x {}", shlex_quote(&exe)),
                );
                return;
            }
            if self.kobold_flags(&b).is_none() {
                // options not known yet: read them, then load
                if !self.panes[p].k_waiting {
                    self.panes[p].k_waiting = true;
                    self.set_top_status(format!(
                        "Reading KoboldCpp {b}'s options (the first run of a new binary unpacks itself, this can take ~10-30 s)…"
                    ));
                    self.fetch_kobold_flags(&b, Some(p));
                }
                return;
            }
        }
        let (cmd, s) = match self.build_command(p) {
            Ok(x) => x,
            Err(e) => {
                self.msg("error", &format!("Could not build the command:\n{e}"));
                return;
            }
        };
        // never start something the estimate says won't fit without saying so
        self.update_estimate(p);
        let confirmed = std::mem::take(&mut self.panes[p].load_confirmed);
        if !confirmed && self.panes[p].est_over && !self.panes[p].autoload {
            let est = self.panes[p].est_text.clone();
            self.ask(
                &format!(
                    "The estimate says these settings don't fit (Context {}):\n\n{est}\n\nLoad anyway? (No: change Context or press Fit first)",
                    fmt_thousands(gi(&s, "ctx", 0))
                ),
                "yesno",
                super::app::Pending::LoadAnyway(p),
            );
            return;
        }
        let o = 1 - p;
        if self.panes[o].proc.is_some() && gi(&self.current_options(o), "port", 0) == gi(&s, "port", 0) {
            self.msg("error", &format!("Port {} is already used by the other pane.", gi(&s, "port", 0)));
            return;
        }
        self.save_options(p);

        let env_extra = self.run_env(p);
        self.panes[p].env_extra = env_extra.clone();
        let mut env: Vec<(String, String)>;
        if self.panes[p].is_vllm() {
            env = vllm_process_env(self.settings.g(), &env_extra);
            env.push(("PYTHONUNBUFFERED".into(), "1".into())); // log lines arrive as they are printed
            let pane = &mut self.panes[p];
            pane.v_launch_plan = pane.vplan.clone();
            pane.v_launch_adj = pane.v_adj_used.clone();
            pane.vbuf.clear();
            pane.vkv_tokens = None;
            pane.vconc = None;
        } else {
            env = std::env::vars().collect();
            for (k, v) in &env_extra {
                env.retain(|e| &e.0 != k);
                env.push((k.clone(), v.clone()));
            }
        }
        {
            let pane = &mut self.panes[p];
            pane.cmd = cmd.clone();
            pane.lines.clear();
            pane.buffers.clear();
            pane.busy = 0;
            pane.last_prompt = None;
            pane.last_gen = None;
            pane.last_accept = None;
        }
        self.clear_log(p);
        let mut first = vec![format!(
            "$ {} {}",
            env_text(&env_extra),
            shlex_join(&mask_cmd(&cmd))
        )];
        if !self.panes[p].is_vllm() {
            first.extend(self.panes[p].k_env_notes.clone());
        }
        self.append_log(p, &first);
        self.panes[p].k_sess = (0, 0.0, 0, 0.0, 0);
        let pty = self.panes[p].is_kobold();
        if pty {
            // KoboldCpp's C++ side block-buffers its output (stats lines included) when it writes
            // to a pipe; on a (pseudo-)terminal it prints each line as it happens.
            env.retain(|e| e.0 != "PYTHONUNBUFFERED");
            env.push(("PYTHONUNBUFFERED".into(), "1".into()));
        }
        // drain stale messages of an earlier run
        while self.panes[p].proc_rx.try_recv().is_ok() {}
        let tx = self.panes[p].proc_tx.clone();
        match proc::spawn(&cmd, &env, pty, tx) {
            Ok(pr) => self.panes[p].proc = Some(pr),
            Err(e) => {
                self.panes[p].proc = None;
                let name = self.panes[p].server_name();
                self.report_failure(p, &format!("Failed to start {name}:\n{e}"), None, "GPU error");
                return;
            }
        }
        let mm = if self.panes[p].is_vllm() || self.panes[p].is_rad() { 0.0 } else { self.mmproj_bytes(p, &s) };
        {
            let pane = &mut self.panes[p];
            pane.user_stopped = false;
            pane.start_time = Some(Instant::now());
            pane.vram_base = gpu_stats(pane.device.as_ref()).0;
            pane.loaded = Some((pane.model_var.clone(), s.clone(), mm));
            pane.gtt_base = gpu_gtt(pane.device.as_ref());
            pane.spilled = false;
            pane.spill_reported = false;
            pane.ready_time = None;
            pane.mon.clear();
        }
        if self.panes[p].is_rad() {
            let mon = self.r_monitors(p);
            self.panes[p].mon = mon;
            self.panes[p].expect_full = false; // Radiance claims the whole card by design
            self.panes[p].expected_vram = 0.0;
            self.r_begin_load(p);
        } else if self.panes[p].is_vllm() {
            let mon = self.v_monitors(p);
            let exp = mon.first().map(|m| m.expected).unwrap_or(0.0);
            self.panes[p].mon = mon;
            self.panes[p].expect_full = true;
            self.panes[p].expected_vram = exp;
        } else {
            if self.dual_active(p) {
                if let Some(plan) = self.panes[p].last_plan.clone() {
                    let full = gi(&s, "n_cpu_moe", 0) == 0;
                    let used_devs: Vec<(Device, f64)> = if plan.mode == "none" {
                        vec![(
                            plan.single.clone().unwrap(),
                            plan.est[1].clone().or(plan.est[0].clone()).map(|e| e.total).unwrap_or(0.0),
                        )]
                    } else {
                        plan.order
                            .iter()
                            .enumerate()
                            .map(|(i, d)| (d.clone(), plan.est[i].clone().map(|e| e.total).unwrap_or(0.0)))
                            .collect()
                    };
                    for (d, exp) in used_devs {
                        let mon = Mon {
                            base: gpu_stats(Some(&d)).0,
                            gtt: gpu_gtt(Some(&d)),
                            dev: d,
                            peak: 0.0,
                            expected: exp,
                            full,
                            dv: None,
                            dg: None,
                            gtt_noted: 0.0,
                        };
                        self.panes[p].mon.push(mon);
                    }
                }
            }
            match self.panes[p].info_g() {
                Some(info) => {
                    let n_eff = info.n_layer.saturating_sub(info.nextn) as i64;
                    let full = gi(&s, "ngl", 999) >= n_eff && gi(&s, "n_cpu_moe", 0) == 0;
                    self.panes[p].expect_full = full;
                    self.panes[p].expected_vram = if full {
                        est_simple(&info, &s, self.mmproj_bytes(p, &s), gf(self.settings.g(), "overhead_mib", 300.0), 0.0).total
                    } else {
                        0.0
                    };
                }
                None => {
                    self.panes[p].expect_full = false;
                    self.panes[p].expected_vram = 0.0;
                }
            }
        }
        if self.panes[p].mon.is_empty() {
            let pane = &mut self.panes[p];
            if let Some(d) = pane.device.clone() {
                pane.mon.push(Mon {
                    dev: d,
                    base: pane.vram_base,
                    gtt: pane.gtt_base,
                    peak: 0.0,
                    expected: pane.expected_vram,
                    full: pane.expect_full,
                    dv: None,
                    dg: None,
                    gtt_noted: 0.0,
                });
            }
        }
        self.panes[p].metrics_prev = None;
        self.panes[p].http_busy = false;
        self.set_state(p, "Loading");
        let url = format!("http://{}:{}/v1", gs(&g, "host", "127.0.0.1"), gi(&s, "port", 0));
        self.panes[p].set_status("url", url);
        self.remember(p, true);
        self.stats_begin_load(p, &s);
        self.panes[p].k_gens = None; // KoboldCpp: total_gens seen in /api/extra/perf
        self.panes[p].k_logged = 0; // requests whose CtxLimit stats line was parsed
    }

    pub fn remember(&mut self, p: usize, running: bool) {
        let key = self.panes[p].last_key();
        let cur = self.panes[p].model_var.clone();
        let g = self.settings.g_mut();
        if !g.get("last").map(|v| v.is_object()).unwrap_or(false) {
            g.insert("last".into(), json!({}));
        }
        let last = g.get_mut("last").unwrap().as_object_mut().unwrap();
        let rel = if running {
            cur
        } else {
            last.get(&key).and_then(|v| v.get("model")).and_then(|v| v.as_str()).unwrap_or("").to_string()
        };
        if !rel.is_empty() {
            last.insert(key, json!({"model": rel, "running": running}));
            self.settings.save();
        }
    }

    /// Load once the selected model's metadata (and MTP candidates) are ready.
    pub fn request_autoload(&mut self, p: usize) {
        self.panes[p].autoload = true;
        let mp = self.model_path(p);
        if self.panes[p].info.as_ref().map(|i| Some(i.path()) == mp.as_ref()).unwrap_or(false) {
            self.schedule(Timer::AutoLoad(p), 300);
        }
    }

    pub fn do_autoload(&mut self, p: usize) {
        if !self.panes[p].autoload {
            return;
        }
        self.panes[p].autoload = false;
        if self.panes[p].proc.is_none() && self.panes[p].device.is_some() && !self.panes[p].model_var.is_empty() {
            self.load(p);
        }
    }

    pub fn stop(&mut self, p: usize, wait: bool, remember: bool) {
        let Some(pr) = &self.panes[p].proc else { return };
        let (pid, child) = (pr.pid, pr.child.clone());
        self.panes[p].user_stopped = true;
        if remember {
            self.remember(p, false);
        }
        self.set_state(p, "Stopping");
        // vLLM shuts down its workers first; Radiance releases a whole card's pools
        let grace = if self.panes[p].is_vllm() { 25 } else if self.panes[p].is_rad() { 20 } else { 8 };
        proc::stop_group(pid, child, grace, wait);
    }

    // ======================================================== monitoring ==
    pub fn poll_pane(&mut self, p: usize) {
        let mut new_lines: Vec<String> = vec![];
        // lines injected by the app itself
        while let Some(l) = self.panes[p].inject.pop_front() {
            self.panes[p].append_lines(&[l.clone()]);
            new_lines.push(l.clone());
            self.parse_line(p, &l);
        }
        for _ in 0..2000 {
            let Ok(msg) = self.panes[p].proc_rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(val) => {
                    if self.panes[p].is_vllm() && vre().poll.is_match(&val) {
                        continue;
                    }
                    if self.panes[p].is_rad() && self.r_hide_line(p, &val) {
                        continue;
                    }
                    if self.panes[p].is_kobold() && kre().progress.is_match(&val) {
                        self.k_progress(p, &val); // live progress: status line, not the log
                        continue;
                    }
                    self.panes[p].append_lines(&[val.clone()]);
                    new_lines.push(val.clone());
                    self.parse_line(p, &val);
                }
                ProcMsg::Exit(id, rc) => {
                    if !new_lines.is_empty() {
                        let nl = std::mem::take(&mut new_lines);
                        self.append_log_ui(p, &nl);
                    }
                    self.on_exit(p, id, rc);
                }
            }
        }
        loop {
            let Ok(msg) = self.panes[p].msg_rx.try_recv() else { break };
            match msg {
                PaneMsg::Http { proc_id, health, metrics } => self.on_http(p, proc_id, health, metrics),
                PaneMsg::RadStats { proc_id, stats } => {
                    if self.panes[p].proc.as_ref().map(|pr| pr.id) == Some(proc_id) {
                        self.panes[p].r_stats = Some(stats);
                    }
                }
                PaneMsg::Info(path, res) => {
                    if Some(&path) != self.model_path(p).as_ref() {
                        continue; // model changed meanwhile
                    }
                    if self.panes[p].info_done.as_ref() == Some(&path) && self.panes[p].info.is_some() {
                        continue; // already read (duplicate request)
                    }
                    self.panes[p].info_reading = None;
                    self.panes[p].info_done = Some(path.clone());
                    match res {
                        Err(e) => {
                            self.panes[p].fit_after_info = false;
                            eprintln!("llm-duo: metadata for {}: {e}", path.display());
                            self.panes[p].info = None;
                            self.panes[p].model_info = format!("Could not read metadata: {e}");
                            if self.panes[p].autoload {
                                self.schedule(Timer::AutoLoad(p), 300);
                            }
                            self.update_estimate(p);
                        }
                        Ok(info) => {
                            self.info_cache_put(&path, info.clone());
                            self.set_info(p, info);
                        }
                    }
                }
            }
        }
        if !new_lines.is_empty() {
            self.append_log_ui(p, &new_lines);
        }
        if self.panes[p].r_probe.is_some() {
            self.r_probe_poll(p);
        }
    }

    pub fn parse_line(&mut self, p: usize, line: &str) {
        if self.panes[p].is_rad() {
            self.r_parse_line(p, line);
            return;
        }
        if self.panes[p].is_vllm() {
            self.v_parse_line(p, line);
            return;
        }
        if self.panes[p].is_kobold() {
            self.k_parse_line(p, line);
            return;
        }
        self.stats_line(p, line);
        let re = lre();
        if re.eot_missing.is_match(line) {
            if let Some(info) = self.panes[p].info_g() {
                let eot = info.kv.get("tokenizer.ggml.eot_token_id").map(|v| v.display());
                let extra = self.panes[p].var_s("extra");
                if let (Some(eot), false) = (eot, extra.contains("tokenizer.ggml.eos_token_id")) {
                    let fix = format!("--override-kv tokenizer.ggml.eos_token_id=int:{eot}");
                    self.panes[p].inject.push_back(format!(
                        "[LLM Duo] WARNING: this model's end-of-turn token ({eot}) is not treated as a stop. Replies may run past their end and MTP acceptance collapse. Fix: add to Extra args:  {fix}"
                    ));
                    self.set_top_status(format!(
                        "{} pane: end-of-turn token not a stop - add to Extra args: {fix}",
                        cap_side(self.panes[p].side)
                    ));
                }
            }
        }
        if let Some(m) = re.buf.captures(line) {
            let v: f64 = m[2].parse().unwrap_or(0.0);
            self.show_llama_buffers(p, &m[1].to_string(), v);
            return;
        }
        if re.ready.is_match(line) && self.panes[p].state == "Loading" {
            self.became_ready(p);
            return;
        }
        if let Some(m) = re.accept.captures(line) {
            self.panes[p].last_accept = m[1].parse().ok();
            self.update_speed(p);
            return;
        }
        if let Some(m) = re.prompt.captures(line) {
            self.panes[p].last_prompt = Some((m[2].parse().unwrap_or(0), m[3].parse().unwrap_or(0.0)));
            self.update_speed(p);
            return;
        }
        if let Some((_, tok, tps)) = eval_match(line) {
            self.panes[p].last_gen = Some((tok, tps));
            self.update_speed(p);
            return;
        }
        if re.idle_all.is_match(line) {
            self.panes[p].busy = 0;
        } else if re.idle_one.is_match(line) {
            self.panes[p].busy = (self.panes[p].busy - 1).max(0);
        } else if re.busy.is_match(line) {
            self.panes[p].busy += 1;
        }
        if self.panes[p].state == "Ready" || self.panes[p].state == "Processing" {
            let st = if self.panes[p].busy > 0 { "Processing" } else { "Ready" };
            self.set_state(p, st);
        }
    }

    pub fn update_speed(&mut self, p: usize) {
        let pane = &mut self.panes[p];
        let mut parts = vec![];
        if let Some((n, t)) = pane.last_prompt {
            parts.push(format!("prompt {n} tok @ {t:.1} t/s"));
        }
        if let Some((n, t)) = pane.last_gen {
            parts.push(format!("gen {n} tok @ {t:.1} t/s"));
        }
        if let Some(a) = pane.last_accept {
            parts.push(format!("draft acceptance {:.0}%", if a <= 1.0 { a * 100.0 } else { a }));
        }
        pane.set_status("speed", if parts.is_empty() { "—".into() } else { parts.join("   •   ") });
    }

    pub fn tick_pane(&mut self, p: usize) {
        self.update_gpu_stats(p);
        self.check_info_read(p);
        if self.panes[p].proc.is_none() {
            return;
        }
        let Some(st) = self.panes[p].start_time else { return };
        let secs = st.elapsed().as_secs();
        let (h, rem) = (secs / 3600, secs % 3600);
        self.panes[p].set_status("uptime", format!("{}:{:02}:{:02}", h, rem / 60, rem % 60));
        // Fallback GPU memory figure when the log doesn't report buffer sizes
        if self.panes[p].buffers.is_empty() && !self.panes[p].mon.is_empty() {
            let multi = self.panes[p].mon.len() > 1;
            let parts: Vec<String> = self.panes[p]
                .mon
                .iter()
                .filter_map(|m| {
                    let used = gpu_stats(Some(&m.dev)).0?;
                    let base = m.base?;
                    let nm = if multi { format!(" {}", m.dev.short_name()) } else { String::new() };
                    Some(format!("{:.2} GiB{nm}", (used - base).max(0.0) / GIB))
                })
                .collect();
            if !parts.is_empty() {
                self.panes[p].set_status("buffers", format!("≈ {}  (VRAM increase since load)", parts.join(" + ")));
            }
        }
        self.start_http_poll(p);
        self.check_spill(p);
    }

    // ---- detect a model whose memory is not (or no longer) in VRAM ----
    pub fn check_spill(&mut self, p: usize) {
        let pane = &self.panes[p];
        if pane.is_rad() {
            // Radiance claims the whole card at startup and pins host memory on purpose
            // (token embedding, vision tower, host pool), so VRAM / GTT changes mean nothing here
            return;
        }
        if pane.proc.is_none() || pane.ready_time.map(|t| t.elapsed() < Duration::from_secs(3)).unwrap_or(true) {
            return;
        }
        if !matches!(pane.state.as_str(), "Ready" | "Processing" | "In system RAM") {
            return;
        }
        let mut reasons = vec![];
        let mut notes = vec![];
        let multi = pane.mon.len() > 1;
        let mut mons = pane.mon.clone();
        for m in mons.iter_mut() {
            let used = gpu_stats(Some(&m.dev)).0;
            let (Some(used), Some(base)) = (used, m.base) else { continue };
            let dv = used - base;
            m.dv = Some(dv);
            m.peak = m.peak.max(dv);
            let gtt = gpu_gtt(Some(&m.dev));
            let dg = match (gtt, m.gtt) {
                (Some(g), Some(b)) => Some(g - b),
                _ => None,
            };
            m.dg = dg;
            let pre = if multi { format!("{}: ", m.dev.name) } else { String::new() };
            let drop = m.peak - dv;
            // A drop only means eviction if the memory went to system RAM (GTT grew);
            // llama.cpp also frees temporary load/compute buffers, which is harmless.
            if drop > (2.0 * GIB).max(0.25 * m.peak) && dg.map(|g| g > 0.5 * drop).unwrap_or(true) {
                reasons.push(
                    format!(
                        "{pre}VRAM held by this model dropped from {:.1} to {:.1} GiB after loading",
                        m.peak / GIB,
                        dv.max(0.0) / GIB
                    ) + &dg.filter(|g| *g != 0.0).map(|g| format!(", and {:.1} GiB moved to system RAM", g / GIB)).unwrap_or_default(),
                );
            }
            if let (true, Some(g)) = (m.full, dg) {
                if g > GIB && g > m.gtt_noted + 2.0 * GIB {
                    // GTT growth with VRAM still full is usually llama-server's own pinned host
                    // buffers (RAM prompt cache, checkpoints) - not evicted weights. Note it only.
                    m.gtt_noted = g;
                    notes.push(format!(
                        "[LLM Duo] note: {pre}{:.1} GiB of system RAM (GTT) is now mapped to this GPU, but its VRAM is still full - most likely llama-server's RAM prompt cache / checkpoints, not evicted weights. If generation suddenly drops to a crawl, lower the context.",
                        g / GIB
                    ));
                }
            }
            if m.full && m.expected != 0.0 && dv < 0.5 * m.expected {
                reasons.push(format!(
                    "{pre}only {:.1} GiB of VRAM is in use, but its share of the model needs ≈ {:.1} GiB",
                    dv.max(0.0) / GIB,
                    m.expected / GIB
                ));
            }
        }
        self.panes[p].mon = mons;
        if !notes.is_empty() {
            self.append_log(p, &notes);
        }
        if !reasons.is_empty() && !self.panes[p].spilled {
            self.panes[p].spilled = true;
            if !self.panes[p].spill_reported {
                self.stats_bump(p, "spills");
                self.history_event(p, "spill", json!({}));
            }
            self.set_state(p, "In system RAM");
            if !self.panes[p].spill_reported {
                self.panes[p].spill_reported = true;
                self.report_spill(p, &reasons);
            }
        } else if reasons.is_empty() && self.panes[p].spilled {
            self.panes[p].spilled = false;
            self.set_state(p, "Ready");
            self.append_log(p, &["[LLM Duo] model memory is back in VRAM".to_string()]);
        }
    }

    fn report_spill(&mut self, p: usize, reasons: &[String]) {
        let cards: Vec<String> = self.panes[p]
            .mon
            .iter()
            .map(|m| {
                let (used, total, _) = gpu_stats(Some(&m.dev));
                format!(
                    "  {}: VRAM {:.1} / {:.1} GiB",
                    m.dev.name,
                    used.unwrap_or(0.0) / GIB,
                    total.unwrap_or(0.0) / GIB
                ) + &m.dg.map(|g| format!(", GTT gained since load {:.1} GiB", g / GIB)).unwrap_or_default()
            })
            .collect();
        let mut lines: Vec<String> = vec![
            "The model is running, but its memory is NOT (all) in GPU VRAM.".into(),
            "The GPU is reading weights from system RAM over PCIe, so it".into(),
            "will be many times slower than normal.".into(),
            String::new(),
            "What was detected:".into(),
        ];
        lines.extend(reasons.iter().map(|r| format!("  - {r}")));
        lines.push(String::new());
        lines.push("Cards now:".into());
        lines.extend(cards);
        lines.extend(
            [
                "",
                "Most likely cause: a card was filled too close to its limit, so the",
                "amdgpu driver moved the model's buffers to system RAM.",
                "The driver moves WHOLE buffers (weights are a few large ones), so the",
                "amount moved can be far bigger than the actual shortfall - usually",
                "only 1-2 GiB less context or one more CPU MoE layer is needed.",
                "",
                "Fix: Stop, lower the context (or raise Settings → Fit margin and press",
                "Fit), then Load again. Also check nothing else is using this GPU.",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        let warn: Vec<String> = reasons.iter().map(|r| format!("[LLM Duo] WARNING: {r}")).collect();
        self.append_log(p, &warn);
        self.report_failure(p, &lines.join("\n"), None, "model not in VRAM");
    }

    // ---- HTTP polling: /health for readiness, /metrics for activity & speed ----
    pub fn start_http_poll(&mut self, p: usize) {
        let Some(pr) = &self.panes[p].proc else { return };
        if self.panes[p].http_busy {
            return;
        }
        self.panes[p].http_busy = true;
        let mut host = gs(self.settings.g(), "host", "127.0.0.1");
        if host == "0.0.0.0" || host == "::" || host.is_empty() {
            host = "127.0.0.1".into();
        }
        let port = gi(&self.current_options(p), "port", 8080);
        let proc_id = pr.id;
        let (vl, kb, rad) = (self.panes[p].is_vllm(), self.panes[p].is_kobold(), self.panes[p].is_rad());
        // Radiance: /stats (decode floor, startup time) every ~10 s while it serves
        let want_stats = rad
            && matches!(self.panes[p].state.as_str(), "Ready" | "Processing")
            && self.panes[p].r_stats_t.map(|t| t.elapsed() > Duration::from_secs(10)).unwrap_or(true);
        if want_stats {
            self.panes[p].r_stats_t = Some(Instant::now());
        }
        let tx = self.panes[p].msg_tx.clone();
        let token = self.poll_token(p);
        std::thread::spawn(move || {
            let to = Duration::from_millis(1500);
            let tk = token.as_deref();
            if kb {
                // KoboldCpp: no /health or /metrics; /api/extra/perf has both
                let (health, metrics) = match super::http::get_auth(&host, port, "/api/extra/perf", to, tk) {
                    Ok(r) => (
                        Some(r.status),
                        if r.status == 200 { serde_json::from_str::<Value>(&r.body).ok().map(Metrics::Kobold) } else { None },
                    ),
                    Err(_) => (None, None),
                };
                let _ = tx.send(PaneMsg::Http { proc_id, health, metrics });
                return;
            }
            let health = super::http::get_auth(&host, port, "/health", to, tk).ok().map(|r| r.status);
            let mut metrics = None;
            if health == Some(200) {
                if let Ok(r) = super::http::get_auth(&host, port, "/metrics", to, tk) {
                    metrics = Some(if rad {
                        Metrics::Rad(super::radiance::parse_rad_metrics(&r.body))
                    } else if vl {
                        Metrics::Vllm(parse_vllm_metrics(&r.body))
                    } else {
                        let mut m = HashMap::new();
                        for c in lre().metric.captures_iter(&r.body) {
                            if let Ok(f) = c[2].parse::<f64>() {
                                m.insert(c[1].to_string(), f);
                            }
                        }
                        Metrics::Llama(m)
                    });
                }
            }
            let _ = tx.send(PaneMsg::Http { proc_id, health, metrics });
            if want_stats && health == Some(200) {
                if let Ok(r) = super::http::get_auth(&host, port, "/stats", to, tk) {
                    if let Ok(v) = serde_json::from_str::<Value>(&r.body) {
                        let _ = tx.send(PaneMsg::RadStats { proc_id, stats: v });
                    }
                }
            }
        });
    }

    pub fn on_http(&mut self, p: usize, proc_id: u64, health: Option<u16>, metrics: Option<Metrics>) {
        self.panes[p].http_busy = false;
        if self.panes[p].proc.as_ref().map(|pr| pr.id) != Some(proc_id) || self.panes[p].state == "Stopping" {
            return;
        }
        if self.panes[p].is_kobold() {
            let v = match &metrics {
                Some(Metrics::Kobold(v)) => Some(v),
                _ => None,
            };
            self.k_http(p, health, v);
            return;
        }
        if health == Some(200) && self.panes[p].state == "Loading" {
            self.became_ready(p);
        }
        let m = match metrics {
            Some(Metrics::Rad(m)) => {
                if !m.is_empty() {
                    self.r_metrics(p, m);
                }
                return;
            }
            Some(Metrics::Vllm(m)) => {
                if !m.is_empty() {
                    self.v_metrics(p, m);
                }
                return;
            }
            Some(Metrics::Llama(m)) => m,
            _ => return,
        };
        if m.is_empty() {
            return;
        }
        self.dash_metrics(p, &m);
        let g = |k: &str| m.get(k).cloned().unwrap_or(0.0);
        let processing = g("requests_processing");
        if self.panes[p].state == "Ready" || self.panes[p].state == "Processing" {
            self.set_state(p, if processing > 0.0 { "Processing" } else { "Ready" });
        }
        let mut avg = vec![];
        if g("prompt_tokens_seconds") != 0.0 {
            avg.push(format!("prompt {:.1} t/s", g("prompt_tokens_seconds")));
        }
        if g("predicted_tokens_seconds") != 0.0 {
            avg.push(format!("gen {:.1} t/s", g("predicted_tokens_seconds")));
        }
        if m.contains_key("tokens_predicted_total") {
            avg.push(format!(
                "{} prompt / {} generated tokens total",
                fmt_thousands(g("prompt_tokens_total") as i64),
                fmt_thousands(g("tokens_predicted_total") as i64)
            ));
        }
        self.panes[p].set_status("avg", if avg.is_empty() { "—".into() } else { avg.join("   •   ") });
        // "Last request" from counter deltas, measured between idle snapshots
        if processing == 0.0 {
            if let Some(prev) = self.panes[p].metrics_prev.clone() {
                let d = |k: &str| g(k) - prev.get(k).cloned().unwrap_or(0.0);
                let (dp, dps) = (d("prompt_tokens_total"), d("prompt_seconds_total"));
                let (dg, dgs) = (d("tokens_predicted_total"), d("tokens_predicted_seconds_total"));
                if dp > 0.0 && dps > 0.0 {
                    self.panes[p].last_prompt = Some((dp as i64, dp / dps));
                }
                if dg > 0.0 && dgs > 0.0 {
                    self.panes[p].last_gen = Some((dg as i64, dg / dgs));
                }
                if dp > 0.0 || dg > 0.0 {
                    self.update_speed(p);
                }
            }
            self.panes[p].metrics_prev = Some(m);
        }
    }

    pub fn on_exit(&mut self, p: usize, id: u64, rc: i32) {
        if self.panes[p].proc.as_ref().map(|pr| pr.id) != Some(id) {
            return;
        }
        self.panes[p].proc = None;
        self.panes[p].spilled = false;
        self.panes[p].web_enabled = false;
        self.cancel_timer(&Timer::Calib(p));
        // dashboard / history bookkeeping
        let now = super::telemetry::unix_now();
        let ready = self.panes[p].dash_ready_t;
        let run_secs = ready.map(|t| now - t);
        if let Some(s) = run_secs {
            self.stats_add_f(p, "run_secs", s);
        }
        self.panes[p].dash_active = None;
        self.panes[p].dash_progress = None;
        let user = self.panes[p].user_stopped;
        let kind = match (user, ready.is_some()) {
            (true, true) => "stop",
            (true, false) => "load_aborted",
            (false, true) => "crash",
            (false, false) => "load_failed",
        };
        if matches!(kind, "load_aborted" | "load_failed") {
            self.stats_bump(p, kind);
        }
        self.history_event(p, kind, json!({"rc": rc, "runSecs": run_secs.map(|s| (s * 10.0).round() / 10.0)}));
        if self.panes[p].user_stopped {
            self.set_state(p, "Stopped");
            self.append_log(p, &[format!("[LLM Duo] server stopped (exit code {rc})")]);
            return;
        }
        self.set_state(p, if rc != 0 { "Crashed" } else { "Exited" });
        self.stats_bump(p, "crashes");
        self.remember(p, false);
        self.append_log(p, &[format!("[LLM Duo] server exited unexpectedly (exit code {rc})")]);
        let mut headline = format!("{} exited unexpectedly with code {rc}", self.panes[p].server_name());
        if self.panes[p].is_vllm() || self.panes[p].is_kobold() || self.panes[p].is_rad() {
            let lines: Vec<String> = {
                let l = &self.panes[p].lines;
                l.iter().skip(l.len().saturating_sub(600)).cloned().collect()
            };
            let hint = if self.panes[p].is_rad() {
                super::radiance::rad_failure_hint(&lines)
            } else if self.panes[p].is_vllm() {
                vllm_failure_hint(&lines)
            } else {
                kobold_failure_hint(&lines)
            };
            if !hint.is_empty() {
                headline += &format!("\n\nLikely cause: {hint}");
            }
        }
        self.report_failure(p, &headline, Some(rc), "GPU error");
    }

    pub fn report_failure(&mut self, p: usize, headline: &str, rc: Option<i32>, title_suffix: &str) {
        let pane = &self.panes[p];
        let (dname, did) = pane.device.as_ref().map(|d| (d.name.clone(), d.id.clone())).unwrap_or(("?".into(), "?".into()));
        let mut header = vec![
            format!("{APP_NAME} error report"),
            format!("Time:      {}", now_iso()),
            format!(
                "Pane:      {}   Backend: {}{}",
                pane.side,
                pane.backend,
                if pane.is_vllm() || pane.is_rad() { String::new() } else { format!(" ({})", self.kbuild(p)) }
            ),
            format!("GPU:       {dname} [{did}]"),
            format!("Model:     {}", self.model_path(p).map(|x| x.display().to_string()).unwrap_or("None".into())),
            format!("Exit code: {}", rc.map(|r| r.to_string()).unwrap_or("None".into())),
            format!("Env:       {}", env_text(&pane.env_extra)),
            format!("Command:   {}", if pane.cmd.is_empty() { "(none)".into() } else { shlex_join(&mask_cmd(&pane.cmd)) }),
            String::new(),
            headline.to_string(),
            String::new(),
            "---- last server output ----".into(),
        ];
        header.extend(pane.lines.iter().skip(pane.lines.len().saturating_sub(400)).cloned());
        let mut text = header.join("\n");
        let stem = Path::new(if pane.model_var.is_empty() { "model" } else { &pane.model_var })
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = format!("{}_{stem}", pane.side);
        let path = match write_log(&name, &text) {
            Ok(p) => Some(p),
            Err(e) => {
                text += &format!("\n\n[could not write log: {e}]");
                None
            }
        };
        // never pop a window on its own (it would steal focus from other apps):
        // flag it in the title bar + a button the user can click when ready
        let side = pane.side;
        self.raise_alert(
            &format!("{APP_NAME}: {side} {title_suffix}"),
            &text,
            path,
            &format!("{} pane: {title_suffix}", cap_side(side)),
        );
    }

    pub fn became_ready(&mut self, p: usize) {
        self.panes[p].ready_time = Some(Instant::now());
        let now = super::telemetry::unix_now();
        self.panes[p].dash_ready_t = Some(now);
        let secs = self.panes[p].dash_load_t.map(|t| ((now - t) * 10.0).round() / 10.0);
        self.history_event(p, "ready", json!({"loadSecs": secs}));
        self.set_state(p, "Ready");
        self.panes[p].web_enabled = true;
        self.schedule(Timer::Calib(p), 4000);
    }

    pub fn record_calibration(&mut self, p: usize) {
        if self.panes[p].is_rad() {
            self.r_record_calibration(p);
            return;
        }
        if self.panes[p].is_vllm() {
            self.v_record_calibration(p);
            return;
        }
        if self.panes[p].proc.is_none() || !matches!(self.panes[p].state.as_str(), "Ready" | "Processing") {
            return;
        }
        self.check_spill(p);
        if self.panes[p].spilled {
            return; // never calibrate from a load that isn't actually in VRAM
        }
        let (Some(base), Some((rel, opts, mm))) = (self.panes[p].vram_base, self.panes[p].loaded.clone()) else { return };
        let Some(used) = gpu_stats(self.panes[p].device.as_ref()).0 else { return };
        let delta = used - base;
        if delta < 512.0 * MIB {
            return;
        }
        if self.dual_active(p) && self.panes[p].last_plan.is_some() {
            self.record_dual_calibration(p, &rel, &opts);
            return;
        }
        if gi(&opts, "ngl", 0) < 0 {
            return; // auto layers: real layer split unknown, so the measurement can't calibrate
        }
        let name = self.panes[p].device.as_ref().unwrap().name.clone();
        let key = self.calib_key(p, &name);
        let mut rec = json!({"params": opts, "mm": jnum(mm), "measured": jnum(delta), "when": now_iso()});
        // no signature when the metadata isn't known: the lookup fills it in from the file later
        if let Some(i) = self.panes[p].info.as_ref() {
            rec["sig"] = json!(i.sig());
        }
        self.settings.set_calib(&key, &rel, rec.as_object().unwrap().clone());
        self.panes[p].borrow_cache.clear();
        if self.panes[p].model_var == rel {
            self.update_estimate(p);
        }
    }

    fn record_dual_calibration(&mut self, p: usize, rel: &str, opts: &Opts) {
        let plan = self.panes[p].last_plan.clone().unwrap();
        if !(plan.mode == "layer" || plan.mode == "none") || gi(opts, "n_cpu_moe", 0) < 0 {
            return;
        }
        let mut cards = serde_json::Map::new();
        for m in &self.panes[p].mon {
            let (Some(used), Some(base)) = (gpu_stats(Some(&m.dev)).0, m.base) else { return };
            cards.insert(m.dev.id.clone(), jnum(used - base));
        }
        let mtp_b = self.panes[p].mtp_bytes(opts);
        let side = [self.mmproj_bytes(p, opts) - mtp_b, mtp_b];
        let mut rec = json!({"params": opts, "cards": cards, "side": [jnum(side[0]), jnum(side[1])], "mode": plan.mode,
                             "when": now_iso()});
        if let Some(i) = self.panes[p].info.as_ref() {
            rec["sig"] = json!(i.sig());
        }
        self.panes[p].borrow_cache.clear();
        if plan.mode == "none" {
            rec["order"] = json!([plan.single.as_ref().map(|d| d.id.clone()).unwrap_or_default()]);
        } else {
            rec["order"] = json!(plan.order.iter().map(|d| d.id.clone()).collect::<Vec<_>>());
            rec["k"] = json!(plan.k);
        }
        let base = self.dev_base(p);
        let key = self.calib_key(p, &base);
        self.settings.set_calib(&key, rel, rec.as_object().unwrap().clone());
        self.update_estimate(p);
    }

    // ============================================================= helpers ==
    pub fn set_state(&mut self, p: usize, st: &str) {
        let prev = self.panes[p].state.clone();
        if st == "Processing" && prev != "Processing" {
            self.panes[p].req_start = Some(Instant::now());
        } else if st == "Ready" && prev == "Processing" {
            // debounce: chime only if it is still idle a moment later
            self.schedule(Timer::Chime(p), 1200);
        }
        let st = if self.panes[p].spilled && (st == "Ready" || st == "Processing") { "In system RAM" } else { st };
        self.panes[p].state = st.to_string();
    }

    pub fn maybe_chime(&mut self, p: usize) {
        let g = self.settings.g();
        if !gb(g, "chime", false) || self.panes[p].state != "Ready" {
            return;
        }
        let Some(start) = self.panes[p].req_start.take() else { return };
        let took = start.elapsed().as_secs_f64() - 1.2;
        if took >= gf(g, "chime_min_secs", 5.0) {
            proc::play_chime();
        }
    }

    pub fn clear_log(&mut self, p: usize) {
        self.push_event(json!({"type": "logClear", "side": self.panes[p].side}));
        for k in ["speed", "avg", "buffers", "uptime", "cache"] {
            self.panes[p].set_status(k, "—");
        }
    }

    /// Add app-generated lines to the log view (like Python's _append_log: not kept in `lines`).
    pub fn append_log(&mut self, p: usize, lines: &[String]) {
        self.append_log_ui(p, lines);
    }

    pub fn append_log_ui(&mut self, p: usize, lines: &[String]) {
        self.push_event(json!({"type": "log", "side": self.panes[p].side, "text": lines.join("\n")}));
    }

    pub fn log_text(&self, p: usize) -> String {
        let pane = &self.panes[p];
        let dev = if self.dual_active(p) {
            self.dual_devices().iter().map(|d| d.name.clone()).collect::<Vec<_>>().join(" + ")
        } else {
            pane.device.as_ref().map(|d| d.name.clone()).unwrap_or("?".into())
        };
        let st = |k: &str| pane.status.get(k).cloned().unwrap_or_default();
        let mut head = vec![
            format!("{APP_NAME} server log"),
            format!("Time:      {}", now_iso()),
            format!(
                "Pane:      {}   GPU(s): {dev}   Backend: {}{}",
                pane.side,
                pane.backend,
                if pane.is_vllm() || pane.is_rad() { String::new() } else { format!(" ({})", self.kbuild(p)) }
            ),
            format!("Model:     {}", self.model_path(p).map(|x| x.display().to_string()).unwrap_or("None".into())),
            format!("Env:       {}", env_text(&pane.env_extra)),
            format!("Command:   {}", if pane.cmd.is_empty() { "(not started)".into() } else { shlex_join(&mask_cmd(&pane.cmd)) }),
            format!("State:     {}", pane.state),
            format!("Last req:  {}", st("speed")),
            format!("GPU mem:   {}", st("buffers")),
            format!("Setup:     {}", st("life")),
            String::new(),
            "---- estimate at time of copy ----".into(),
            pane.est_text.clone(),
        ];
        if self.dual_active(p) {
            head.push(String::new());
            head.push(pane.dual_info.clone());
        }
        head.push(String::new());
        head.push("---- server output ----".into());
        head.extend(pane.lines.iter().cloned());
        head.join("\n")
    }

    pub fn copy_log(&mut self, p: usize) {
        let t = self.log_text(p);
        self.clipboard(&t);
    }

    pub fn save_log(&mut self, p: usize) {
        let stem = Path::new(if self.panes[p].model_var.is_empty() { "model" } else { &self.panes[p].model_var })
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = format!("{}_{stem}_manual", self.panes[p].side);
        match write_log(&name, &self.log_text(p)) {
            Ok(path) => self.set_top_status(format!("Log saved: {}", path.display())),
            Err(e) => self.msg("error", &format!("Could not save log:\n{e}")),
        }
    }

    pub fn open_web(&mut self, p: usize) {
        let port = gi(&self.current_options(p), "port", 8080);
        let mut host = gs(self.settings.g(), "host", "127.0.0.1");
        if host == "0.0.0.0" || host == "::" || host.is_empty() {
            host = "127.0.0.1".into();
        }
        // Radiance serves its chat page at / in recent builds; the dashboard has its own path
        let path = if self.panes[p].is_vllm() {
            "/docs"
        } else if self.panes[p].is_rad() {
            "/dashboard"
        } else {
            ""
        };
        open_url(&format!("http://{host}:{port}{path}"));
    }

    // ============================================================ backend ==
    pub fn apply_backend_ui(&mut self, p: usize) {
        self.panes[p].k_last_build = None;
        if self.panes[p].is_vllm() {
            self.panes[p].badge = ("vLLM (Experimental)".into(), EXPERIMENTAL_FG.into());
        } else if self.panes[p].is_rad() {
            self.panes[p].badge = ("Radiance".into(), RADIANCE_FG.into());
        } else {
            self.k_refresh_row(p);
        }
        self.panes[p].dual_mode_values = if self.panes[p].is_rad() {
            vec!["tensor".into()]
        } else if self.panes[p].is_vllm() {
            VLLM_SPLIT_MODES.iter().map(|s| s.to_string()).collect()
        } else {
            self.split_modes_for(p)
        };
    }

    /// Switch this pane's backend (from the Settings dialog).
    pub fn set_backend(&mut self, p: usize, new: &str) {
        if new == self.panes[p].backend || !BACKENDS.contains(&new) || self.panes[p].proc.is_some() {
            return;
        }
        if self.has_timer(&Timer::Save(p)) {
            self.save_options(p);
        }
        let rel = self.panes[p].model_var.clone();
        if !rel.is_empty() {
            let key = self.panes[p].last_key();
            let g = self.settings.g_mut();
            if !g.get("last").map(|v| v.is_object()).unwrap_or(false) {
                g.insert("last".into(), json!({}));
            }
            g.get_mut("last").unwrap().as_object_mut().unwrap().insert(key, json!({"model": rel, "running": false}));
        }
        self.panes[p].backend = new.to_string();
        let side = self.panes[p].side;
        self.settings.g_mut().insert(format!("{side}_backend"), Value::from(new));
        self.settings.save();
        self.apply_backend_ui(p);
        self.panes[p].info = None;
        self.panes[p].model_var.clear();
        let models = self.models_for(new);
        self.set_models(p, models);
        let lk = self.panes[p].last_key();
        let mut last = self.settings.g().get("last").and_then(|v| v.get(&lk)).cloned();
        if last.is_none() && !rel.is_empty() && self.panes[p].models.contains(&rel) {
            // llama.cpp <-> KoboldCpp: same GGUF files
            last = Some(json!({"model": rel}));
        }
        let lm = last.as_ref().and_then(|l| l.get("model")).and_then(|v| v.as_str()).map(String::from);
        match lm {
            Some(m) if self.panes[p].models.contains(&m) => {
                self.panes[p].model_var = m;
                self.on_model_selected(p);
            }
            _ => {
                self.panes[p].model_info = if self.panes[p].models.is_empty() { self.no_models_text(p) } else { String::new() };
                self.update_estimate(p);
            }
        }
        self.refresh_dual_ui(p);
        self.refresh_life(p);
        for k in ["url", "speed", "avg", "buffers", "cache"] {
            self.panes[p].set_status(k, "—");
        }
        self.set_top_status(format!("{} pane: backend {}", cap_side(side), backend_label(new)));
    }

    /// Grey out everything except the GPU header (right pane in dual mode).
    pub fn set_pane_disabled(&mut self, p: usize, disabled: bool, note: &str) {
        if disabled && !self.panes[p].pane_disabled {
            self.panes[p].pane_disabled = true;
            self.panes[p].dual_note = note.to_string();
        } else if !disabled && self.panes[p].pane_disabled {
            self.panes[p].pane_disabled = false;
            self.panes[p].dual_note.clear();
        }
    }
}
