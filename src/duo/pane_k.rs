//! KoboldCpp side of a pane (the Python `KoboldPaneMixin`): llama.cpp's options, estimates and
//! dual planner, with KoboldCpp's CLI and log.

use serde_json::{json, Value};
use std::path::Path;

use super::app::App;
use super::devices::Device;
use super::kobold::{kre, lre, snap_batch};
use super::mtp::make_relaxed_template;
use super::util::*;

impl App {
    /// Build-dependent parts of the Build row, the pane badge and the split modes.
    pub fn k_refresh_row(&mut self, p: usize) {
        if !(self.panes[p].is_kobold() || self.panes[p].is_llama()) {
            return;
        }
        let b = self.kbuild(p);
        let exe = self.k_bin(p, Some(&b));
        let kob = self.panes[p].is_kobold();
        let where_ = if kob { "KoboldCpp" } else { "llama-server" };
        let (txt, col) = if exe.is_empty() {
            (format!("⚠ no {b} binary set - ⚙ Settings → {where_}"), "#d0342c")
        } else if !Path::new(&exe).is_file() {
            (format!("⚠ {b} binary not found: {exe}"), "#d0342c")
        } else if kob && self.kobold_flags(&b).is_none() {
            self.fetch_kobold_flags(&b, None);
            (format!("{}  (reading its options…)", file_name(&exe)), "#666")
        } else {
            (if kob { file_name(&exe) } else { exe.clone() }, "#666")
        };
        self.panes[p].k_note = (txt, col.into());
        self.panes[p].badge = (format!("{} · {b}", if kob { "KoboldCpp" } else { "llama.cpp" }), "#2a7fd4".into());
        if self.panes[p].k_last_build.as_deref() != Some(b.as_str()) {
            self.panes[p].k_last_build = Some(b.clone());
            let modes = self.split_modes_for(p);
            self.panes[p].dual_mode_values = modes.clone();
            if !modes.contains(&self.panes[p].var_s("dual_mode")) {
                self.set_var_notify(p, "dual_mode", Value::from("layer"));
            }
            let mp = self.model_path(p);
            if let Some(info) = self.panes[p].info_g() {
                if Some(&info.path) == mp.as_ref() {
                    // not a previous model's info
                    let c = info.mtp_candidates.clone();
                    self.apply_mtp_candidates(p, c);
                }
            }
        }
    }

    /// KoboldCpp-specific MTP notes; returns false to fall through to the shared file notes.
    pub fn k_mtp_note(&mut self, p: usize) -> bool {
        let has_head = self.panes[p].info_g().map(|i| i.nextn > 0).unwrap_or(false);
        let sel = self.panes[p].mtp_sel();
        if sel == "(built-in)" {
            self.panes[p].mtp_note = "model's own MTP head (--usemtp)  •  adds its layer + its KV cache".into();
            return true;
        }
        if self.panes[p].mtp_cands.is_empty() {
            let note = if has_head && !self.supports(p, "--usemtp") {
                "this KoboldCpp build is too old for MTP (needs 1.115+)".to_string()
            } else if has_head {
                "MTP off  •  this model has a built-in MTP head (--usemtp)".into()
            } else {
                String::new()
            };
            self.panes[p].mtp_note = note;
            return true;
        }
        false
    }

    pub fn vk_env(&mut self, p: usize, devs: &[Device]) -> Vec<(String, String)> {
        let mut idx = vec![];
        for d in devs {
            let i = match self.vulkan_index(d) {
                Some(i) => i,
                None => {
                    let note = format!(
                        "[LLM Duo] WARNING: no Vulkan device matched {} ({}); guessing Vulkan index {}",
                        d.name,
                        if self.vk_error.is_empty() { "unknown".to_string() } else { self.vk_error.clone() },
                        d.index
                    );
                    self.panes[p].k_env_notes.push(note);
                    d.index
                }
            };
            idx.push(i.to_string());
        }
        vec![("GGML_VK_VISIBLE_DEVICES".into(), idx.join(","))]
    }

    pub fn k_build_command(&mut self, p: usize) -> Result<(Vec<String>, Opts), String> {
        let s = self.current_options(p);
        let path = self.model_path(p).ok_or("no model selected")?;
        let g = self.settings.g().clone();
        let build = gs(&s, "kbuild", "ROCm");
        let exe = self.k_bin(p, Some(&build));
        let mut cmd: Vec<String> = vec![
            exe,
            "--model".into(),
            path.to_string_lossy().into_owned(),
            "--skiplauncher".into(),
            "--contextsize".into(),
            gi(&s, "ctx", 8192).clamp(256, 524288).to_string(),
            "--port".into(),
            gi(&s, "port", 5001).to_string(),
        ];
        let host = gs(&g, "host", "");
        if !host.is_empty() {
            cmd.extend(["--host".into(), host]);
        }
        let mut plan = None;
        if self.dual_active(p) {
            plan = self.dual_plan(p, &s);
            self.panes[p].last_plan = plan.clone();
        }
        let row = plan.as_ref().map(|pl| pl.mode == "row").unwrap_or(false);
        if build == "ROCm" {
            cmd.push("--usecuda".into());
            if row {
                cmd.push("rowsplit".into());
            }
        } else {
            cmd.push("--usevulkan".into());
        }
        if let Some(pl) = &plan {
            cmd.extend(["--gpulayers".into(), "999".into()]);
            if pl.mode != "none" {
                cmd.push("--tensor_split".into());
                let ts = pl.ts.clone().unwrap_or_else(|| "1,1".into());
                cmd.extend(ts.split(',').map(|x| x.trim().to_string()));
                if build == "ROCm" && self.supports(p, "--device") {
                    cmd.extend(["--device".into(), pl.order.iter().map(|d| d.id.clone()).collect::<Vec<_>>().join(",")]);
                }
                cmd.extend(["--maingpu".into(), "1".into()]);
            }
        } else {
            cmd.extend(["--gpulayers".into(), gi(&s, "ngl", 999).to_string()]); // -1 = KoboldCpp's own autofit
        }
        if gs(&s, "fa", "on") == "off" && self.supports(p, "--noflashattention") {
            cmd.push("--noflashattention".into());
        }
        let kv = gs(&s, "kv", "f16");
        if kv != "f16" {
            cmd.extend(["--quantkv".into(), kv]);
        }
        let b = snap_batch(gi(&s, "batch", 2048));
        cmd.extend(["--batchsize".into(), b.to_string()]);
        if self.supports(p, "--ubatchsize") {
            cmd.extend(["--ubatchsize".into(), snap_batch(gi(&s, "ubatch", 512).min(b)).to_string()]);
        }
        if gi(&s, "threads", 0) > 0 {
            cmd.extend(["--threads".into(), gi(&s, "threads", 0).to_string()]);
        }
        let ncm = gi(&s, "n_cpu_moe", 0);
        if let Some(cs) = plan.as_ref().and_then(|pl| pl.cpu_set.clone()).filter(|c| !c.is_empty()) {
            let layers = cs.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("|");
            cmd.extend(["--overridetensors".into(), format!("blk\\.({layers})\\.ffn_(up|down|gate|gate_up)_exps\\.=CPU")]);
        } else if ncm > 0 {
            cmd.extend(["--moecpu".into(), ncm.to_string()]);
        }
        if gb(&s, "jinja", true) {
            cmd.push("--jinja".into());
            if gb(&s, "relax_tpl", false) && self.supports(p, "--jinjatemplate") {
                let tpl = self.panes[p].info.as_ref().and_then(|i| i.chat_template());
                let (tpl_path, _n) = make_relaxed_template(tpl, &path)?;
                cmd.extend(["--jinjatemplate".into(), tpl_path.to_string_lossy().into_owned()]);
            }
        }
        let mm = gs(&s, "mmproj", "(none)");
        if mm != "(none)" {
            cmd.extend(["--mmproj".into(), path.parent().unwrap().join(&mm).to_string_lossy().into_owned()]);
        }
        let mtp = gs(&s, "mtp", "(off)");
        if mtp == "(built-in)" && self.supports(p, "--usemtp") {
            cmd.extend(["--usemtp".into(), "--draftamount".into(), gi(&s, "mtp_n", 4).to_string()]);
        } else if !mtp.is_empty() && mtp != "(off)" && mtp != "(built-in)" && self.supports(p, "--draftmodel") {
            // MTP drafter / Gemma assistant file (KoboldCpp 1.115+ detects the MTP kind itself)
            cmd.extend([
                "--draftmodel".into(),
                path.parent().unwrap().join(&mtp).to_string_lossy().into_owned(),
                "--draftamount".into(),
                gi(&s, "mtp_n", 4).to_string(),
            ]);
        }
        if gb(&s, "kquiet", true) {
            cmd.push("--quiet".into());
        }
        let extra = gs(&s, "extra", "");
        if !extra.trim().is_empty() {
            cmd.extend(shlex_split(&extra)?);
        }
        Ok((cmd, s))
    }

    pub fn k_env(&mut self, p: usize) -> Vec<(String, String)> {
        let build = self.kbuild(p);
        let plan = if self.dual_active(p) { self.panes[p].last_plan.clone() } else { None };
        let mut devs: Vec<Device> = match &plan {
            Some(pl) if pl.mode == "none" => pl.single.clone().into_iter().collect(),
            Some(pl) => pl.order.clone(),
            None => self.panes[p].device.clone().into_iter().collect(),
        };
        self.panes[p].k_env_notes.clear();
        if build == "ROCm" {
            if plan.as_ref().map(|pl| pl.mode != "none").unwrap_or(false) && self.supports(p, "--device") {
                devs.sort_by_key(|d| d.index); // --device sets the order
            }
            return vec![(
                "HIP_VISIBLE_DEVICES".into(),
                devs.iter().map(|d| d.index.to_string()).collect::<Vec<_>>().join(","),
            )];
        }
        self.vk_env(p, &devs)
    }

    // ---------------- log + HTTP ---------------- //
    pub fn k_progress(&mut self, p: usize, line: &str) {
        let Some(m) = kre().progress.captures(line) else { return };
        let txt = if let (Some(a), Some(b)) = (m.get(1), m.get(2)) {
            format!(
                "reading prompt {} / {} tokens…",
                fmt_thousands(a.as_str().parse().unwrap_or(0)),
                fmt_thousands(b.as_str().parse().unwrap_or(0))
            )
        } else {
            format!(
                "generating {} / {} tokens…",
                fmt_thousands(m[3].parse().unwrap_or(0)),
                fmt_thousands(m[4].parse().unwrap_or(0))
            )
        };
        self.panes[p].set_status("speed", txt);
        if self.panes[p].state == "Ready" {
            self.set_state(p, "Processing");
        }
    }

    pub fn show_llama_buffers(&mut self, p: usize, kind: &str, mib: f64) {
        let pane = &mut self.panes[p];
        if let Some(e) = pane.buffers.iter_mut().find(|e| e.0 == kind) {
            e.1 += mib;
        } else {
            pane.buffers.push((kind.to_string(), mib));
        }
        let tot: f64 = pane.buffers.iter().map(|e| e.1).sum();
        let parts = pane.buffers.iter().map(|(k, v)| format!("{k} {:.2}", v / 1024.0)).collect::<Vec<_>>().join("  ");
        pane.set_status("buffers", format!("{:.2} GiB   ({parts} GiB)", tot / 1024.0));
    }

    pub fn k_parse_line(&mut self, p: usize, line: &str) {
        if let Some(m) = lre().buf.captures(line) {
            let v: f64 = m[2].parse().unwrap_or(0.0);
            self.show_llama_buffers(p, &m[1].to_string(), v);
            return;
        }
        let k = kre();
        if k.ready.is_match(line) && self.panes[p].state == "Loading" {
            self.became_ready(p);
            return;
        }
        if let Some(m) = k.done.captures(line) {
            self.panes[p].k_logged += 1;
            let used: i64 = m[1].parse().unwrap_or(0);
            let n_proc: i64 = m[3].parse().unwrap_or(0);
            let p_s: f64 = m[4].parse().unwrap_or(0.0);
            let p_tps: f64 = m[5].parse().unwrap_or(0.0);
            let n_gen: i64 = m[6].parse().unwrap_or(0);
            let g_s: f64 = m[8].parse().unwrap_or(0.0);
            let g_tps: f64 = m[9].parse().unwrap_or(0.0);
            {
                let pane = &mut self.panes[p];
                if n_proc != 0 {
                    pane.last_prompt = Some((n_proc, p_tps));
                }
                pane.last_gen = Some((n_gen, g_tps));
            }
            self.update_speed(p);
            let pane = &mut self.panes[p];
            let t = &mut pane.k_sess;
            t.0 += n_proc;
            t.1 += p_s;
            t.2 += n_gen;
            t.3 += g_s;
            t.4 += 1;
            let t = *t;
            let mut avg = vec![];
            if t.1 > 0.0 && t.0 != 0 {
                avg.push(format!("prompt {:.1} t/s", t.0 as f64 / t.1));
            }
            if t.3 > 0.0 {
                avg.push(format!("gen {:.1} t/s", t.2 as f64 / t.3));
            }
            avg.push(format!(
                "{} request(s), {} prompt / {} generated tokens since load",
                t.4,
                fmt_thousands(t.0),
                fmt_thousands(t.2)
            ));
            pane.set_status("avg", avg.join("   •   "));
            let mut r = serde_json::Map::new();
            r.insert("p_ms".into(), json!(p_s * 1000.0));
            r.insert("p_tok".into(), json!(n_proc));
            r.insert("g_ms".into(), json!(g_s * 1000.0));
            r.insert("g_tok".into(), json!(n_gen));
            r.insert("n_end".into(), json!(used));
            self.stats_record(p, &r);
            if self.panes[p].state == "Processing" {
                self.set_state(p, "Ready");
            }
            return;
        }
        if let Some(m) = k.draft.captures(line) {
            let ok: i64 = m[1].parse().unwrap_or(0);
            let bad: i64 = m[2].parse().unwrap_or(0);
            if ok + bad != 0 {
                self.panes[p].last_accept = Some(ok as f64 / (ok + bad) as f64);
                self.update_speed(p);
                if let Some(ids) = self.panes[p].stats_ids_loaded.clone() {
                    if let Some(mut st) = self.settings.get_stats(&ids.0, &ids.1) {
                        st.insert("draft_acc".into(), (gi(&st, "draft_acc", 0) + ok).into());
                        st.insert("draft_gen".into(), (gi(&st, "draft_gen", 0) + ok + bad).into());
                        self.settings.set_stats(&ids.0, &ids.1, st);
                    }
                }
            }
        }
    }

    pub fn k_after_flags(&mut self, p: usize, ok: bool, err: &str) {
        self.panes[p].k_waiting = false;
        self.k_refresh_row(p);
        if ok {
            self.load(p);
        } else {
            self.msg("error", &format!("Could not read KoboldCpp's options:\n\n{err}"));
        }
    }

    pub fn k_http(&mut self, p: usize, health: Option<u16>, m: Option<&Value>) {
        if health == Some(200) && self.panes[p].state == "Loading" {
            self.became_ready(p);
        }
        let Some(m) = m.and_then(|v| v.as_object()) else { return };
        let busy = gi(m, "idle", 1) == 0 || gi(m, "queue", 0) > 0;
        if self.panes[p].state == "Ready" || self.panes[p].state == "Processing" {
            self.set_state(p, if busy { "Processing" } else { "Ready" });
        }
        // Backup for the speed stats if a request finished without its CtxLimit line in the log
        let Some(gens) = m.get("total_gens").and_then(|v| v.as_i64()) else { return };
        let pane = &mut self.panes[p];
        let Some(prev) = pane.k_gens else {
            pane.k_gens = Some(gens);
            pane.k_logged = pane.k_logged.max(gens);
            return;
        };
        if gens > prev {
            pane.k_gens = Some(gens);
            pane.k_lag = 0;
        } else if gens > pane.k_logged && !busy {
            pane.k_lag += 1;
            if pane.k_lag >= 2 {
                // still no log line two polls later: use perf
                pane.k_logged = gens;
                let n_in = gi(m, "last_input_count", 0);
                let n_out = gi(m, "last_token_count", 0);
                let t_pp = gf(m, "last_process_time", 0.0);
                let t_gen = gf(m, "last_eval_time", 0.0);
                if n_in != 0 {
                    pane.last_prompt = Some((n_in, gf(m, "last_process_speed", 0.0)));
                }
                pane.last_gen = Some((n_out, gf(m, "last_eval_speed", 0.0)));
                self.update_speed(p);
                let mut r = serde_json::Map::new();
                r.insert("p_ms".into(), json!(t_pp * 1000.0));
                r.insert("p_tok".into(), json!(n_in));
                r.insert("g_ms".into(), json!(t_gen * 1000.0));
                r.insert("g_tok".into(), json!(n_out));
                self.stats_record(p, &r);
            }
        }
    }
}

pub fn file_name(p: &str) -> String {
    Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.to_string())
}
