//! Everything a pane does differently when its backend is vLLM (the Python `VllmPaneMixin`).

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

use super::app::App;
use super::consts::*;
use super::devices::{gpu_gtt, gpu_stats, Device};
use super::estimate::{pcie_link, Manual};
use super::mtp::make_relaxed_template;
use super::pane::{cap_side, calib_clamp, Mon};
use super::util::*;
use super::vllm::*;

impl App {
    pub fn v_apply_info(&mut self, p: usize, info: Arc<VllmModelInfo>) {
        let mut vals = vec!["(off)".to_string()];
        if info.nextn != 0 {
            vals.push("(built-in)".into());
        }
        let mut changed = false;
        {
            let pane = &mut self.panes[p];
            pane.v_mtp_values = vals.clone();
            pane.v_mtp_enabled = info.nextn != 0;
            pane.v_text_enabled = info.multimodal;
            pane.v_quant_enabled = info.quant.is_empty();
            pane.populating = true;
            if !vals.contains(&pane.var_s("mtp")) {
                pane.put_var("mtp", &Value::from("(off)"));
            }
            if pane.v_reason_saved == Some(None) {
                // never configured: pick by family
                let a = info.arch.to_lowercase();
                let pick = if a.contains("qwen3") {
                    "qwen3"
                } else if a.contains("glm4moe") || a.contains("glm4_moe") {
                    "glm45"
                } else if a.contains("deepseek") && info.path.to_string_lossy().to_lowercase().contains("r1") {
                    "deepseek_r1"
                } else {
                    "(none)"
                };
                pane.put_var("reasoning", &Value::from(pick));
                pane.v_reason_saved = Some(Some(pick.to_string()));
                changed = pick != "(none)";
            }
            pane.populating = false;
        }
        self.v_mtp_note(p);
        if changed {
            self.on_option_change(p);
        }
    }

    pub fn v_mtp_note(&mut self, p: usize) {
        let pane = &mut self.panes[p];
        let Some(i) = pane.info_v() else {
            pane.v_mtp_note.clear();
            return;
        };
        pane.v_mtp_note = if i.nextn == 0 {
            "no MTP layers declared in config.json".into()
        } else if pane.var_s("mtp") == "(built-in)" {
            format!("model's own MTP head: {} layer(s), {:.2} GiB of weights + its KV cache", i.nextn, i.mtp_bytes / GIB)
        } else {
            format!("MTP off  •  this model has a built-in MTP head ({} layer(s))", i.nextn)
        };
    }

    pub fn v_devs(&self, p: usize) -> Vec<Device> {
        if self.dual_active(p) {
            return self.dual_devices();
        }
        self.panes[p].device.clone().into_iter().collect()
    }

    pub fn v_total(d: &Device) -> f64 {
        gpu_stats(Some(d)).1.filter(|t| *t != 0.0).unwrap_or(d.total_mib as f64 * MIB)
    }

    /// Free VRAM on a card, not counting this pane's own running model.
    pub fn v_free(&self, p: usize, d: &Device) -> f64 {
        let (used, total, _) = gpu_stats(Some(d));
        let total = total.filter(|t| *t != 0.0).unwrap_or(d.total_mib as f64 * MIB);
        let Some(mut used) = used else { return total };
        if self.panes[p].proc.is_some() {
            if let Some(Some(base)) = self.panes[p].mon.iter().find(|m| m.dev.id == d.id).map(|m| m.base) {
                used = base;
            }
        }
        total - used
    }

    pub fn v_adjust(&mut self, p: usize) -> VAdjust {
        let key = self.dev_key(p);
        let (c, src, borrowed) = self.calib_lookup(p, &key);
        self.panes[p].v_borrowed = borrowed;
        let Some(c) = c.filter(|c| truthy(c.get("vllm"))) else { return VAdjust::default() };
        let mut delta: HashMap<String, f64> = c
            .get("delta")
            .and_then(|v| v.as_object())
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect())
            .unwrap_or_default();
        if gi(&c, "hv", 0) != VLLM_HEURISTIC_VERSION {
            if let Some(src) = src.and_then(|s| s.v().cloned()) {
                // recorded against the older heuristic: re-base each card's offset
                let mut prm = default_vllm_model();
                if let Some(Value::Object(o)) = c.get("params") {
                    for (k, v) in o {
                        prm.insert(k.clone(), v.clone());
                    }
                }
                let order: Vec<String> = c
                    .get("order")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let mode = gs(&c, "mode", "single");
                for (i, dev_id) in order.iter().enumerate() {
                    let Some(d) = delta.get_mut(dev_id) else { continue };
                    let first = mode != "pipeline" || i == 0;
                    let last = mode != "pipeline" || i == order.len() - 1;
                    let tp = if mode == "tensor" { 2.0 } else { 1.0 };
                    *d += vllm_heuristic_extra(&src, &prm, last, first, tp, 1)
                        - vllm_heuristic_extra(&src, &prm, last, first, tp, VLLM_HEURISTIC_VERSION);
                }
            }
        }
        VAdjust { delta, factor: gf(&c, "factor", 1.0), present: true }
    }

    pub fn v_plan(&mut self, p: usize, s: &Opts) -> Option<VPlan> {
        let info = self.panes[p].info_v()?;
        let devs = self.v_devs(p);
        if devs.is_empty() {
            return None;
        }
        let manual = Manual { mode: gs(s, "dual_mode", ""), ts: gs(s, "dual_ts", ""), first: gs(s, "dual_first", "") };
        let adj = self.v_adjust(p);
        self.panes[p].v_adj_used = adj.clone();
        let totals: HashMap<String, f64> = devs.iter().map(|d| (d.id.clone(), Self::v_total(d))).collect();
        let strat = gs(s, "dual_strategy", "Auto VRAM");
        let plan = plan_vllm(&info, s, &devs, &totals, &strat, if strat == "Manual" { Some(&manual) } else { None }, &adj);
        self.panes[p].vplan = Some(plan.clone());
        Some(plan)
    }

    pub fn v_update_estimate(&mut self, p: usize) {
        let Some(info) = self.panes[p].info_v() else {
            self.panes[p].est_text = if self.info_pending(p) { "Estimate: reading model metadata…" } else { "Estimate: select a model" }.into();
            self.panes[p].est_pct = 0.0;
            return;
        };
        let devs = self.v_devs(p);
        if devs.is_empty() {
            self.panes[p].est_text = "Estimate: assign a GPU to this pane (Settings)".into();
            self.panes[p].est_pct = 0.0;
            return;
        }
        let s = self.v_current_options(p);
        let plan = self.v_plan(p, &s).unwrap();
        let ctx = gi(&s, "ctx", 8192);
        let cap = info.n_ctx_train;
        let mut lines = vec![];
        let mut warn = vec![];
        let mut worst: f64 = 0.0;
        let mut short_free = false;
        let multi = plan.cards.len() > 1 || self.dual_active(p);
        let gpu_util = gf(&s, "gpu_util", 0.9);
        let ncards = plan.cards.len();
        for (ci, (d, c)) in plan.order.iter().zip(&plan.cards).enumerate() {
            let need = c.weights + c.fixed + vllm_kv_need(c, &info, ctx as f64);
            let pct = if c.budget > 0.0 { 100.0 * need / c.budget } else { 999.0 };
            worst = worst.max(pct);
            let free = self.v_free(p, d);
            if c.budget > free + 64.0 * MIB {
                short_free = true;
                warn.push(format!(
                    "{} has only {:.1} GiB free; GPU memory {:.2} needs {:.1} (max {:.2})",
                    d.short_name(),
                    free / GIB,
                    gpu_util,
                    c.budget / GIB,
                    ((free - 64.0 * MIB) / c.total).max(0.0)
                ));
            }
            let lay = if plan.mode == "pipeline" {
                let (a, b) = c.layers.unwrap_or((0, 0));
                format!("layers {a}-{b}") + if ci == ncards - 1 { " + output" } else { "" }
            } else if plan.mode == "tensor" {
                "half of every layer".into()
            } else {
                String::new()
            };
            let head = if multi { format!("{}: {lay}  ", d.short_name()) } else { String::new() };
            lines.push(
                format!("{head}budget {:.1} of {:.1} GiB  •  weights {:.2}", c.budget / GIB, c.total / GIB, c.weights / GIB)
                    + &if c.offload != 0.0 { format!(" (+{:.1} in RAM)", c.offload / GIB) } else { String::new() }
                    + &format!(
                        "  •  runtime+activation ≈ {:.2}  •  KV pool ≈ {:.2} GiB",
                        c.fixed / GIB,
                        c.kv.max(0.0) / GIB
                    ),
            );
        }
        let tokens = plan.cards.iter().map(|c| vllm_capacity(c, &info, ctx.max(1) as f64)).min().unwrap_or(0);
        let maxc = if cap != 0 { plan.max_ctx.min(cap) } else { plan.max_ctx };
        if plan.cards.iter().all(|c| c.kv > 0.0) {
            let conc = if ctx != 0 { tokens as f64 / ctx as f64 } else { 0.0 };
            lines.push(
                format!("KV pool holds ≈ {} tokens  •  Context {} ", fmt_thousands(tokens), fmt_thousands(ctx))
                    + &if plan.max_ctx >= ctx { format!("fits ×{conc:.1}") } else { "does NOT fit".into() }
                    + &format!("  •  Fit → {}", fmt_thousands(if maxc >= 2048 { maxc / 1024 * 1024 } else { maxc })),
            );
        } else {
            lines.push("No room left for a KV cache - weights + activations exceed the budget".into());
        }
        let mut txt = format!("Estimated memory (vLLM)\n{}", lines.join("\n"));
        let adj = self.panes[p].v_adj_used.clone();
        if adj.present {
            let mut dl: Vec<(&String, &f64)> = adj.delta.iter().collect();
            dl.sort_by(|a, b| a.0.cmp(b.0));
            txt += &format!(
                "\n✓ calibrated from a real load (runtime {} GiB, KV ×{:.2})",
                dl.iter().map(|(_, v)| format!("{:+.2}", *v / GIB)).collect::<Vec<_>>().join(", "),
                adj.factor
            );
            if let Some(b) = &self.panes[p].v_borrowed {
                txt += &format!(
                    " - borrowed from {} (same architecture)",
                    std::path::Path::new(b).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                );
            }
        } else {
            txt += "\nuncalibrated: the first load measures this model's real numbers";
        }
        if cap != 0 && ctx > cap {
            warn.push(format!("Context exceeds the model's trained length ({}) - vLLM will refuse", fmt_thousands(cap)));
        }
        if !plan.note.is_empty() && plan.mode == "tensor" && !plan.fits {
            warn.push(plan.note.clone());
        }
        if info.multimodal && !gb(&s, "text_only", false) {
            txt += "\nvision enabled: activation estimate is rough until calibrated (Text only frees memory)";
        }
        if let (Some(q), true) = (s.get("quant").and_then(|v| v.as_str()), info.quant.is_empty()) {
            let lb: f64 = info.layer_bytes.iter().sum();
            txt += &format!(
                "\nquantized at load ({q}): layer weights ≈ {:.1} GiB instead of {:.1}",
                lb * vllm_online_quant(q).unwrap_or(1.0) / GIB,
                lb / GIB
            );
        }
        if !warn.is_empty() {
            txt += &format!("\n⚠ {}", warn.join("; "));
        }
        let red = !plan.fits || short_free || (cap != 0 && ctx > cap);
        let dual_txt = if self.dual_active(p) {
            let mut head = match plan.mode.as_str() {
                "single" => format!("Plan: single GPU on {}", plan.order[0].name),
                "pipeline" => format!(
                    "Plan: pipeline split, {} + {} layers (first on {})",
                    plan.k,
                    info.n_layer - plan.k,
                    plan.order[0].short_name()
                ),
                _ => "Plan: tensor split (every layer halved)".into(),
            };
            if !plan.note.is_empty() {
                head += &format!(" - {}", plan.note);
            }
            Some(format!("{head}\n{}", self.v_compat_lines().join("\n")))
        } else {
            None
        };
        let pane = &mut self.panes[p];
        pane.est_text = txt;
        pane.est_pct = worst.min(100.0);
        pane.est_color = if red { "#d0342c" } else if worst >= 92.0 { "#d08a00" } else { "#2e9e44" }.into();
        pane.est_over = red;
        if let Some(d) = dual_txt {
            pane.dual_info = d;
        }
    }

    pub fn v_compat_lines(&self) -> Vec<String> {
        let devs = self.dual_devices();
        let mut out = vec![devs
            .iter()
            .map(|d| {
                let (g, w) = pcie_link(d);
                format!("{}: ", d.short_name())
                    + &match g {
                        Some(g) => format!("PCIe {g}.0 x{}", w.map(|x| x.to_string()).unwrap_or("?".into())),
                        None => "PCIe ?".into(),
                    }
            })
            .collect::<Vec<_>>()
            .join(", ")];
        let mut t: Vec<i64> = devs.iter().map(|d| d.total_mib).collect();
        t.sort();
        if t.len() == 2 && t[1] as f64 > t[0] as f64 * 1.25 {
            out.push(format!(
                "Mismatched VRAM ({:.0} vs {:.0} GiB): pipeline split uses an uneven layer ratio; tensor split is capped by the smaller card",
                t[0] as f64 / 1024.0,
                t[1] as f64 / 1024.0
            ));
        }
        out
    }

    pub fn v_fit(&mut self, p: usize) {
        if self.panes[p].proc.is_some() {
            self.msg("info", "Stop the model first - Fit measures free VRAM on the GPU(s).");
            return;
        }
        let (Some(info), false) = (self.panes[p].info_v(), self.v_devs(p).is_empty()) else {
            self.msg("info", "Select a model (and assign a GPU) first.");
            return;
        };
        let mut s = self.v_current_options(p);
        let cap = if info.n_ctx_train != 0 { info.n_ctx_train } else { 131072 };
        s.insert("ctx".into(), 1.into());
        let plan = self.v_plan(p, &s).unwrap();
        let m = plan.max_ctx.min(cap);
        if m < 1024 {
            let rows: Vec<String> = plan
                .order
                .iter()
                .zip(&plan.cards)
                .map(|(d, c)| {
                    format!(
                        "{}: budget {:.2} GiB, weights {:.2}, runtime+activation ≈ {:.2}, left for KV {:.2} GiB",
                        d.name,
                        c.budget / GIB,
                        c.weights / GIB,
                        c.fixed / GIB,
                        c.kv / GIB
                    )
                })
                .collect();
            self.msg(
                "warning",
                &format!(
                    "Not even 1024 tokens of context fit.\n\n{}\n\nTry: raise GPU memory, lower Batch tokens, tick Text only (vision models), fp8 KV cache, a quantized model, or Dual GPU mode.",
                    rows.join("\n")
                ),
            );
            return;
        }
        let ctx = if m >= 2048 { m / 1024 * 1024 } else { m / 256 * 256 };
        self.set_var_notify(p, "ctx", Value::from(ctx.to_string()));
        let how = match plan.mode.as_str() {
            "single" => format!("on {}", plan.order[0].name),
            "pipeline" => format!("pipeline split {}/{}", plan.k, info.n_layer - plan.k),
            _ => "tensor split".into(),
        };
        let kvmin = plan.cards.iter().map(|c| c.kv).fold(f64::INFINITY, f64::min);
        self.set_top_status(
            format!("{} pane: Fit chose context {} {how}", cap_side(self.panes[p].side), fmt_thousands(ctx))
                + if ctx >= cap { " (model's max)" } else { "" }
                + &format!("  •  KV ≈ {:.2} GiB", kvmin / GIB),
        );
    }

    pub fn v_build_command(&mut self, p: usize) -> Result<(Vec<String>, Opts), String> {
        let s = self.v_current_options(p);
        let path = self.model_path(p).ok_or("no model selected")?;
        let Some(info) = self.panes[p].info_v() else {
            return Err("model metadata not read yet - wait a moment and try again".into());
        };
        let plan = self.v_plan(p, &s).ok_or("no GPU assigned")?;
        let g = self.settings.g().clone();
        let mut cmd: Vec<String> = vec![
            expanduser(&gs(&g, "vllm_bin", "vllm")).to_string_lossy().into_owned(),
            "serve".into(),
            path.to_string_lossy().into_owned(),
            "--served-model-name".into(),
            std::path::Path::new(&self.panes[p].model_var).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            "--host".into(),
            gs(&g, "host", "127.0.0.1"),
            "--port".into(),
            gi(&s, "port", 8042).to_string(),
            "--max-model-len".into(),
            gi(&s, "ctx", 8192).to_string(),
            "--gpu-memory-utilization".into(),
            format!("{:.2}", gf(&s, "gpu_util", 0.9)),
            "--max-num-seqs".into(),
            gi(&s, "parallel", 1).to_string(),
        ];
        if gs(&s, "kv_dtype", "auto") != "auto" {
            cmd.extend(["--kv-cache-dtype".into(), gs(&s, "kv_dtype", "")]);
        }
        if gs(&s, "dtype", "auto") != "auto" {
            cmd.extend(["--dtype".into(), gs(&s, "dtype", "")]);
        }
        if let (Some(q), true) = (s.get("quant").and_then(|v| v.as_str()), info.quant.is_empty()) {
            cmd.extend(["--quantization".into(), q.to_string()]);
        }
        if gi(&s, "max_batched", 0) > 0 {
            cmd.extend(["--max-num-batched-tokens".into(), gi(&s, "max_batched", 0).to_string()]);
        }
        let co = gf(&s, "cpu_offload", 0.0);
        if co > 0.0 {
            cmd.extend(["--cpu-offload-gb".into(), py_g(co)]);
        }
        if gb(&s, "eager", false) {
            cmd.push("--enforce-eager".into());
        }
        if gb(&s, "text_only", false) && info.multimodal {
            let cfg_txt = serde_json::to_string(&info.cfg).unwrap_or_default();
            let mut parts = vec!["\"image\": 0".to_string()];
            if cfg_txt.contains("video") {
                parts.push("\"video\": 0".into());
            }
            if matches!(info.cfg.get("audio_config"), Some(Value::Object(_))) {
                parts.push("\"audio\": 0".into());
            }
            cmd.extend(["--limit-mm-per-prompt".into(), format!("{{{}}}", parts.join(", "))]);
        }
        let reasoning = gs(&s, "reasoning", "(none)");
        if !reasoning.is_empty() && reasoning != "(none)" {
            cmd.extend(["--reasoning-parser".into(), reasoning]);
        }
        if gb(&s, "relax_tpl", false) {
            let (tpl_path, n_fixed) = make_relaxed_template(info.chat_template.clone(), &path)?;
            cmd.extend(["--chat-template".into(), tpl_path.to_string_lossy().into_owned()]);
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
        if gs(&s, "mtp", "") == "(built-in)" && info.nextn != 0 {
            cmd.extend([
                "--speculative-config".into(),
                format!("{{\"method\": \"mtp\", \"num_speculative_tokens\": {}}}", gi(&s, "mtp_n", 3)),
            ]);
        }
        if plan.mode == "pipeline" {
            cmd.extend(["--pipeline-parallel-size".into(), "2".into()]);
        } else if plan.mode == "tensor" {
            cmd.extend(["--tensor-parallel-size".into(), "2".into()]);
        }
        let extra = gs(&s, "extra", "");
        if !extra.trim().is_empty() {
            cmd.extend(shlex_split(&extra)?);
        }
        Ok((cmd, s))
    }

    pub fn v_env(&self, p: usize) -> Vec<(String, String)> {
        let mut env = vec![];
        if let Some(plan) = &self.panes[p].vplan {
            env.push((
                "HIP_VISIBLE_DEVICES".to_string(),
                plan.order.iter().map(|d| d.index.to_string()).collect::<Vec<_>>().join(","),
            ));
            if plan.mode == "pipeline" {
                let n = self.panes[p].info.as_ref().map(|i| i.n_layer()).unwrap_or(0);
                env.push(("VLLM_PP_LAYER_PARTITION".into(), format!("{},{}", plan.k, n - plan.k)));
            }
        }
        for (k, v) in vllm_base_env(self.settings.g()) {
            if let Some(e) = env.iter_mut().find(|e| e.0 == k) {
                e.1 = v;
            } else {
                env.push((k, v));
            }
        }
        env
    }

    pub fn v_copy_prefix(&self) -> String {
        match vllm_venv(&gs(self.settings.g(), "vllm_bin", "")) {
            Some(v) => format!("source {} && ", shlex_quote(&v.join("bin").join("activate").to_string_lossy())),
            None => String::new(),
        }
    }

    pub fn v_monitors(&self, p: usize) -> Vec<Mon> {
        let Some(plan) = &self.panes[p].vplan else { return vec![] };
        plan.order
            .iter()
            .zip(&plan.cards)
            .map(|(d, c)| Mon {
                dev: d.clone(),
                base: gpu_stats(Some(d)).0,
                gtt: gpu_gtt(Some(d)),
                peak: 0.0,
                expected: c.budget,
                full: c.offload == 0.0,
                dv: None,
                dg: None,
                gtt_noted: 0.0,
            })
            .collect()
    }

    // ---------------- monitoring ---------------- //
    pub fn v_parse_line(&mut self, p: usize, line: &str) {
        let re = vre();
        let rank = re.tag.captures(line).map(|m| vllm_rank(&m[1])).unwrap_or(0);
        for (key, rx) in [("load", &re.load), ("kv", &re.kvmem), ("graph", &re.graph)] {
            if let Some(mm) = rx.captures(line) {
                let v: f64 = mm[1].parse().unwrap_or(0.0);
                self.panes[p].vbuf.entry(rank).or_default().insert(key.into(), v);
                self.v_show_buffers(p);
                return;
            }
        }
        if let Some(mm) = re.kvtok.captures(line) {
            self.panes[p].vkv_tokens = mm[1].replace(',', "").parse().ok();
            self.panes[p].vconc = re.conc.captures(line).and_then(|c| c[2].parse().ok());
            self.v_show_buffers(p);
            return;
        }
        if re.ready.is_match(line) && self.panes[p].state == "Loading" {
            self.became_ready(p);
        }
    }

    pub fn v_show_buffers(&mut self, p: usize) {
        let pane = &mut self.panes[p];
        let tot = |k: &str| pane.vbuf.values().map(|b| b.get(k).cloned().unwrap_or(0.0)).sum::<f64>();
        let (load, kv, graph) = (tot("load"), tot("kv"), tot("graph"));
        pane.buffers = vec![("vllm".into(), 1.0)];
        let mut parts = vec![format!("weights {load:.2}")];
        if kv != 0.0 {
            parts.push(format!("KV {kv:.2}"));
        }
        if graph != 0.0 {
            parts.push(format!("graphs {graph:.2}"));
        }
        let mut txt = parts.join(" + ") + " GiB";
        if let Some(t) = pane.vkv_tokens.filter(|t| *t != 0) {
            txt += &format!("   •   KV cache {} tokens", fmt_thousands(t));
            if let Some(c) = pane.vconc.filter(|c| *c != 0.0) {
                txt += &format!(" (×{c:.1} at full context)");
            }
        }
        pane.set_status("buffers", txt);
    }

    pub fn v_metrics(&mut self, p: usize, m: HashMap<String, f64>) {
        let g = |k: &str| m.get(k).cloned().unwrap_or(0.0);
        let running = g("num_requests_running");
        let waiting = g("num_requests_waiting");
        if self.panes[p].state == "Ready" || self.panes[p].state == "Processing" {
            self.set_state(p, if running > 0.0 { "Processing" } else { "Ready" });
        }
        let (pt, hits) = (g("prompt_tokens_total"), g("prefix_cache_hits_total"));
        let gt = g("generation_tokens_total");
        let pre = g("request_prefill_time_seconds_sum");
        let dec = g("request_decode_time_seconds_sum");
        let mut avg = vec![];
        if pre > 0.0 && pt - hits > 0.0 {
            avg.push(format!("prompt {:.1} t/s", (pt - hits) / pre));
        }
        if dec > 0.0 && gt > 0.0 {
            avg.push(format!("gen {:.1} t/s", gt / dec));
        }
        if pt != 0.0 || gt != 0.0 {
            avg.push(format!("{} prompt / {} generated tokens total", fmt_thousands(pt as i64), fmt_thousands(gt as i64)));
        }
        if let Some(kvu) = m.get("kv_cache_usage_perc").or(m.get("gpu_cache_usage_perc")) {
            avg.push(format!("KV in use {:.0}%", kvu * 100.0));
        }
        self.panes[p].set_status("avg", if avg.is_empty() { "—".into() } else { avg.join("   •   ") });
        if running != 0.0 || waiting != 0.0 {
            return;
        }
        let prev = self.panes[p].metrics_prev.replace(m.clone());
        let Some(prev) = prev else { return };
        let d = |k: &str| g(k) - prev.get(k).cloned().unwrap_or(0.0);
        let dg = d("generation_tokens_total");
        if dg <= 0.0 {
            return;
        }
        let (dp, dh) = (d("prompt_tokens_total"), d("prefix_cache_hits_total").max(0.0));
        let comp = (dp - dh).max(0.0);
        let dpre = d("request_prefill_time_seconds_sum");
        let ddec = d("request_decode_time_seconds_sum");
        if comp > 0.0 && dpre > 0.0 {
            self.panes[p].last_prompt = Some((comp as i64, comp / dpre));
        }
        if ddec > 0.0 {
            self.panes[p].last_gen = Some((dg as i64, dg / ddec));
        }
        let mut r = serde_json::Map::new();
        r.insert("p_ms".into(), json!(dpre * 1000.0));
        r.insert("p_tok".into(), json!(comp as i64));
        r.insert("g_ms".into(), json!(ddec * 1000.0));
        r.insert("g_tok".into(), json!(dg as i64));
        r.insert("n_end".into(), json!((dp + dg) as i64));
        let ddraft = d("spec_decode_num_draft_tokens_total");
        if ddraft > 0.0 {
            let acc = d("spec_decode_num_accepted_tokens_total");
            self.panes[p].last_accept = Some(acc / ddraft);
            r.insert("acc".into(), json!(acc as i64));
            r.insert("gen".into(), json!(ddraft as i64));
        }
        self.update_speed(p);
        self.stats_record(p, &r);
    }

    pub fn v_record_calibration(&mut self, p: usize) {
        if self.panes[p].proc.is_none() || !matches!(self.panes[p].state.as_str(), "Ready" | "Processing") {
            return;
        }
        self.check_spill(p);
        let Some(plan) = self.panes[p].v_launch_plan.clone() else { return };
        if self.panes[p].spilled || self.panes[p].loaded.is_none() {
            return;
        }
        if !(0..plan.cards.len()).all(|r| self.panes[p].vbuf.get(&r).map(|b| b.contains_key("kv")).unwrap_or(false)) {
            return;
        }
        let (rel, opts, _) = self.panes[p].loaded.clone().unwrap();
        let old = self.panes[p].v_launch_adj.clone();
        let old_f = if old.present { old.factor } else { 1.0 };
        let Some(info) = self.panes[p].info_v() else { return };
        let mut delta = serde_json::Map::new();
        let mut caps = vec![];
        let ctx = gi(&opts, "ctx", 8192).max(1) as f64;
        for (r, (d, c)) in plan.order.iter().zip(&plan.cards).enumerate() {
            let kv_meas = self.panes[p].vbuf[&r]["kv"] * GIB;
            let fixed_meas = c.budget - c.weights - kv_meas;
            let nd = calib_clamp(old.delta.get(&d.id).cloned().unwrap_or(0.0) + fixed_meas - c.fixed, 4.0 * GIB);
            delta.insert(d.id.clone(), json!(nd));
            let per = vllm_kv_need(c, &info, ctx) / ctx;
            if per > 0.0 {
                caps.push(kv_meas / per);
            }
        }
        let mut factor = old_f;
        if let (Some(t), false) = (self.panes[p].vkv_tokens.filter(|t| *t != 0), caps.is_empty()) {
            let mn = caps.iter().cloned().fold(f64::INFINITY, f64::min);
            factor = (old_f * mn / t as f64).clamp(0.2, 5.0);
        }
        let mut rec = json!({
            "vllm": true, "hv": VLLM_HEURISTIC_VERSION, "params": opts, "delta": delta, "factor": factor,
            "kv_tokens": self.panes[p].vkv_tokens, "mode": plan.mode,
            "order": plan.order.iter().map(|d| d.id.clone()).collect::<Vec<_>>(),
            "when": now_iso(),
        });
        if let Some(i) = self.panes[p].info.as_ref() {
            rec["sig"] = json!(i.sig());
        }
        let key = self.dev_key(p);
        self.settings.set_calib(&key, &rel, rec.as_object().unwrap().clone());
        self.panes[p].borrow_cache.clear();
        if self.panes[p].model_var == rel {
            self.update_estimate(p);
        }
    }
}

/// Python's `f"{x:g}"`
pub fn py_g(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        let s = format!("{:.6}", x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}
