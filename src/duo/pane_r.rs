//! Everything a pane does differently when its backend is Radiance.
//!
//! Radiance sizes memory itself (see radiance.rs), so the "estimate" here is what the engine
//! reported for the same memory settings - from a real load or from Fit's plan check, a short
//! start of the server that is stopped as soon as the startup log states the KV pool.

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::app::App;
use super::devices::{gpu_gtt, gpu_stats, Device};
use super::pane::{cap_side, Mon};
use super::proc::{self, Proc, ProcMsg};
use super::radiance::*;
use super::telemetry::unix_now;
use super::util::*;

/// Fit's plan check in flight.
pub struct Probe {
    pub proc: Proc,
    pub rx: Receiver<ProcMsg>,
    pub t0: Instant,
    pub ctx: i64,
    pub fp: String,
    pub rel: String,
    pub lines: Vec<String>,
    pub kv: Option<i64>,
    pub backs: Option<f64>,
    pub sessions: Option<f64>,
    pub killed: bool,
    pub seqs: i64,
    /// bisection over the context: largest that started, smallest that did not fit
    pub lo: Option<i64>,
    pub hi: Option<i64>,
    pub tries: u32,
    pub budget: Budget,
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(300);

fn list_or(v: String, list: &[&str], d: &str) -> String {
    if list.contains(&v.as_str()) { v } else { d.to_string() }
}

impl App {
    // ============================================================ options ==
    pub fn r_current_options(&self, p: usize) -> Opts {
        let v = &self.panes[p];
        let spec = v.var_s("r_spec").trim().to_string();
        let spec = if spec == "auto" || spec.parse::<u32>().is_ok() { spec } else { "auto".into() };
        let reasoning = v.var_s("r_reasoning").trim().to_string();
        let mut o = Map::new();
        o.insert("ctx".into(), v.int("ctx", 32768).max(0).into());
        o.insert("parallel".into(), v.int("parallel", 1).max(1).into());
        o.insert("r_kv".into(), list_or(v.var_s("r_kv"), &RAD_KV_TYPES, "fp8").into());
        o.insert("r_batched".into(), v.int("r_batched", 2048).max(64).into());
        o.insert("r_headroom".into(), v.int("r_headroom", 96).max(0).into());
        o.insert("r_spec".into(), spec.into());
        o.insert("r_spec_ngram".into(), v.int("r_spec_ngram", 0).clamp(0, 64).into());
        let nmax = v.int("r_ngram_max", 4).clamp(1, 15);
        o.insert("r_ngram_max".into(), nmax.into());
        o.insert("r_ngram_min".into(), v.int("r_ngram_min", 2).clamp(1, nmax).into());
        o.insert("r_draft_window".into(), list_or(v.var_s("r_draft_window"), &RAD_DRAFT_WINDOW, "fixed").into());
        o.insert("r_vision".into(), list_or(v.var_s("r_vision"), &RAD_VISION, "(default)").into());
        o.insert("r_embed".into(), list_or(v.var_s("r_embed"), &RAD_EMBED, "(default)").into());
        o.insert("r_placement".into(), list_or(v.var_s("r_placement"), &RAD_PLACEMENT, "auto").into());
        o.insert("r_host_pool".into(), v.int("r_host_pool", 0).max(0).into());
        o.insert("r_prefix_mib".into(), v.int("r_prefix_mib", 0).max(0).into());
        o.insert("r_prefill_span".into(), v.int("r_prefill_span", 1).max(1).into());
        o.insert("r_reasoning".into(), Value::from(if reasoning.is_empty() { "(default)".to_string() } else { reasoning }));
        o.insert("r_tp_wire".into(), list_or(v.var_s("r_tp_wire"), &RAD_TP_WIRE, "exact").into());
        o.insert("dual_strategy".into(), RAD_STRATEGIES[0].into());
        o.insert("dual_mode".into(), "tensor".into());
        o.insert("port".into(), v.int("port", self.default_port(p)).into());
        o.insert("extra".into(), Value::from(v.var_s("extra")));
        o
    }

    pub fn r_prepare_options(&mut self, _p: usize, s: &mut Opts) {
        s.insert("dual_strategy".into(), RAD_STRATEGIES[0].into());
        s.insert("dual_mode".into(), "tensor".into());
    }

    pub fn r_apply_info(&mut self, _p: usize, _info: Arc<RadInfo>) {}

    /// Cards this pane would run on: both in Dual GPU mode (tensor parallel), else its own.
    pub fn r_devs(&self, p: usize) -> Vec<Device> {
        if self.dual_active(p) {
            return self.dual_devices();
        }
        self.panes[p].device.clone().into_iter().collect()
    }

    /// Identifies a memory plan: the cards and every option that changes the split.
    pub fn r_pool_fp(&self, p: usize, s: &Opts) -> String {
        let mut names: Vec<String> = self.r_devs(p).iter().map(|d| d.name.clone()).collect();
        names.sort();
        let vals: Vec<String> = RAD_POOL_KEYS.iter().map(|k| opt_str(s.get(*k))).collect();
        format!("{}|{}", names.join("+"), vals.join(","))
    }

    /// The KV pool recorded for this model and these memory settings.
    pub fn r_pool(&self, p: usize, s: &Opts) -> Option<Opts> {
        let rel = &self.panes[p].model_var;
        let rec = self.settings.get_calib(&self.dev_key(p), rel)?;
        let fp = self.r_pool_fp(p, s);
        rec.get("pools").and_then(|v| v.get(&fp)).and_then(|v| v.as_object()).cloned()
    }

    fn r_save_pool(&mut self, p: usize, rel: &str, fp: &str, data: Value) {
        let key = self.dev_key(p);
        let mut rec = self.settings.get_calib(&key, rel).unwrap_or_default();
        rec.insert("radiance".into(), true.into());
        let mut pools = rec.get("pools").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        pools.insert(fp.to_string(), data);
        // keep the 24 newest plans
        if pools.len() > 24 {
            let mut v: Vec<(String, String)> = pools.iter().map(|(k, x)| (k.clone(), gs(x.as_object().unwrap_or(&Map::new()), "when", ""))).collect();
            v.sort_by(|a, b| a.1.cmp(&b.1));
            for (k, _) in v.iter().take(pools.len() - 24) {
                pools.remove(k);
            }
        }
        rec.insert("pools".into(), Value::Object(pools));
        rec.insert("when".into(), now_iso().into());
        if let Some(i) = self.panes[p].info.as_ref() {
            rec.insert("sig".into(), i.sig().into());
        }
        self.settings.set_calib(&key, rel, rec);
    }

    // =========================================================== estimate ==
    pub fn r_update_estimate(&mut self, p: usize) {
        let Some(info) = self.panes[p].info_r() else {
            self.panes[p].est_text = if self.info_pending(p) { "Estimate: reading model metadata…" } else { "Estimate: select a model" }.into();
            self.panes[p].est_pct = 0.0;
            return;
        };
        let devs = self.r_devs(p);
        if devs.is_empty() {
            self.panes[p].est_text = "Estimate: assign a GPU to this pane (Settings)".into();
            self.panes[p].est_pct = 0.0;
            return;
        }
        let s = self.r_current_options(p);
        let tp = devs.len() as f64;
        let ctx = gi(&s, "ctx", 0);
        let seqs = gi(&s, "parallel", 1).max(1);
        let head = gi(&s, "r_headroom", 96) as f64 * MIB;
        let mut lines = vec![];
        let mut warn = vec![];
        let mut red = false;
        let mut yellow = false;
        let mut min_left = f64::INFINITY;
        let mut tight_total = f64::INFINITY;
        for d in &devs {
            let total = Self::v_total(d);
            tight_total = tight_total.min(total);
            let free = if self.panes[p].proc.is_some() { total } else { self.v_free(p, d) };
            let w = info.bytes / tp;
            let left = free - head - w;
            min_left = min_left.min(left);
            let mut l = format!("{}: card {:.1} GiB", d.short_name(), total / GIB);
            if self.panes[p].proc.is_none() && free < total - 1.5 * GIB {
                l += &format!(", {:.1} free now", free / GIB);
                warn.push(format!(
                    "{:.1} GiB of {} is already in use - Radiance budgets only what is free when it starts",
                    (total - free) / GIB,
                    d.short_name()
                ));
                yellow = true;
            }
            l += &format!("  •  weights ≈ {:.2}{}", w / GIB, if tp > 1.0 { " (its half)" } else { "" });
            l += &format!("  •  ≈ {:.1} GiB left for KV + activations", left.max(0.0) / GIB);
            lines.push(l);
        }
        let offload = gi(&s, "r_host_pool", 0) > 0 || gs(&s, "r_placement", "auto") == "expert_tiered";
        if min_left < 0.5 * GIB && !offload {
            if min_left < 0.0 {
                warn.push("the container is larger than the card - it only fits if the token embedding / vision tower go to RAM or experts to a Host pool (MoE)".into());
            }
            red = min_left < -2.0 * GIB;
            yellow = true;
        }
        let pool = self.r_pool(p, &s);
        let cap = info.n_ctx_train;
        match &pool {
            Some(pl) => {
                let built = gi(pl, "kv_tokens", 0);
                let known = pool_capacity(pl);
                let tokens = known.unwrap_or(built);
                let src = if gs(pl, "source", "") == "probe" { "Fit's plan check" } else { "a real load" };
                let when = gs(pl, "when", "");
                let day = if when.len() >= 10 { format!(", {}", &when[..10]) } else { String::new() };
                let l = match known {
                    Some(c) if c > built + built / 20 => format!(
                        "The card holds ≈ {} tokens of KV at these settings (from {src}{day}; that run asked for {})",
                        fmt_thousands(c),
                        fmt_thousands(built)
                    ),
                    Some(_) => format!("KV pool {} tokens - all the card holds at these settings (from {src}{day})", fmt_thousands(built)),
                    None => {
                        yellow = true;
                        format!(
                            "KV pool {} tokens (from {src}{day}) - only what that run asked for; the card may hold more: press Fit to check",
                            fmt_thousands(built)
                        )
                    }
                };
                lines.push(l);
                let ectx = if ctx == 0 { cap } else { ctx };
                if ectx > 0 && tokens > 0 {
                    let per = tokens as f64 / ectx as f64;
                    let mut l2 = format!("Context {} × {seqs} seq{}", fmt_thousands(ectx), if seqs == 1 { "" } else { "s" });
                    if ectx > tokens {
                        l2 += "  •  does NOT fit the pool - Radiance refuses a context the pool cannot hold once";
                        red = true;
                    } else if per + 1e-9 < seqs as f64 {
                        l2 += &format!("  •  the pool holds {per:.1} full-length sessions (fine for shorter chats; a full one in every seq would preempt)");
                        yellow = true;
                    } else {
                        l2 += &format!("  •  fits ×{per:.1}");
                    }
                    let fit = r_fit_ctx(tokens, seqs, cap);
                    l2 += &format!("  •  Fit → {}", fmt_thousands(fit));
                    lines.push(l2);
                }
            }
            None => {
                lines.push("KV pool not measured yet for these memory settings - press Fit (a short plan check) or Load; the startup log shows the real figure".into());
                yellow = true;
            }
        }
        if cap != 0 && ctx > cap {
            warn.push(format!(
                "Context exceeds the model's training length ({}) - add --rope-scaling yarn:FACTOR in Extra args to go past it",
                fmt_thousands(cap)
            ));
            yellow = true;
        }
        if ctx == 0 {
            lines.push(format!("Context 0 = the model's own training context{}", if cap != 0 { format!(" ({})", fmt_thousands(cap)) } else { String::new() }));
        }
        if info.vision && gs(&s, "r_vision", "(default)") != "ram" {
            lines.push("vision model: Vision tower → ram moves the encoder to system RAM and gives its VRAM to the KV pool (≈55 ms per image)".into());
        }
        if gi(&s, "r_batched", 2048) > 4096 || seqs > 16 {
            lines.push("large Batch tokens / Max seqs grow the activation arena, which comes out of the KV pool".into());
        }
        let mut txt = format!(
            "Radiance plans memory itself: it claims {} less {} MiB headroom and splits it between weights and a paged KV pool\n{}",
            if tp > 1.0 { "both cards" } else { "the card" },
            gi(&s, "r_headroom", 96),
            lines.join("\n")
        );
        if tp > 1.0 {
            txt += &format!("\nTensor parallel (--tp {}): every layer split across the cards; the KV pool is sized to the tightest card", devs.len());
            let mut t: Vec<i64> = devs.iter().map(|d| d.total_mib).collect();
            t.sort();
            if t.len() == 2 && t[1] as f64 > t[0] as f64 * 1.25 {
                warn.push(format!(
                    "mismatched cards ({:.0} vs {:.0} GiB): the larger card's extra memory goes unused - the larger card alone usually holds more context",
                    t[0] as f64 / 1024.0,
                    t[1] as f64 / 1024.0
                ));
                yellow = true;
            }
        }
        if gi(&s, "r_spec_ngram", 0) > 0 && info.draft {
            warn.push("this container has a draft head, so Radiance refuses Prompt lookup at startup: set it to 0 and use Spec tokens".into());
            yellow = true;
        }
        if matches!(info.kind, RadKind::Embedding | RadKind::Rerank) {
            let ectx = if ctx == 0 { cap } else { ctx };
            let batched = gi(&s, "r_batched", 2048);
            if ectx > batched {
                warn.push(format!(
                    "Batch tokens ({}) is below Context ({}): this model reads each input whole in one step, so raise Batch tokens to the longest input you will send",
                    fmt_thousands(batched),
                    fmt_thousands(ectx)
                ));
                yellow = true;
            }
        }
        if !info.note.is_empty() {
            warn.push(info.note.clone());
        }
        if !warn.is_empty() {
            txt += &format!("\n⚠ {}", warn.join("; "));
        }
        let pct = match &pool {
            Some(pl) => {
                let tokens = pool_capacity(pl).unwrap_or_else(|| gi(pl, "kv_tokens", 0)).max(1) as f64;
                let ectx = if ctx == 0 { cap.max(1) } else { ctx } as f64;
                100.0 * (ectx * seqs as f64 / tokens).min(1.0)
            }
            None => 100.0 * ((info.bytes / tp + head) / tight_total.max(1.0)).min(1.0),
        };
        let dual_txt = if self.dual_active(p) {
            let names: Vec<String> = devs.iter().map(|d| d.short_name()).collect();
            Some(format!(
                "Plan: tensor parallel over {} (--tp {}). Radiance has no layer (pipeline) split.\n{}",
                names.join(" + "),
                devs.len(),
                self.v_compat_lines().join("\n")
            ))
        } else {
            None
        };
        let pane = &mut self.panes[p];
        pane.est_text = txt;
        pane.est_pct = pct;
        pane.est_color = if red { "#d0342c" } else if yellow { "#d08a00" } else { "#2e9e44" }.into();
        pane.est_over = red;
        if let Some(d) = dual_txt {
            pane.dual_info = d;
        }
    }

    // ================================================================ fit ==
    pub fn r_fit(&mut self, p: usize) {
        if self.panes[p].r_probe.is_some() {
            self.r_probe_stop(p);
            self.set_top_status(format!("{} pane: Fit's plan check cancelled", cap_side(self.panes[p].side)));
            return;
        }
        if self.panes[p].proc.is_some() {
            self.msg("info", "Stop the model first - Fit asks Radiance for its memory plan with the card free.");
            return;
        }
        let (Some(info), false) = (self.panes[p].info_r(), self.r_devs(p).is_empty()) else {
            self.msg("info", "Select a model (and assign a GPU) first.");
            return;
        };
        let s = self.r_current_options(p);
        if let Some(pl) = self.r_pool(p, &s) {
            if let Some(cap) = pool_capacity(&pl) {
                self.r_apply_fit(p, cap, &gs(&pl, "source", ""));
                return;
            }
            // recorded before LLM Duo read the budget table: that pool is only what was asked for
        }
        let ctx = if info.n_ctx_train > 0 { info.n_ctx_train } else { 0 };
        self.r_probe_start(p, ctx, None, None, 0);
    }

    fn r_apply_fit(&mut self, p: usize, tokens: i64, source: &str) {
        let s = self.r_current_options(p);
        let seqs = gi(&s, "parallel", 1).max(1);
        let cap = self.panes[p].info_r().map(|i| i.n_ctx_train).unwrap_or(0);
        if tokens < 1024 {
            self.msg("warning", &format!("Radiance's KV pool holds only {} tokens with these settings.\n\nTry: fp8 KV cache, a smaller Batch tokens or Max seqs, Vision tower → ram, or a Host pool (MoE models).", tokens));
            return;
        }
        let ctx = r_fit_ctx(tokens, seqs, cap);
        self.set_var_notify(p, "ctx", Value::from(ctx.to_string()));
        self.set_top_status(
            format!(
                "{} pane: Fit chose context {} (the card holds ≈ {} tokens of KV{}{})",
                cap_side(self.panes[p].side),
                fmt_thousands(ctx),
                fmt_thousands(tokens),
                if seqs > 1 { format!(" ÷ {seqs} seqs") } else { String::new() },
                if source == "probe" { ", from the plan check" } else { ", measured at load" }
            ) + if cap != 0 && ctx >= cap { "  •  the model's max" } else { "" },
        );
    }

    /// Start radiance with these settings and stop it once the startup log states the KV pool.
    fn r_probe_start(&mut self, p: usize, ctx: i64, lo: Option<i64>, hi: Option<i64>, tries: u32) {
        let g = self.settings.g().clone();
        if !expanduser(&gs(&g, "radiance_bin", "")).is_file() {
            self.msg("error", &format!("Radiance binary not found:\n{}\n\nSet it in Settings.", gs(&g, "radiance_bin", "")));
            return;
        }
        let (mut cmd, s) = match self.r_build_command(p, Some(ctx)) {
            Ok(x) => x,
            Err(e) => {
                self.msg("error", &format!("Could not build the command:\n{e}"));
                return;
            }
        };
        // a port of its own so a plan check never collides with a running server
        if let Some(i) = cmd.iter().position(|a| a == "--port") {
            cmd[i + 1] = (gi(&s, "port", 8080) + 37).to_string();
        }
        let env_extra = self.r_env(p);
        let mut env: Vec<(String, String)> = std::env::vars().collect();
        for (k, v) in &env_extra {
            env.retain(|e| &e.0 != k);
            env.push((k.clone(), v.clone()));
        }
        let (tx, rx) = channel();
        let pr = match proc::spawn(&cmd, &env, false, tx) {
            Ok(pr) => pr,
            Err(e) => {
                self.msg("error", &format!("Could not start radiance for the plan check:\n{e}"));
                return;
            }
        };
        let fp = self.r_pool_fp(p, &s);
        let rel = self.panes[p].model_var.clone();
        self.panes[p].r_probe = Some(Probe {
            proc: pr,
            rx,
            t0: Instant::now(),
            ctx,
            fp,
            rel,
            lines: vec![],
            kv: None,
            backs: None,
            sessions: None,
            killed: false,
            seqs: gi(&s, "parallel", 1).max(1),
            lo,
            hi,
            tries,
            budget: Budget::default(),
        });
        let shown = format!(
            "$ {} {}",
            env_text(&env_extra),
            shlex_join(&cmd)
        );
        self.append_log(p, &[format!("[LLM Duo] Fit: plan check (stopped as soon as Radiance states its KV pool){}", if ctx > 0 { format!(" at context {}", fmt_thousands(ctx)) } else { String::new() }), shown]);
        self.set_top_status(format!(
            "{} pane: Fit is asking Radiance for its memory plan… (press Fit again to cancel)",
            cap_side(self.panes[p].side)
        ));
        self.panes[p].state = "Checking plan".into();
    }

    fn r_probe_stop(&mut self, p: usize) {
        if let Some(pr) = self.panes[p].r_probe.as_mut() {
            if !pr.killed {
                pr.killed = true;
                proc::stop_group(pr.proc.pid, pr.proc.child.clone(), 3, false);
            }
        }
        self.panes[p].r_probe = None;
        if self.panes[p].proc.is_none() {
            self.panes[p].state = "Stopped".into();
        }
    }

    /// Kill any plan check (on quit).
    pub fn r_probe_kill_all(&mut self) {
        for p in 0..2 {
            if self.panes[p].r_probe.is_some() {
                self.r_probe_stop(p);
            }
        }
    }

    pub fn r_probe_poll(&mut self, p: usize) {
        let mut out_lines = vec![];
        let mut exit: Option<i32> = None;
        {
            let Some(pr) = self.panes[p].r_probe.as_mut() else { return };
            for _ in 0..2000 {
                let Ok(msg) = pr.rx.try_recv() else { break };
                match msg {
                    ProcMsg::Line(l) => {
                        if let Some(n) = kv_tokens_of(&l) {
                            pr.kv = Some(n);
                        }
                        pr.budget.line(&l);
                        if let Some(c) = rre().backs.captures(&l) {
                            pr.backs = c[1].parse().ok();
                            pr.sessions = c[2].parse().ok();
                        }
                        pr.lines.push(l.clone());
                        out_lines.push(l);
                    }
                    ProcMsg::Exit(_, rc) => exit = Some(rc),
                }
            }
        }
        if !out_lines.is_empty() {
            self.append_log(p, &out_lines);
        }
        let pr = self.panes[p].r_probe.as_ref().unwrap();
        let serving = pr.lines.iter().any(|l| rre().serving.is_match(l));
        // the pool line and the sessions line come together; wait a moment for the second
        // (a start that fails right after stating a pool did not work at that size)
        let failed = exit.map(|rc| rc != 0).unwrap_or(false);
        let got = !failed && pr.kv.is_some() && (pr.backs.is_some() || pr.t0.elapsed() > Duration::from_secs(4) || serving || exit.is_some());
        if got {
            let (kv, backs, sessions, fp, rel, ctx) = (pr.kv.unwrap(), pr.backs, pr.sessions, pr.fp.clone(), pr.rel.clone(), pr.ctx);
            let (seqs, hi, tries) = (pr.seqs, pr.hi, pr.tries);
            let capacity = pr.budget.capacity();
            self.r_probe_stop(p);
            // the pool is capped at Max seqs x context: a pool that big may hide a larger one.
            // The budget table says how much (its `unusable` line); without it, after a context
            // that did not fit, look between the two
            if let (true, Some(h), None) = (ctx > 0 && kv >= seqs * ctx, hi, capacity) {
                if h - ctx > 4096 && tries < 8 {
                    let mid = ((ctx + h) / 2 / 1024 * 1024).max(ctx + 1024);
                    self.append_log(p, &[format!("[LLM Duo] Fit: {} fits - trying {}", fmt_thousands(ctx), fmt_thousands(mid))]);
                    self.r_probe_start(p, mid, Some(ctx), Some(h), tries + 1);
                    return;
                }
            }
            let cap = capacity.unwrap_or(kv).max(kv);
            self.append_log(p, &[format!(
                "[LLM Duo] Fit: Radiance plans a KV pool of {} tokens{} - plan check stopped",
                fmt_thousands(kv),
                if cap > kv { format!("; the card could hold ≈ {}", fmt_thousands(cap)) } else { String::new() }
            )]);
            self.r_save_pool(
                p,
                &rel,
                &fp,
                json!({"kv_tokens": kv, "cap_tokens": cap, "backs": backs, "sessions": sessions, "ctx": ctx, "source": "probe", "when": now_iso()}),
            );
            if self.panes[p].model_var == rel {
                self.r_apply_fit(p, cap, "probe");
                self.update_estimate(p);
            }
            return;
        }
        let lines = pr.lines.clone();
        let (ctx, lo, tries) = (pr.ctx, pr.lo, pr.tries);
        if let Some(rc) = exit {
            self.panes[p].r_probe = None;
            self.panes[p].state = "Stopped".into();
            let hint = rad_failure_hint(&lines);
            let text = lines.join("\n");
            let didnt_fit = text.contains("DID NOT FIT") || text.to_lowercase().contains("did not fit");
            if didnt_fit && tries < 8 && (ctx == 0 || ctx > 4096) {
                let hi = if ctx == 0 { 1 << 20 } else { ctx };
                let next = match lo {
                    Some(l) => (l + hi) / 2 / 1024 * 1024,
                    None if ctx == 0 => 131072,
                    None => (ctx / 2 / 1024 * 1024).max(4096),
                };
                if lo.map(|l| next <= l).unwrap_or(false) || next >= hi {
                    // bisection done: the largest context that started
                    let l = lo.unwrap_or(next);
                    self.append_log(p, &[format!("[LLM Duo] Fit: largest context that starts ≈ {}", fmt_thousands(l))]);
                    self.r_probe_start(p, l, None, None, 9);
                    return;
                }
                self.append_log(p, &[format!("[LLM Duo] Fit: {} did not fit - trying {}", if ctx == 0 { "the training context".to_string() } else { fmt_thousands(ctx) }, fmt_thousands(next))]);
                self.r_probe_start(p, next, lo, Some(hi), tries + 1);
                return;
            }
            let tail: Vec<String> = lines.iter().rev().take(14).rev().cloned().collect();
            self.msg(
                "warning",
                &format!(
                    "Radiance's plan check ended (exit code {rc}) before it stated its KV pool.\n\n{}{}\n\nLast lines:\n{}",
                    if hint.is_empty() { String::new() } else { format!("Likely cause: {hint}\n\n") },
                    "Fit can still be done by hand: Load once and read 'KV pool' in Status - Fit reuses that figure afterwards.",
                    tail.join("\n")
                ),
            );
            self.set_top_status(format!("{} pane: Fit's plan check failed - see the log", cap_side(self.panes[p].side)));
            return;
        }
        let no_kv = pr.lines.iter().any(|l| rre().serving.is_match(l) && l.contains("completions and chat are 501"));
        if no_kv {
            self.r_probe_stop(p);
            self.msg("info", "This model keeps no KV cache, so Fit doesn't apply. Context only has to cover the longest input you will send.");
            return;
        }
        if serving || pr.t0.elapsed() > PROBE_TIMEOUT {
            self.r_probe_stop(p);
            self.msg(
                "warning",
                "Radiance started but its log never stated the KV pool in a form LLM Duo recognises.\n\nLoad the model once and look for the 'kv cache … tokens of context' line - and please send that startup log so the parser can be fixed.",
            );
        }
    }

    // ============================================================ command ==
    /// The radiance command line. `probe_ctx`: Fit's plan check at that context.
    pub fn r_build_command(&mut self, p: usize, probe_ctx: Option<i64>) -> Result<(Vec<String>, Opts), String> {
        let s = self.r_current_options(p);
        let path = self.model_path(p).ok_or("no model selected")?;
        let g = self.settings.g().clone();
        let devs = self.r_devs(p);
        if devs.is_empty() {
            return Err("no GPU assigned".into());
        }
        let name = {
            let n = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            n.strip_suffix(".rad").map(String::from).unwrap_or(n)
        };
        let mut cmd: Vec<String> = vec![
            expanduser(&gs(&g, "radiance_bin", "radiance")).to_string_lossy().into_owned(),
            "--model".into(),
            path.to_string_lossy().trim_end_matches('/').to_string(),
            "--served-model-name".into(),
            name,
            "--host".into(),
            gs(&g, "host", "127.0.0.1"),
            "--port".into(),
            gi(&s, "port", 8080).to_string(),
            "--max-model-len".into(),
            probe_ctx.unwrap_or_else(|| gi(&s, "ctx", 32768)).to_string(),
            "--kv-cache-dtype".into(),
            gs(&s, "r_kv", "fp8"),
            "--max-num-seqs".into(),
            gi(&s, "parallel", 1).to_string(),
            "--max-num-batched-tokens".into(),
            gi(&s, "r_batched", 2048).to_string(),
        ];
        let mut push = |k: &str, v: String| {
            cmd.push(k.to_string());
            cmd.push(v);
        };
        if gi(&s, "r_headroom", 96) != 96 {
            push("--gpu-headroom-mib", gi(&s, "r_headroom", 96).to_string());
        }
        for (flag, v) in spec_flags(&s) {
            push(flag, v);
        }
        if gs(&s, "r_draft_window", "fixed") == "adaptive" {
            push("--draft-window", "adaptive".into());
        }
        for (k, flag) in [("r_vision", "--vision-tower-placement"), ("r_embed", "--embedding-placement")] {
            let v = gs(&s, k, "(default)");
            if v != "(default)" {
                push(flag, v);
            }
        }
        let pl = gs(&s, "r_placement", "auto");
        if pl != "auto" {
            push("--placement", pl);
        }
        for (k, flag) in [("r_host_pool", "--host-pool-mib"), ("r_prefix_mib", "--prefix-cache-mib")] {
            if gi(&s, k, 0) > 0 {
                push(flag, gi(&s, k, 0).to_string());
            }
        }
        if gi(&s, "r_prefill_span", 1) > 1 {
            push("--prefill-span", gi(&s, "r_prefill_span", 1).to_string());
        }
        let re = gs(&s, "r_reasoning", "(default)");
        if re != "(default)" && !re.is_empty() {
            push("--reasoning-effort", re);
        }
        if devs.len() > 1 {
            push("--tp", devs.len().to_string());
            if gs(&s, "r_tp_wire", "exact") != "exact" {
                push("--tp-wire", gs(&s, "r_tp_wire", "exact"));
            }
        }
        let extra = gs(&s, "extra", "");
        if !extra.trim().is_empty() {
            cmd.extend(shlex_split(&extra)?);
        }
        Ok((cmd, s))
    }

    /// Environment for radiance: only on its own command line, never the user's shell
    /// (HIP_VISIBLE_DEVICES there would hide the other card from everything).
    pub fn r_env(&mut self, p: usize) -> Vec<(String, String)> {
        self.panes[p].k_env_notes.clear();
        let g = self.settings.g().clone();
        let devs = self.r_devs(p);
        let mut env: Vec<(String, String)> = vec![];
        let set = |env: &mut Vec<(String, String)>, k: &str, v: String| {
            if let Some(e) = env.iter_mut().find(|e| e.0 == k) {
                e.1 = v;
            } else {
                env.push((k.to_string(), v));
            }
        };
        let home = gs(&g, "radiance_home", "");
        if !home.trim().is_empty() {
            set(&mut env, "RADIANCE_HOME", expanduser(home.trim()).to_string_lossy().into_owned());
        }
        if !devs.is_empty() {
            set(&mut env, "HIP_VISIBLE_DEVICES", devs.iter().map(|d| d.index.to_string()).collect::<Vec<_>>().join(","));
        }
        let models = gs(&g, "radiance_models_dir", "");
        for part in shlex_split(&gs(&g, "radiance_env", "")).unwrap_or_default() {
            if let Some((k, v)) = part.split_once('=') {
                if !k.is_empty() {
                    set(&mut env, k, v.replace("{models}", &models));
                }
            }
        }
        if self.r_key_in_env(p) {
            if let Some((_, key)) = self.server_secret(p) {
                set(&mut env, RAD_KEY_ENV, key);
            }
        }
        self.panes[p].r_devs = devs;
        env
    }

    /// This pane's API key goes in the environment: Radiance 1.15.1 and later take it there.
    pub fn r_key_in_env(&self, p: usize) -> bool {
        self.panes[p].is_rad() && {
            let bin = expanduser(&gs(self.settings.g(), "radiance_bin", "radiance"));
            binary_has_flag(&bin, "--api-key-env")
        }
    }

    pub fn r_monitors(&self, p: usize) -> Vec<Mon> {
        self.r_devs(p)
            .into_iter()
            .map(|d| Mon {
                base: gpu_stats(Some(&d)).0,
                gtt: gpu_gtt(Some(&d)),
                dev: d,
                peak: 0.0,
                expected: 0.0,
                full: false,
                dv: None,
                dg: None,
                gtt_noted: 0.0,
            })
            .collect()
    }

    pub fn r_begin_load(&mut self, p: usize) {
        let pane = &mut self.panes[p];
        pane.r_kv_tokens = None;
        pane.r_sessions = None;
        pane.r_backs = None;
        pane.r_kv_bytes = None;
        pane.r_budget = Budget::default();
        pane.r_arena = None;
        pane.r_settled.clear();
        pane.r_stats = None;
        pane.r_stats_t = None;
        pane.r_401 = 0;
    }

    /// True for a log line to keep out of the log view: a repeated 401 for a missing / wrong API
    /// key from this machine or the local network. The first one of a run is shown, with a note.
    pub fn r_hide_line(&mut self, p: usize, line: &str) -> bool {
        let Some(host) = refused_401_from(line) else { return false };
        if !is_local_addr(&host) {
            return false;
        }
        self.panes[p].r_401 += 1;
        if self.panes[p].r_401 == 1 {
            self.panes[p].inject.push_back(
                "[LLM Duo] further 401 'invalid api key' lines from local addresses are hidden this run (Radiance's chat page sends some requests without the key)".into(),
            );
            return false;
        }
        true
    }

    // =========================================================== the log ==
    pub fn r_parse_line(&mut self, p: usize, line: &str) {
        let re = rre();
        let mut shown = self.panes[p].r_budget.line(line);
        if let Some(n) = kv_tokens_of(line) {
            self.panes[p].r_kv_tokens = Some(n);
            if let Some(c) = re.kv_gib.captures(line) {
                self.panes[p].r_kv_bytes = Some(gib_of(&c[1], &c[2]));
            }
            shown = true;
        }
        if let Some(c) = re.backs.captures(line) {
            self.panes[p].r_backs = c[1].parse().ok();
            self.panes[p].r_sessions = c[2].parse().ok();
            shown = true;
        }
        if let Some(c) = re.arena.captures(line) {
            if self.panes[p].r_arena.is_none() {
                self.panes[p].r_arena = Some(gib_of(&c[1], &c[2]));
                shown = true;
            }
        }
        if let Some(c) = re.settled.captures(line) {
            let card: i64 = c[1].parse().unwrap_or(0);
            let free = gib_of(&c[2], &c[3]);
            let st = &mut self.panes[p].r_settled;
            st.retain(|x| x.0 != card);
            st.push((card, free));
            shown = true;
        }
        if shown {
            self.r_show_buffers(p);
        }
        if re.chat501.is_match(line) {
            self.panes[p].inject.push_back(
                "[LLM Duo] WARNING: this container carries no chat template, so /v1/chat/completions answers 501. Use /v1/completions, or re-convert with the checkpoint's chat_template.jinja beside tokenizer_config.json.".into(),
            );
        }
        if re.refused.is_match(line) {
            self.set_top_status(format!(
                "{} pane: a request's prompt did not fit the context - raise Context (Fit) or shorten the prompt",
                cap_side(self.panes[p].side)
            ));
        }
        if re.serving.is_match(line) && self.panes[p].state == "Loading" {
            self.became_ready(p);
        }
    }

    pub fn r_show_buffers(&mut self, p: usize) {
        let pane = &mut self.panes[p];
        let mut parts = vec![];
        if let Some(t) = pane.r_kv_tokens {
            let mut s = format!("KV pool {} tokens", fmt_thousands(t));
            if let Some(b) = pane.r_kv_bytes {
                s += &format!(" ({:.2} GiB)", b / GIB);
            }
            if let Some(c) = pane.r_budget.capacity().filter(|c| *c as f64 > t as f64 * 1.05) {
                s += &format!(", the card could hold ≈ {}", fmt_thousands(c));
                if let Some(u) = pane.r_budget.unusable {
                    s += &format!(" ({:.1} GiB unused: raise Context or Max seqs)", u / GIB);
                }
            }
            parts.push(s);
        }
        if let Some(sx) = pane.r_sessions {
            parts.push(format!("{sx:.1} full-length sessions") + &pane.r_backs.map(|b| format!(" ({b:.0}% of the worst case)")).unwrap_or_default());
        }
        if let Some(a) = pane.r_arena {
            parts.push(format!("activation arena {:.2} GiB", a / GIB));
        }
        if !pane.r_settled.is_empty() {
            let mut st = pane.r_settled.clone();
            st.sort_by_key(|x| x.0);
            parts.push(st.iter().map(|(c, f)| format!("card {c} settled at {:.0} MiB free", f / MIB)).collect::<Vec<_>>().join(", "));
        }
        if !parts.is_empty() {
            pane.buffers = vec![("radiance".into(), 1.0)];
            pane.set_status("buffers", parts.join("   •   "));
        }
    }

    // ============================================================ metrics ==
    pub fn r_metrics(&mut self, p: usize, m: HashMap<String, f64>) {
        let g = |k: &str| m.get(k).cloned().unwrap_or(0.0);
        let has = |k: &str| m.contains_key(k);
        let running = g("vllm:num_requests_running");
        let waiting = g("vllm:num_requests_waiting");
        self.dash_metrics(p, &m);
        if matches!(self.panes[p].state.as_str(), "Ready" | "Processing") {
            self.set_state(p, if running > 0.0 || waiting > 0.0 { "Processing" } else { "Ready" });
        }
        // ---- status: averages since start
        let mut avg = vec![];
        let steps_d = g("radiance:engine_steps_decode_total");
        if steps_d > 0.0 && has("radiance:engine_step_seconds_decode_total") {
            avg.push(format!("decode step {:.1} ms", 1000.0 * g("radiance:engine_step_seconds_decode_total") / steps_d));
        }
        if let Some(st) = &self.panes[p].r_stats {
            let ns = |k: &str| st.get(k).and_then(|v| v.as_f64()).filter(|x| *x > 0.0);
            if let (Some(b), r) = (ns("step_ns_decode_best"), ns("step_ns_decode_recent")) {
                avg.push(format!("best {:.2} ms", b / 1e6) + &r.map(|r| format!(" · recent {:.2}", r / 1e6)).unwrap_or_default());
            }
        }
        let (pt, gt) = (g("vllm:prompt_tokens_total"), g("vllm:generation_tokens_total"));
        if pt != 0.0 || gt != 0.0 {
            avg.push(format!("{} prompt / {} generated tokens total", fmt_thousands(pt as i64), fmt_thousands(gt as i64)));
        }
        if let Some(kvu) = m.get("vllm:kv_cache_usage_perc").or(m.get("vllm:gpu_cache_usage_perc")) {
            avg.push(format!("KV in use {:.0}%", kvu * 100.0));
        }
        if g("radiance:draft_tokens_total") > 0.0 {
            avg.push(format!("draft acceptance {:.0}%", 100.0 * g("radiance:draft_accepted_total") / g("radiance:draft_tokens_total")));
        }
        if g("radiance:lookup_proposals_total") > 0.0 {
            avg.push(format!(
                "prompt lookup {} proposals, {} held back",
                fmt_thousands(g("radiance:lookup_proposals_total") as i64),
                fmt_thousands(g("radiance:lookup_held_total") as i64)
            ));
        }
        if g("radiance:num_requests_preempted") > 0.0 {
            avg.push(format!("{} preempted", g("radiance:num_requests_preempted") as i64));
        }
        if g("radiance:preempt_forced_total") > 0.0 {
            avg.push(format!("{} forced preemptions", g("radiance:preempt_forced_total") as i64));
        }
        self.panes[p].set_status("avg", if avg.is_empty() { "—".into() } else { avg.join("   •   ") });

        let now = unix_now();
        let busy = running > 0.0 || waiting > 0.0;
        if self.panes[p].metrics_prev.is_none() {
            self.panes[p].metrics_prev = Some(m.clone());
            return;
        }
        let prev = self.panes[p].metrics_prev.clone().unwrap();
        let d = |k: &str| g(k) - prev.get(k).cloned().unwrap_or(0.0);
        let gen_key = if has("vllm:generation_tokens_total") { "vllm:generation_tokens_total" } else { "radiance:decode_tokens_total" };
        if busy {
            // request in flight (live dashboard): tokens since the last idle snapshot
            let ctx = self.panes[p].loaded.as_ref().map(|(_, o, _)| gi(o, "ctx", 0)).filter(|c| *c > 0);
            if self.panes[p].dash_active.is_none() {
                self.panes[p].dash_active = Some((now, None, ctx));
                self.panes[p].dash_progress = Some(0.0);
            }
            let dg = d(gen_key);
            if dg > 0.0 {
                self.panes[p].dash_progress = Some(1.0);
                self.panes[p].dash_gen = Some((dg as i64, g("radiance:decode_tokens_per_second")));
            } else {
                let computed = d("radiance:prefill_tokens_total");
                if let (Some((t0, _, c)), true) = (self.panes[p].dash_active, computed > 0.0) {
                    self.panes[p].dash_active = Some((t0, Some(computed as i64), c));
                }
            }
            return;
        }
        self.panes[p].metrics_prev = Some(m.clone());
        let dreq = d("vllm:request_success_total");
        let ddis = d("radiance:client_disconnect_total");
        let dg = d(gen_key).max(0.0);
        let dcached = d("radiance:attn_cached_tokens_total").max(0.0);
        let dcomp = if has("radiance:prefill_tokens_total") {
            d("radiance:prefill_tokens_total").max(0.0)
        } else {
            (d("vllm:prompt_tokens_total") - dcached).max(0.0)
        };
        let total = d("vllm:prompt_tokens_total").max(dcomp + dcached);
        let step_all = d("radiance:engine_step_seconds_total");
        let step_dec = d("radiance:engine_step_seconds_decode_total");
        let mut p_s = (step_all - step_dec).max(0.0);
        if p_s <= 0.0 {
            p_s = d("vllm:request_prefill_time_seconds_sum").max(d("vllm:time_to_first_token_seconds_sum")).max(0.0);
        }
        let mut g_s = step_dec.max(0.0);
        if g_s <= 0.0 {
            g_s = d("vllm:request_decode_time_seconds_sum").max(0.0);
        }
        let had_active = self.panes[p].dash_active.is_some();
        self.panes[p].dash_gen = None;
        if dg <= 0.0 && dreq <= 0.0 {
            if ddis > 0.0 || had_active {
                // the client went away before anything was generated
                if ddis > 0.0 {
                    self.stats_bump(p, "aborts");
                    let mut r = Map::new();
                    r.insert("p_tok".into(), json!(dcomp as i64));
                    r.insert("p_ms".into(), json!(p_s * 1000.0));
                    self.dash_request(p, &r, None, true);
                } else {
                    self.panes[p].dash_active = None;
                    self.panes[p].dash_progress = None;
                }
            }
            return;
        }
        let mut r = Map::new();
        r.insert("p_ms".into(), json!(p_s * 1000.0));
        r.insert("p_tok".into(), json!(dcomp as i64));
        r.insert("g_ms".into(), json!(g_s * 1000.0));
        r.insert("g_tok".into(), json!(dg as i64));
        r.insert("n_end".into(), json!((total + dg) as i64));
        let ddraft = d("radiance:draft_tokens_total");
        if ddraft > 0.0 {
            let acc = d("radiance:draft_accepted_total").max(0.0);
            self.panes[p].last_accept = Some(acc / ddraft);
            r.insert("acc".into(), json!(acc as i64));
            r.insert("gen".into(), json!(ddraft as i64));
        }
        if dcomp > 0.0 && p_s > 0.0 {
            self.panes[p].last_prompt = Some((dcomp as i64, dcomp / p_s));
        }
        if dg > 0.0 && g_s > 0.0 {
            self.panes[p].last_gen = Some((dg as i64, dg / g_s));
        }
        self.update_speed(p);
        if ddis > 0.0 && dreq <= 0.0 {
            // generated part of a reply, then the client disconnected
            self.stats_bump(p, "aborts");
            let cache = if total > 0.0 { Some((dcached as i64, total as i64)) } else { None };
            self.dash_request(p, &r, cache, true);
        } else {
            self.stats_record(p, &r);
        }
    }

    // ======================================================== calibration ==
    /// After a load: remember the KV pool the engine made, for the estimate and Fit.
    pub fn r_record_calibration(&mut self, p: usize) {
        if self.panes[p].proc.is_none() || !matches!(self.panes[p].state.as_str(), "Ready" | "Processing") {
            return;
        }
        let (Some(kv), Some((rel, opts, _))) = (self.panes[p].r_kv_tokens, self.panes[p].loaded.clone()) else { return };
        let fp = self.r_pool_fp(p, &opts);
        let pane = &self.panes[p];
        let data = json!({
            "kv_tokens": kv, "cap_tokens": pane.r_budget.capacity().map(|c| c.max(kv)), "sessions": pane.r_sessions, "backs": pane.r_backs,
            "kv_gib": pane.r_kv_bytes.map(|b| (b / GIB * 100.0).round() / 100.0),
            "arena_gib": pane.r_arena.map(|b| (b / GIB * 100.0).round() / 100.0),
            "ctx": gi(&opts, "ctx", 0), "source": "load", "when": now_iso(),
        });
        self.r_save_pool(p, &rel, &fp, data);
        if self.panes[p].model_var == rel {
            self.update_estimate(p);
        }
    }
}

/// The speculation flags. `--spec-ngram` sets the window itself: another value beside it is refused.
fn spec_flags(s: &Opts) -> Vec<(&'static str, String)> {
    let mut out = vec![];
    let spec = gs(s, "r_spec", "auto");
    if gi(s, "r_spec_ngram", 0) > 0 {
        out.push(("--spec-ngram", gi(s, "r_spec_ngram", 0).to_string()));
    } else if spec != "auto" {
        out.push(("--num-speculative-tokens", spec));
    }
    if gi(s, "r_ngram_min", 2) != 2 {
        out.push(("--spec-ngram-min", gi(s, "r_ngram_min", 2).to_string()));
    }
    if gi(s, "r_ngram_max", 4) != 4 {
        out.push(("--spec-ngram-max", gi(s, "r_ngram_max", 4).to_string()));
    }
    out
}

/// What a recorded plan says the card could hold: the budget table's figure, or the pool itself
/// when the card (not Context × Max seqs) limited it. None: the pool was only what was asked for.
pub fn pool_capacity(pl: &Opts) -> Option<i64> {
    if let Some(c) = pl.get("cap_tokens").and_then(|v| v.as_i64()) {
        return Some(c);
    }
    let kv = gi(pl, "kv_tokens", 0);
    let backs = pl.get("backs").and_then(|v| v.as_f64()).unwrap_or(100.0);
    if kv > 0 && backs < 99.5 { Some(kv) } else { None }
}

/// Largest context for `seqs` sequences in a pool of `tokens`, capped at the training length.
pub fn r_fit_ctx(tokens: i64, seqs: i64, cap: i64) -> i64 {
    let mut m = tokens / seqs.max(1);
    if cap > 0 {
        m = m.min(cap);
    }
    if m >= 2048 { m / 1024 * 1024 } else { m / 256 * 256 }
}

#[allow(dead_code)]
fn _p(_: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spec() {
        let o = |v: Value| v.as_object().unwrap().clone();
        assert!(spec_flags(&o(json!({"r_spec": "auto", "r_spec_ngram": 0, "r_ngram_min": 2, "r_ngram_max": 4}))).is_empty());
        assert_eq!(spec_flags(&o(json!({"r_spec": "4"}))), vec![("--num-speculative-tokens", "4".to_string())]);
        assert_eq!(spec_flags(&o(json!({"r_spec": "4", "r_spec_ngram": 5}))), vec![("--spec-ngram", "5".to_string())]);
        assert_eq!(
            spec_flags(&o(json!({"r_spec": "4", "r_ngram_min": 3, "r_ngram_max": 8}))),
            vec![("--num-speculative-tokens", "4".to_string()), ("--spec-ngram-min", "3".to_string()), ("--spec-ngram-max", "8".to_string())]
        );
    }

    #[test]
    fn fit_ctx() {
        assert_eq!(r_fit_ctx(262160, 1, 262144), 262144);
        assert_eq!(r_fit_ctx(262160, 2, 262144), 131072);
        assert_eq!(r_fit_ctx(100_000, 1, 0), 99328);
        assert_eq!(r_fit_ctx(1500, 1, 0), 1280);
    }
}
