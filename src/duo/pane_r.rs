//! Everything a pane does differently when its backend is Radiance.
//!
//! Radiance sizes memory itself (see radiance.rs), so the "estimate" here is what the engine
//! reported for the same memory settings - from a real load or from Fit's plan check, a short
//! start of the server that is stopped once the startup log states the KV pool and every card
//! has passed weight placement.

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
    /// cards in the run, and how many have passed placement ("load: rank N, mover took")
    pub ranks: usize,
    pub placed: usize,
    /// Radiance's warning that the experts do not fit beside one full-length sequence
    pub short: Option<(f64, f64, f64)>,
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(300);

/// Shown before Load when Radiance 2.0 would accept the settings and then fail the first request.
pub const TREE_I8_WARNING: &str = "RADIANCE 2.0 WILL FAIL THIS MODEL ON ITS FIRST REQUEST.\n\n\
This container's DFlash2 drafter is verified as a draft tree (Gemma 4 + DFlash2), and Radiance 2.0's \
tree code cannot move rows of the i8 KV cache (516-byte rows; it needs a multiple of 16). It loads and \
reports ready, then stops with \"KV group N's cache (i8, 516-byte rows) is not one a tree's acceptance \
can move\" and refuses every request after it.\n\n\
Radiance 2.1.0 fixes this (#454): update Radiance (~/radiance/update.sh). Until then, settings that \
run: Spec tokens 0 (i8 kept, no drafter), or KV cache bf16 (drafter kept, about half the context). \
LLM Duo changes neither for you.";

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
        o.insert("r_kv".into(), list_or(v.var_s("r_kv"), &RAD_KV_TYPES, "i8").into());
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
        o.insert("r_ngram_place".into(), list_or(v.var_s("r_ngram_place"), &RAD_NGRAM, "(default)").into());
        o.insert("r_placement".into(), list_or(v.var_s("r_placement"), &RAD_PLACEMENT, "auto").into());
        o.insert("r_host_pool".into(), v.int("r_host_pool", 0).max(0).into());
        o.insert("r_prefix_mib".into(), v.int("r_prefix_mib", 0).max(0).into());
        o.insert("r_prefill_span".into(), v.int("r_prefill_span", 1).max(1).into());
        o.insert("r_reasoning".into(), Value::from(if reasoning.is_empty() { "(default)".to_string() } else { reasoning }));
        o.insert("r_tp_wire".into(), list_or(v.var_s("r_tp_wire"), &RAD_TP_WIRE, "exact").into());
        o.insert("r_cvec".into(), list_or(v.var_s("r_cvec"), &RAD_CVEC, "disabled").into());
        o.insert("r_state_dtype".into(), list_or(v.var_s("r_state_dtype"), &RAD_STATE_DTYPE, "(default)").into());
        o.insert("r_kv_resident".into(), v.int("r_kv_resident", 0).max(0).into());
        o.insert("r_weight_prefetch".into(), list_or(v.var_s("r_weight_prefetch"), &RAD_PREFETCH, "on").into());
        o.insert("r_disk_tier".into(), v.var_b("r_disk_tier").into());
        o.insert("r_gen_auto".into(), v.var_b("r_gen_auto").into());
        o.insert("r_tp_attn_split".into(), list_or(v.var_s("r_tp_attn_split"), &RAD_ATTN_SPLIT, "heads").into());
        o.insert("r_p2p".into(), list_or(v.var_s("r_p2p"), &RAD_P2P, "auto").into());
        for t in RAD_TEXT_OPTS.iter() {
            o.insert(t.key.into(), Value::from(v.var_s(t.key).trim().to_string()));
        }
        o.insert("dual_strategy".into(), RAD_STRATEGIES[0].into());
        o.insert("dual_mode".into(), "tensor".into());
        o.insert("port".into(), v.int("port", self.default_port(p)).into());
        o.insert("extra".into(), Value::from(v.var_s("extra")));
        o
    }

    /// The pre-Load warning for the tree-drafter + i8 combination (r_tree_blocks_i8).
    pub fn r_tree_note(&self) -> &'static str {
        TREE_I8_WARNING
    }

    /// This container's drafter is verified as a tree (Gemma 4 + DFlash2), speculation is on and
    /// the i8 cache is asked for: a combination Radiance 2.0 accepts at startup and then fails on
    /// the first request ("KV group N's cache (i8, 516-byte rows) is not one a tree's acceptance
    /// can move"). Speculation off (Spec tokens 0) makes i8 safe again.
    pub fn r_tree_blocks_i8(&self, p: usize, o: &Opts) -> bool {
        let tree = self.panes[p].info_r().map(|i| i.tree_drafter).unwrap_or(false);
        if !tree || gs(o, "r_kv", "i8") != "i8" {
            return false;
        }
        // fixed in Radiance 2.1.0 (#454)
        if tree_i8_fixed(&expanduser(&gs(self.settings.g(), "radiance_bin", "radiance"))) {
            return false;
        }
        let extra = shlex_split(&gs(o, "extra", "")).unwrap_or_default();
        let spec_off_extra = extra.windows(2).any(|w| w[0] == "--num-speculative-tokens" && w[1] == "0")
            || extra.iter().any(|a| a == "--num-speculative-tokens=0");
        gs(o, "r_spec", "auto") != "0" && !spec_off_extra
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
        let mut vals: Vec<String> = RAD_POOL_KEYS.iter().map(|k| opt_str(s.get(*k))).collect();
        // the n-gram table's tier moves VRAM too; appended only when set, so plans recorded
        // before the option existed (all at the default) still match
        let ng = gs(s, "r_ngram_place", "(default)");
        if ng != "(default)" {
            vals.push(format!("ngram={ng}"));
        }
        // the same for the Advanced options that move memory
        let st = gs(s, "r_state_dtype", "(default)");
        if st != "(default)" {
            vals.push(format!("state={st}"));
        }
        if gi(s, "r_kv_resident", 0) > 0 {
            vals.push(format!("kvres={}", gi(s, "r_kv_resident", 0)));
        }
        if gs(s, "r_tp_attn_split", "heads") == "rows" {
            vals.push("attn=rows".into());
        }
        if gb(s, "r_disk_tier", false) {
            vals.push("disktier".into());
        }
        format!("{}|{}", names.join("+"), vals.join(","))
    }

    /// The KV pool recorded for this model and these memory settings.
    pub fn r_pool(&self, p: usize, s: &Opts) -> Option<Opts> {
        let rel = &self.panes[p].model_var;
        let rec = self.settings.get_calib(&self.dev_key(p), rel)?;
        let fp = self.r_pool_fp(p, s);
        rec.get("pools")
            .and_then(|v| v.get(&fp))
            .and_then(|v| v.as_object())
            // a plan check recorded before it waited for placement may hold a pool for weights
            // that never fit (it stopped at the KV line)
            .filter(|o| gs(o, "source", "") != "probe" || gb(o, "placed", false))
            .cloned()
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

    // ======================================================= prefix cache ==
    /// Remember what a token of context costs in the host tier, as the running engine said, for
    /// this model, these cards and this KV cache type.
    fn r_save_prefix_bpt(&mut self, p: usize, bpt: f64) {
        let Some((rel, opts, _)) = self.panes[p].loaded.clone() else { return };
        let key = self.dev_key(p);
        let mut rec = self.settings.get_calib(&key, &rel).unwrap_or_default();
        rec.insert("radiance".into(), true.into());
        let mut m = rec.get("prefix_bpt").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        m.insert(gs(&opts, "r_kv", "i8"), json!(bpt.round()));
        rec.insert("prefix_bpt".into(), Value::Object(m));
        self.settings.set_calib(&key, &rel, rec);
    }

    /// What the prefix-cache calculator needs: the host-tier cost of a token of context for the
    /// selected model / cards / KV cache type, where that figure came from, the context a
    /// conversation can reach, and the system memory it would be pinned from.
    pub fn r_prefix_calc(&self, p: usize) -> Value {
        let s = self.r_current_options(p);
        let kv = gs(&s, "r_kv", "i8");
        let ranks = self.r_devs(p).len().max(1) as f64;
        let info = self.panes[p].info_r();
        let ctx = match gi(&s, "ctx", 0) {
            0 => info.as_ref().map(|i| i.n_ctx_train).unwrap_or(0),
            c => c,
        };
        let rel = &self.panes[p].model_var;
        let rec = self.settings.get_calib(&self.dev_key(p), rel).unwrap_or_default();
        // relative bytes a token of each cache type, for carrying a figure across types
        let w = |t: &str| match t {
            "bf16" => 2.0,
            "fp8" | "i8" => 1.0,
            "fp8_v4" => 0.75,
            "k6_v4" => 0.625,
            _ => 1.0,
        };
        let mut bpt = None;
        let mut source = String::new();
        let mut exact = false;
        let measured = rec.get("prefix_bpt").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        if let Some(b) = measured.get(&kv).and_then(|v| v.as_f64()) {
            bpt = Some(b);
            exact = true;
            source = format!("measured: Radiance's own kv tiers line for this model at KV cache {kv}");
        } else if let Some((t, b)) = measured.iter().find_map(|(t, v)| v.as_f64().map(|b| (t.clone(), b))) {
            bpt = Some(b * w(&kv) / w(&t));
            source = format!("scaled from Radiance's kv tiers line at KV cache {t} (approximate)");
        }
        if bpt.is_none() {
            // the KV pool a load built: GiB a card over tokens, times the cards (one host slot
            // holds every rank's shard)
            let pools = rec.get("pools").and_then(|v| v.as_object()).cloned().unwrap_or_default();
            let mut best: Option<(bool, String, f64)> = None;
            for (fp, v) in &pools {
                let Some(o) = v.as_object() else { continue };
                let (Some(g), t) = (o.get("kv_gib").and_then(|v| v.as_f64()), gi(o, "kv_tokens", 0)) else { continue };
                if t <= 0 || g <= 0.0 {
                    continue;
                }
                let pkv = fp.split('|').nth(1).and_then(|x| x.split(',').next()).unwrap_or("i8").to_string();
                let b = g * GIB / t as f64 * ranks * w(&kv) / w(&pkv);
                let same = pkv == kv;
                if best.as_ref().map(|x| same && !x.0).unwrap_or(true) {
                    best = Some((same, pkv, b));
                }
            }
            if let Some((same, pkv, b)) = best {
                bpt = Some(b);
                source = if same {
                    format!("estimated from the KV pool a load built at KV cache {kv} (a load with Prefix cache on measures it exactly)")
                } else {
                    format!("estimated from a load at KV cache {pkv}, scaled to {kv} (approximate)")
                };
            }
        }
        // pinned host memory already asked for: since 2.0 the host pool is the model's total,
        // split among its cards (1.x pinned it on every card)
        let host_pool = gi(&s, "r_host_pool", 0) as f64 * MIB * if self.r_config_mode() { 1.0 } else { ranks };
        json!({
            "ok": bpt.is_some(),
            "bpt": bpt.unwrap_or(0.0),
            "exact": exact,
            "source": if bpt.is_some() { source } else {
                "no measurement yet for this model on these cards: Load it once (better with Prefix cache set to any size, \
                 e.g. 1024, so Radiance states the exact figure), then open this again".to_string()
            },
            "ctx": ctx,
            "seqs": gi(&s, "parallel", 1).max(1),
            "kv": kv,
            "ranks": ranks as i64,
            "current": gi(&s, "r_prefix_mib", 0),
            "key": "r_prefix_mib",
            "flag": "--prefix-cache-mib",
            "pinned": true,
            "otherPinned": host_pool,
            "otherPinnedName": "Host pool",
            "what": "Radiance copies a finished conversation's KV to pinned system RAM, so returning to it restores in a \
                     fraction of a second instead of re-reading the whole prompt. A tier smaller than a conversation keeps only its start.",
        })
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
        let ngram = gs(&s, "r_ngram_place", "(default)");
        let card_min = devs.iter().map(|d| Self::v_total(d)).fold(f64::INFINITY, f64::min);
        let table = info.ngram_bytes;
        // vram puts the whole table on every card; Radiance's own choice (auto) takes vram only
        // when it fits beside the other weights
        let table_on_card = ngram_on_card(&ngram, info.bytes, table, tp, head, card_min);
        // the cards' total: the host pool stays in the pane's own `offload` handling below
        let weights = card_weight_bytes(info.bytes, table, table_on_card, 0.0, tp) * tp;
        for d in &devs {
            let total = Self::v_total(d);
            tight_total = tight_total.min(total);
            // a plan check (running, or stopped and still exiting) holds VRAM the load will have
            let probing = self.panes[p].r_probe.is_some() || self.panes[p].r_probe_dying.is_some();
            let free = if self.panes[p].proc.is_some() || probing { total } else { self.v_free(p, d) };
            let w = weights / tp;
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
        // YaRN (Advanced → RoPE scaling) stretches the trained length: FACTOR × ORIG (default the
        // container's own)
        let reach = match rope_spec(&gs(&s, "r_rope", "")) {
            Ok(Some((f, orig))) => Some((orig.unwrap_or(cap) as f64 * f) as i64),
            _ => None,
        };
        if cap != 0 && ctx > cap {
            match reach {
                Some(r) if r >= ctx => lines.push(format!(
                    "Context beyond the trained {} through YaRN (reaches {}); measure quality on your own work",
                    fmt_thousands(cap),
                    fmt_thousands(r)
                )),
                Some(r) => {
                    warn.push(format!(
                        "Context exceeds what RoPE scaling reaches ({}): raise its FACTOR or lower Context",
                        fmt_thousands(r)
                    ));
                    yellow = true;
                }
                None => {
                    warn.push(format!(
                        "Context exceeds the model's training length ({}) - set Advanced → RoPE scaling to yarn:FACTOR to go past it",
                        fmt_thousands(cap)
                    ));
                    yellow = true;
                }
            }
        }
        // Advanced options a plugin may not serve: Radiance refuses the start by name, so say it now
        let arch = info.arch.to_lowercase();
        if gs(&s, "r_cvec", "disabled") != "disabled" && !arch.starts_with("qwen4exp") {
            warn.push(format!(
                "Control vector: only Qwen3.8-Flash-Next (qwen4exp) applies one; Radiance refuses to start this {} container with it - set it to disabled",
                if arch.is_empty() { "unknown" } else { arch.as_str() }
            ));
            red = true;
        }
        if gs(&s, "r_state_dtype", "(default)") == "f16" && !arch.starts_with("qwen35") {
            warn.push("State cache f16: only the qwen35 plugins (Qwen3.8-27B and kin) serve it; Radiance refuses the start elsewhere".into());
            red = true;
        }
        if gs(&s, "r_tp_attn_split", "heads") == "rows" && tp > 1.0 && !arch.starts_with("glm") {
            warn.push("Attention split rows: only GLM-5.3's plugin declares the row split; Radiance refuses the start for this model".into());
            red = true;
        }
        let kvres = gi(&s, "r_kv_resident", 0);
        if kvres > 0 {
            if !(arch.starts_with("qwen4exp") || arch.starts_with("glm")) {
                lines.push("KV resident: this model may have no sparse attention group to stream; Radiance then ignores it with a warning".into());
            }
            if kvres < 32768 {
                lines.push(format!("KV resident {}: Radiance raises it to 32 768 tokens", fmt_thousands(kvres)));
            }
            if gi(&s, "r_prefix_mib", 0) > 0 {
                warn.push("Prefix cache is off while a KV group streams (KV resident), so its host memory goes unused".into());
                yellow = true;
            }
        }
        if gb(&s, "r_disk_tier", false) {
            lines.push("Weights disk tier: what fits neither the cards nor the Host pool is read from the container each step, paced by the drive".into());
        }
        let bad: Vec<String> = RAD_TEXT_OPTS.iter().filter_map(|o| text_opt_problem(o, &gs(&s, o.key, ""))).collect();
        if !bad.is_empty() {
            warn.push(format!("{} - Radiance refuses the start", bad.join("; ")));
            red = true;
        }
        if ctx == 0 {
            lines.push(format!("Context 0 = the model's own training context{}", if cap != 0 { format!(" ({})", fmt_thousands(cap)) } else { String::new() }));
        }
        if info.vision && gs(&s, "r_vision", "(default)") != "ram" {
            lines.push("vision model: Vision tower → ram moves the encoder to system RAM and gives its VRAM to the KV pool (≈55 ms per image)".into());
        }
        if table > 0.0 {
            let tgib = table / GIB;
            lines.push(format!(
                "n-gram table {tgib:.1} GiB → {}",
                match ngram.as_str() {
                    "vram" => "on every card".to_string(),
                    "ram" => "pinned in system RAM, read by the cards over PCIe".to_string(),
                    "disk" => "read from the container as it decodes (the drive paces each step)".to_string(),
                    _ if table_on_card => "Radiance's choice: on the cards".to_string(),
                    _ => "Radiance's choice: pinned RAM if it seems to fit, else read from disk".to_string(),
                }
            ));
            let host_pool = gi(&s, "r_host_pool", 0) as f64 * MIB * if self.r_config_mode() { 1.0 } else { tp };
            let pinned = host_pool + gi(&s, "r_prefix_mib", 0) as f64 * MIB;
            match ngram.as_str() {
                "vram" if table + head > card_min => {
                    warn.push(format!(
                        "N-gram table vram puts all {tgib:.1} GiB of the table on every card, more than a card holds: choose ram or disk"
                    ));
                    red = true;
                }
                "ram" => {
                    let (_, avail) = super::prefix::meminfo();
                    let need = table + pinned + 4.0 * GIB;
                    if avail > 0.0 && need > avail {
                        warn.push(format!(
                            "N-gram table ram needs the table ({tgib:.1} GiB) free beside the Host pool and Prefix cache, plus 4 GiB to spare: {:.1} GiB, and {:.1} GiB of RAM is available now. Radiance refuses the start; lower Host pool or choose disk",
                            need / GIB,
                            avail / GIB
                        ));
                        red = true;
                    }
                }
                "(default)" if !table_on_card && offload => {
                    warn.push(format!(
                        "N-gram table (default) lets Radiance pin the {tgib:.1} GiB table in RAM, which can leave the Host pool too little pinned memory (startup then stops with out of memory). Choose disk, or ram if RAM holds both"
                    ));
                    yellow = true;
                }
                _ => {}
            }
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
        if let Some(gst) = info.gen {
            let auto = gb(&s, "r_gen_auto", false);
            if gst.is_image() {
                let side = gst.max_side();
                lines.push(format!(
                    "image generator: the largest image is about {side}x{side} ({} latent tokens); a step needs {} rows for it with a full prompt, {} with guidance (two branches)",
                    fmt_thousands(gst.max_tokens),
                    fmt_thousands(gst.rows(1)),
                    fmt_thousands(gst.rows(2))
                ));
            }
            if auto {
                lines.push(format!(
                    "Radiance sizes it: Batch tokens{}, Max seqs and Context are left to the plugin (its startup report lists them); the activation arena grows with the step",
                    if gst.auto_rows > 0 { format!(" ({} rows)", fmt_thousands(gst.auto_rows)) } else { String::new() }
                ));
            }
            if let Some((r, t)) = gen_batch_problem(gst, gi(&s, "r_batched", 2048), auto) {
                warn.push(t);
                yellow = true;
                red |= r;
            }
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
            None => 100.0 * ((weights / tp + head) / tight_total.max(1.0)).min(1.0),
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
        if self.panes[p].r_probe.is_some() || self.panes[p].r_probe_next.is_some() {
            self.panes[p].r_probe_next = None;
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
            self.msg("warning", &format!("Radiance's KV pool holds only {} tokens with these settings.\n\nTry: the i8 KV cache, a smaller Batch tokens or Max seqs, Vision tower → ram, or a Host pool (MoE models).", tokens));
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
        // the last plan check still holds its VRAM until it exits, and Radiance budgets only
        // what is free when it starts
        if self.r_probe_releasing(p) {
            self.panes[p].r_probe_next = Some((ctx, lo, hi, tries));
            self.panes[p].state = "Checking plan".into();
            return;
        }
        self.panes[p].r_probe_next = None;
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
        let (cmd, yaml) = match self.r_launch_cmd(&cmd, &env, &format!("{}-plan", self.panes[p].side)) {
            Ok(x) => x,
            Err(e) => {
                self.msg("error", &format!("Could not write Radiance's configuration for the plan check:\n{e}"));
                return;
            }
        };
        let (tx, rx) = channel();
        let pr = match proc::spawn_server(&cmd, &env, false, tx) {
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
            ranks: self.r_devs(p).len().max(1),
            placed: 0,
            short: None,
        });
        let shown = format!(
            "$ {} {}",
            env_text(&env_extra),
            shlex_join(&cmd)
        );
        let mut log = vec![format!("[LLM Duo] Fit: plan check (stopped once Radiance states its KV pool and places the weights){}", if ctx > 0 { format!(" at context {}", fmt_thousands(ctx)) } else { String::new() }), shown];
        log.extend(super::radiance::yaml_log_lines(&yaml));
        self.append_log(p, &log);
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
        if let Some(pr) = self.panes[p].r_probe.take() {
            self.panes[p].r_probe_dying = Some(pr.proc.child.clone());
        }
        if self.panes[p].proc.is_none() {
            self.panes[p].state = "Stopped".into();
        }
    }

    /// True while a stopped plan check has not exited yet (its VRAM is still held).
    pub fn r_probe_releasing(&mut self, p: usize) -> bool {
        let Some(c) = self.panes[p].r_probe_dying.as_ref() else { return false };
        let alive = match c.lock() {
            Ok(mut ch) => matches!(ch.try_wait(), Ok(None)),
            Err(_) => false,
        };
        if !alive {
            self.panes[p].r_probe_dying = None;
        }
        alive
    }

    /// The poll loop's share: the running plan check, or the next one once the last has exited.
    pub fn r_probe_tick(&mut self, p: usize) {
        if self.panes[p].r_probe.is_some() {
            self.r_probe_poll(p);
            return;
        }
        if self.panes[p].r_probe_dying.is_some() && !self.r_probe_releasing(p) {
            match self.panes[p].r_probe_next.take() {
                Some((ctx, lo, hi, tries)) => self.r_probe_start(p, ctx, lo, hi, tries),
                // its VRAM is free now: the estimate's "free now" reads right again
                None => self.update_estimate(p),
            }
        }
    }

    /// The experts do not fit beside even a short context's cache: say so, and what would make
    /// room. Nothing is changed.
    fn r_report_no_fit(&mut self, p: usize, ctx: i64, kv1: f64, deficit: f64, ranks: usize) {
        let side = cap_side(self.panes[p].side);
        let model = self.panes[p].model_var.clone();
        let cards = if ranks > 1 { format!("these {ranks} cards") } else { "one card".to_string() };
        // what the experts still lack with (almost) no cache, and with the full context's cache
        let at_min = (deficit - kv1).max(0.0);
        // since 2.0 the host pool is the model's total, split among its cards
        let per = if self.r_config_mode() { ranks as f64 } else { 1.0 };
        let mib = |b: f64| ((b * per / MIB / 1024.0).ceil() as i64 * 1024).max(1024);
        let cur = gi(&self.r_current_options(p), "r_host_pool", 0);
        let mut t = format!(
            "{model} does not fit on {cards} at any context with these settings.\n\n\
             Radiance's plan: one full-length sequence's KV cache{} takes {:.2} GiB a card, and the experts are then {:.2} GiB \
             short of what they need on the card. With almost no context they would still be ≈ {:.2} GiB short, so Radiance \
             refuses the start (DID NOT FIT).\n\nWhat would make room (nothing has been changed):\n",
            if ctx > 0 { format!(" at context {}", fmt_thousands(ctx)) } else { String::new() },
            kv1 / GIB,
            deficit / GIB,
            at_min / GIB,
        );
        if ranks == 1 {
            t += "•  Dual GPU mode: the weights split across both cards (tensor parallel)\n";
        }
        t += &format!(
            "•  Host pool (MiB){}: experts the card cannot hold stay in pinned system RAM and cross PCIe as they are \
             needed (slower decode). ≈ {} MiB{} for this context, ≈ {} MiB with a small one\n",
            if cur > 0 { format!(", now {cur}") } else { String::new() },
            mib(deficit),
            if per > 1.0 { " in total" } else { "" },
            mib(at_min),
        );
        self.msg("warning", &t);
        self.append_log(p, &[format!(
            "[LLM Duo] Fit: the experts do not fit beside any context's cache on {cards} ({:.2} GiB short with almost none) - plan check stopped",
            at_min / GIB
        )]);
        self.set_top_status(format!("{side} pane: Fit - this model does not fit on {cards} with these settings"));
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
                        if is_rank_placed(&l) {
                            pr.placed += 1;
                        }
                        if let Some(x) = expert_shortfall(&l) {
                            pr.short = Some(x);
                        }
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
        // Radiance states the KV pool before it places the weights: the plan only counts once
        // every card has passed placement (a "DID NOT FIT" stops the start after the pool line)
        let failed = exit.map(|rc| rc != 0).unwrap_or(false);
        let placed = pr.placed >= pr.ranks || serving || exit == Some(0);
        let got = !failed && pr.kv.is_some() && placed;
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
                json!({"kv_tokens": kv, "cap_tokens": cap, "backs": backs, "sessions": sessions, "ctx": ctx, "source": "probe", "placed": true, "when": now_iso()}),
            );
            if self.panes[p].model_var == rel {
                self.r_apply_fit(p, cap, "probe");
                self.update_estimate(p);
            }
            return;
        }
        let lines = pr.lines.clone();
        let (ctx, lo, tries, short, ranks) = (pr.ctx, pr.lo, pr.tries, pr.short, pr.ranks);
        if let Some(rc) = exit {
            self.panes[p].r_probe = None;
            self.panes[p].state = "Stopped".into();
            let hint = rad_failure_hint(&lines);
            let text = lines.join("\n");
            let didnt_fit = text.contains("DID NOT FIT") || text.to_lowercase().contains("did not fit");
            // the experts need more of the card than one full-length sequence's cache leaves:
            // the cache shrinks with the context, so a context that frees `deficit` fits
            let shortfall = short.filter(|_| didnt_fit).map(|(kv1, exp, need)| (kv1, (need - exp).max(0.0)));
            if let Some((kv1, deficit)) = shortfall {
                if deficit >= kv1 * 0.95 || (ctx > 0 && ctx <= 4096) {
                    self.r_report_no_fit(p, ctx, kv1, deficit, ranks);
                    return;
                }
            }
            if didnt_fit && tries < 8 && (ctx == 0 || ctx > 4096) {
                let hi = if ctx == 0 { 1 << 20 } else { ctx };
                let next = match (lo, shortfall) {
                    (Some(l), _) => (l + hi) / 2 / 1024 * 1024,
                    // the cache's share that frees enough for the experts, less a little for
                    // per-sequence state that does not shrink with the context
                    (None, Some((kv1, deficit))) if ctx > 0 => {
                        ((ctx as f64 * (kv1 - deficit) / kv1 * 0.97) as i64 / 1024 * 1024).max(4096)
                    }
                    (None, _) if ctx == 0 => 131072,
                    (None, _) => (ctx / 2 / 1024 * 1024).max(4096),
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
            if didnt_fit {
                self.msg(
                    "warning",
                    &format!(
                        "Radiance's plan did not fit{} (exit code {rc}), and Fit found no smaller context that does.\n\n{}Its log says what did not fit (the 'DID NOT FIT' block). Try: Dual GPU mode, a Host pool (MoE models), the i8 KV cache, or Vision tower → ram.\n\nLast lines:\n{}",
                        if ctx > 0 { format!(" at context {}", fmt_thousands(ctx)) } else { String::new() },
                        if hint.is_empty() { String::new() } else { format!("Likely cause: {hint}\n\n") },
                        tail.join("\n")
                    ),
                );
                self.set_top_status(format!("{} pane: Fit - these settings do not fit, see the log", cap_side(self.panes[p].side)));
                return;
            }
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

    /// The selected model's generator step, if it is a generator (image, video, audio, 3D): from the
    /// model info, or from `rad-info` while that is still being read.
    pub fn r_gen(&self, p: usize) -> Option<&'static GenStep> {
        match self.panes[p].info_r() {
            Some(i) => i.gen,
            None => self.model_path(p).and_then(|m| rad_arch(&m)).and_then(|a| gen_step(&a)),
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
            // 2.0 reserves ':' for base:adapter
            n.strip_suffix(".rad").map(String::from).unwrap_or(n).replace(':', "-")
        };
        // a generator with "Radiance sizes it": its plugin sizes the step, the images in a pass and
        // the conditioning (RadArchProbe::gen_rows), which it does only for flags not given
        let gen_auto = probe_ctx.is_none() && gb(&s, "r_gen_auto", false) && self.r_gen(p).is_some();
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
        ];
        if !gen_auto {
            cmd.extend(["--max-model-len".into(), probe_ctx.unwrap_or_else(|| gi(&s, "ctx", 32768)).to_string()]);
        }
        cmd.extend(["--kv-cache-dtype".into(), gs(&s, "r_kv", "i8")]);
        if !gen_auto {
            cmd.extend([
                "--max-num-seqs".into(),
                gi(&s, "parallel", 1).to_string(),
                "--max-num-batched-tokens".into(),
                gi(&s, "r_batched", 2048).to_string(),
            ]);
        }
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
        for (k, flag) in [("r_vision", "--vision-tower-placement"), ("r_embed", "--embedding-placement"), ("r_ngram_place", "--ngram-placement")] {
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
        let cv = gs(&s, "r_cvec", "disabled");
        if cv != "disabled" {
            push("--control-vector", cv);
        }
        let st = gs(&s, "r_state_dtype", "(default)");
        if st != "(default)" {
            push("--state-cache-dtype", st);
        }
        if gi(&s, "r_kv_resident", 0) > 0 {
            push("--kv-resident", gi(&s, "r_kv_resident", 0).to_string());
        }
        if gs(&s, "r_weight_prefetch", "on") == "off" {
            push("--weight-prefetch", "off".into());
        }
        for t in RAD_TEXT_OPTS.iter() {
            let v = gs(&s, t.key, "");
            if !v.trim().is_empty() {
                push(t.flag, v.trim().to_string());
            }
        }
        if devs.len() > 1 {
            push("--tp", devs.len().to_string());
            if gs(&s, "r_tp_wire", "exact") != "exact" {
                push("--tp-wire", gs(&s, "r_tp_wire", "exact"));
            }
            if gs(&s, "r_tp_attn_split", "heads") == "rows" {
                push("--tp-attn-split", "rows".into());
            }
            if gs(&s, "r_p2p", "auto") != "auto" {
                push("--p2p", gs(&s, "r_p2p", "auto"));
            }
        }
        if gb(&s, "r_disk_tier", false) {
            cmd.push("--weights-disk-tier".into()); // a switch: rad-yaml writes it as `true`
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
            uses_config(&bin) || binary_has_flag(&bin, "--api-key-env")
        }
    }

    /// The radiance binary in Settings reads a YAML file (2.0+) rather than flags.
    pub fn r_config_mode(&self) -> bool {
        uses_config(&expanduser(&gs(self.settings.g(), "radiance_bin", "radiance")))
    }

    /// The command that actually starts radiance for this flag line: itself for a 1.x binary;
    /// for 2.0+, `radiance --config FILE` with the flags turned into that file. `tag` names the
    /// file. Returns the command and the YAML (empty for 1.x).
    pub fn r_launch_cmd(&self, flags: &[String], env: &[(String, String)], tag: &str) -> Result<(Vec<String>, String), String> {
        if !self.r_config_mode() {
            return Ok((flags.to_vec(), String::new()));
        }
        let path = config_path(tag);
        let yaml = write_config(flags, env, &path)?;
        Ok((vec![flags[0].clone(), "--config".into(), path.to_string_lossy().into_owned()], yaml))
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
        if let Some(bpt) = tier_bytes_per_token(line) {
            self.r_save_prefix_bpt(p, bpt);
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

