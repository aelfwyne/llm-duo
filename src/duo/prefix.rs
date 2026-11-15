//! The prefix-cache calculator: how much system RAM to give a backend's host-memory prompt cache
//! so that N finished conversations of a given length survive being pushed out of VRAM.
//!
//! Every backend that has one gets the same calculator:
//!   llama.cpp  --cache-ram MiB           (host prompt cache, default 8192, -1 no limit, 0 off)
//!   vLLM       --kv-offloading-size GiB  (native KV offloading to CPU, all ranks together)
//!   Radiance   --prefix-cache-mib MiB    (pinned host tier of the prefix cache)
//!
//! What a conversation of L tokens costs is modelled as
//!   bpt * L + swaBpt * min(L, window) + fixed
//! (full-attention layers, sliding-window layers, per-sequence recurrent state); the dialog does
//! the arithmetic so it can follow the numbers typed into it.

use serde_json::{json, Map, Value};

use super::app::App;
use super::consts::LLAMA_CACHE_RAM_DEFAULT;
use super::estimate::est_simple;
use super::util::*;

impl App {
    pub fn prefix_calc(&mut self, p: usize) -> Value {
        let mut v = if self.panes[p].is_rad() {
            self.r_prefix_calc(p)
        } else if self.panes[p].is_vllm() {
            self.v_prefix_calc(p)
        } else if self.panes[p].is_llama() {
            self.g_prefix_calc(p)
        } else {
            json!({ "ok": false, "source": "KoboldCpp has no size-limited host prompt cache" })
        };
        let o = v.as_object_mut().unwrap();
        let (mem_total, mem_avail) = meminfo();
        let other = o.get("otherPinned").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let pinned = o.get("pinned").and_then(|x| x.as_bool()).unwrap_or(false);
        // amdgpu pins at most about RAM less 1/64 and 1.5 GiB, all pinned memory together;
        // ordinary (pageable) memory is bounded by the RAM itself
        let cap = if pinned { mem_total - mem_total / 64.0 - 1.5 * GIB - other } else { mem_total - 4.0 * GIB - other };
        o.insert("memTotalMib".into(), json!((mem_total / MIB).round()));
        o.insert("memAvailMib".into(), json!((mem_avail / MIB).round()));
        o.insert("otherPinnedMib".into(), json!((other / MIB).round()));
        o.insert("capMib".into(), json!((cap.max(0.0) / MIB).floor()));
        o.insert("running".into(), json!(self.panes[p].proc.is_some()));
        for (k, d) in [("swaBpt", json!(0.0)), ("window", json!(0)), ("fixed", json!(0.0)), ("unit", json!("MiB")), ("unlimited", json!(false))] {
            o.entry(k).or_insert(d);
        }
        v
    }

    /// llama.cpp: a cached prompt is the slot's whole sequence state (llama_state_seq_get_size),
    /// i.e. the KV of every layer at that length, sliding-window layers only for their window,
    /// plus a recurrent model's state. The estimator already knows all three.
    fn g_prefix_calc(&mut self, p: usize) -> Value {
        let s = self.current_options(p);
        let current = gi(&s, "cache_ram", LLAMA_CACHE_RAM_DEFAULT);
        let supported = self.supports(p, "--cache-ram");
        let seqs = gi(&s, "parallel", 1).max(1);
        let ctx = gi(&s, "ctx", 8192);
        let base = json!({
            "key": "cache_ram", "flag": "--cache-ram", "current": current, "unlimited": true, "kv": gs(&s, "kv", "f16"),
            "ctx": (ctx / seqs).max(256), "seqs": seqs, "ranks": 1, "pinned": false,
            "what": "llama-server keeps the state of finished prompts in system RAM, so a conversation that comes back \
                     (another chat, a regenerated reply, a slot that was reused) restores instead of re-reading its prompt. \
                     It is pageable RAM, not pinned. -1 = no limit, 0 = off; llama-server's default is 8192 MiB.",
        });
        let mut o = base.as_object().unwrap().clone();
        if !supported {
            o.insert("ok".into(), false.into());
            o.insert("source".into(), "this llama-server build has no --cache-ram (update llama.cpp)".into());
            return Value::Object(o);
        }
        let Some(info) = self.panes[p].info_g() else {
            o.insert("ok".into(), false.into());
            o.insert("source".into(), "select a model first".into());
            return Value::Object(o);
        };
        let mut a = s.clone();
        a.insert("ngl".into(), (-1).into());
        a.insert("parallel".into(), 1.into());
        a.insert("mtp".into(), "".into());
        let kv_at = |c: i64| {
            let mut x = a.clone();
            x.insert("ctx".into(), c.into());
            est_simple(&info, &x, 0.0, 0.0, 0.0)
        };
        let ub = gi(&s, "ubatch", 512) as f64;
        let window = if info.swa != 0.0 && !gs(&s, "extra", "").contains("--swa-full") {
            (((info.swa + ub) / 256.0).ceil() * 256.0) as i64
        } else {
            0
        };
        // past the window, only full-attention layers grow
        let c = (window * 2).max(65536);
        let (e1, e2) = (kv_at(c), kv_at(2 * c));
        let bpt = ((e2.kv - e1.kv) / c as f64).max(0.0);
        let swa_total = (e1.kv - bpt * c as f64).max(0.0);
        let swa_bpt = if window > 0 { swa_total / window as f64 } else { 0.0 };
        o.insert("ok".into(), (bpt + swa_bpt > 0.0).into());
        o.insert("bpt".into(), bpt.into());
        o.insert("swaBpt".into(), swa_bpt.into());
        o.insert("window".into(), window.into());
        o.insert("fixed".into(), e1.state.max(0.0).into());
        o.insert("exact".into(), false.into());
        o.insert(
            "source".into(),
            format!(
                "from the model's layers and KV cache {}{}",
                gs(&s, "kv", "f16"),
                if window > 0 { format!("; sliding-window layers keep only {} tokens", fmt_thousands(window)) } else { String::new() }
            )
            .into(),
        );
        Value::Object(o)
    }

    /// vLLM: offloaded blocks are the KV of every layer (all ranks together), sliding-window
    /// layers up to their window.
    fn v_prefix_calc(&mut self, p: usize) -> Value {
        let s = self.current_options(p);
        let ranks = self.v_devs(p).len().max(1);
        let co = gf(&s, "cpu_offload", 0.0).max(0.0) * GIB * ranks as f64;
        let mut o: Map<String, Value> = json!({
            "key": "kv_offload", "flag": "--kv-offloading-size", "unit": "GiB",
            "current": gf(&s, "kv_offload", 0.0), "kv": gs(&s, "kv_dtype", "auto"),
            "ctx": gi(&s, "ctx", 8192), "seqs": gi(&s, "parallel", 1).max(1), "ranks": ranks, "pinned": true,
            "otherPinned": co, "otherPinnedName": "CPU offload",
            "what": "vLLM offloads KV blocks to pinned system RAM (--kv-offloading-size, all cards together), so a prefix \
                     that left VRAM is copied back instead of re-computed. 0 = off.",
        })
        .as_object()
        .unwrap()
        .clone();
        let Some(info) = self.panes[p].info_v() else {
            o.insert("ok".into(), false.into());
            o.insert("source".into(), "select a model first".into());
            return Value::Object(o);
        };
        let eb = if gs(&s, "kv_dtype", "auto").starts_with("fp8") { 1.0 } else { 2.0 };
        let sum = |k: &str| (0..info.kinds.len()).filter(|&i| info.kinds[i] == k).map(|i| info.kv_tok(i, eb)).sum::<f64>();
        let (full, swa) = (sum("full"), sum("swa"));
        let window = if info.window > 0.0 { info.window as i64 } else { 0 };
        let (bpt, swa_bpt) = if window > 0 { (full, swa) } else { (full + swa, 0.0) };
        o.insert("ok".into(), (bpt + swa_bpt > 0.0).into());
        o.insert("bpt".into(), bpt.into());
        o.insert("swaBpt".into(), swa_bpt.into());
        o.insert("window".into(), window.into());
        o.insert("exact".into(), false.into());
        o.insert(
            "source".into(),
            format!(
                "from the model's config.json at KV cache {}{}",
                gs(&s, "kv_dtype", "auto"),
                if window > 0 { format!("; sliding-window layers keep only {} tokens", fmt_thousands(window)) } else { String::new() }
            )
            .into(),
        );
        Value::Object(o)
    }
}

/// System memory (MemTotal, MemAvailable) in bytes, 0 where /proc/meminfo has no answer.
pub fn meminfo() -> (f64, f64) {
    let txt = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| {
        txt.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
            .map(|k| k * 1024.0)
            .unwrap_or(0.0)
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

