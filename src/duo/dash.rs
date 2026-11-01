//! Performance dashboards: live telemetry for a pane (llama.cpp) and the history of every
//! configuration. The UI asks for these on demand (`call("dash", ...)`), so nothing here
//! goes into the regular UI state.
//!
//! History is kept in two places:
//! * the `stats` section of the settings (one running total per GPU/backend + model, reset
//!   whenever the load options change - the same data the Python version keeps), and
//! * `history.jsonl` in the data folder: one line per load, request, crash, ... and one
//!   "retired" line holding the totals of a configuration whose options were changed,
//!   so nothing is lost when the running totals start over.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use serde_json::{json, Map, Value};

use super::app::App;
use super::devices::Device;
use super::pane::cap_side;
use super::telemetry::{read_sensors, unix_now, GpuSample, KEEP_REQUESTS, KEEP_SAMPLES};
use super::util::*;

fn history_path() -> std::path::PathBuf {
    dirs().data.join("history.jsonl")
}

fn o(v: Option<f64>) -> Value {
    match v {
        Some(x) if x.is_finite() => json!((x * 100.0).round() / 100.0),
        _ => Value::Null,
    }
}

fn peak(w: &[&GpuSample], f: &dyn Fn(&GpuSample) -> Option<f64>) -> Option<f64> {
    w.iter().filter_map(|s| f(s)).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
}

fn short_gpu(name: &str) -> String {
    name.replace("AMD Radeon ", "").replace("NVIDIA GeForce ", "").trim().to_string()
}

/// "Vulkan:DUAL:A+B" -> ("llama.cpp", "Vulkan", "Dual: A + B")
pub fn describe_key(key: &str) -> (String, String, String, bool) {
    let mut rest = key;
    let mut backend = "llama.cpp".to_string();
    let mut build = "ROCm".to_string();
    if let Some(r) = rest.strip_prefix("Radiance:") {
        backend = "Radiance".into();
        build = String::new();
        rest = r;
    } else if let Some(r) = rest.strip_prefix("vLLM:") {
        backend = "vLLM".into();
        build = String::new();
        rest = r;
    } else if let Some(r) = rest.strip_prefix("Kobold-ROCm:") {
        backend = "KoboldCpp".into();
        rest = r;
    } else if let Some(r) = rest.strip_prefix("Kobold-Vulkan:") {
        backend = "KoboldCpp".into();
        build = "Vulkan".into();
        rest = r;
    } else if let Some(r) = rest.strip_prefix("Kobold:") {
        backend = "KoboldCpp".into();
        rest = r;
    } else if let Some(r) = rest.strip_prefix("Vulkan:") {
        build = "Vulkan".into();
        rest = r;
    }
    if let Some(r) = rest.strip_prefix("DUAL:") {
        let names: Vec<String> = r.split('+').map(short_gpu).collect();
        return (backend, build, format!("Dual: {}", names.join(" + ")), true);
    }
    (backend, build, short_gpu(rest), false)
}

fn rate(tok: f64, ms: f64) -> Option<f64> {
    if tok > 0.0 && ms > 0.0 {
        Some(tok / (ms / 1000.0))
    } else {
        None
    }
}

fn pct(a: f64, b: f64) -> Option<f64> {
    if b > 0.0 {
        Some(100.0 * a / b)
    } else {
        None
    }
}

/// A finding: `id` stays the same while the condition lasts (so the UI can keep it until it
/// is dismissed), `scope` names the server or card it is about.
fn insight(id: &str, scope: &str, level: &str, title: &str, text: String) -> Value {
    json!({"id": id, "scope": scope, "level": level, "title": title, "text": text})
}

/// One configuration row for the history dashboard, from a stats object.
fn stats_row(key: &str, rel: &str, st: &Opts, status: &str) -> Value {
    let (backend, build, hw, dual) = describe_key(key);
    let f = |k: &str| gf(st, k, 0.0);
    let fp: Opts = st
        .get("fp")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    let mut setup = Map::new();
    for k in ["ctx", "kv", "kv_dtype", "ngl", "n_cpu_moe", "batch", "ubatch", "parallel", "threads", "fa", "mtp", "mtp_n",
              "dual_strategy", "dual_mode", "extra", "gpu_util", "dtype", "max_batched", "cpu_offload",
              "r_kv", "r_batched", "r_spec", "r_spec_ngram", "r_draft_window", "r_vision", "r_placement", "r_host_pool", "r_tp_wire"] {
        if (k.starts_with("dual_") || k == "r_tp_wire") && !dual {
            continue; // stored for every setup, only meaningful in Dual GPU mode
        }
        if let Some(v) = fp.get(k) {
            setup.insert(k.into(), v.clone());
        }
    }
    let mtp_on = !matches!(gs(&fp, "mtp", "(off)").as_str(), "(off)" | "") || (fp.contains_key("r_spec") && gs(&fp, "r_spec", "auto") != "0");
    json!({
        "id": format!("{key}|{rel}|{status}|{}|{:x}", gs(st, "since", ""), hash(&gs(st, "fp", ""))),
        "key": key, "model": rel, "backend": backend, "build": build, "hardware": hw, "dual": dual,
        "status": status,
        "runs": gi(st, "loads", 0), "requests": gi(st, "requests", 0), "aborts": gi(st, "aborts", 0),
        "crashes": gi(st, "crashes", 0), "loadFailed": gi(st, "load_failed", 0), "loadAborted": gi(st, "load_aborted", 0),
        "spills": gi(st, "spills", 0), "runSecs": f("run_secs"),
        "genTps": o(rate(f("gen_tokens"), f("gen_ms"))),
        "genMin": o(st.get("gen_tps_min").and_then(|v| v.as_f64())),
        "genMax": o(st.get("gen_tps_max").and_then(|v| v.as_f64())),
        "promptTps": o(rate(f("big_prompt_tokens"), f("big_prompt_ms"))),
        "mtp": if mtp_on { o(pct(f("draft_acc"), f("draft_gen"))) } else { Value::Null },
        "cache": o(pct(f("cache_tokens"), f("cache_total"))),
        "genTokens": gi(st, "gen_tokens", 0), "promptTokens": gi(st, "prompt_tokens", 0),
        "since": gs(st, "since", ""), "last": st.get("last").cloned().unwrap_or(Value::Null),
        "setup": setup,
    })
}

const RAD_DASH_METRICS: [&str; 19] = [
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total", "vllm:num_preemptions_total",
    "radiance:decode_tokens_per_second", "radiance:prefill_tokens_per_second", "radiance:engine_steps_total",
    "radiance:draft_acceptance_ratio", "radiance:draft_depth_adaptive", "radiance:attn_cached_tokens_total",
    "radiance:linear_prefix_cache_hit_rate", "radiance:prefix_cache_evictions_total", "radiance:num_requests_preempted",
    "radiance:client_disconnect_total", "radiance:http_admission_rejected_total",
];

fn hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish() & 0xffff_ffff
}

struct DRe {
    new_prompt: regex::Regex,
    progress: regex::Regex,
    cancel: regex::Regex,
    launch: regex::Regex,
    gen: regex::Regex,
    task: regex::Regex,
}

/// The task id in a llama-server slot line ("... | task 771 | ...").
pub fn task_id(line: &str) -> Option<i64> {
    dre().task.captures(line).and_then(|m| m[1].parse().ok())
}

fn dre() -> &'static DRe {
    static R: std::sync::OnceLock<DRe> = std::sync::OnceLock::new();
    R.get_or_init(|| DRe {
        new_prompt: regex::Regex::new(r"new prompt, n_ctx_slot = (\d+).*?task\.n_tokens = (\d+)").unwrap(),
        // "prompt processing progress, n_past = …, progress = 0.43" (older) or
        // "prompt processing, n_tokens = 4096, progress = 0.09, t = …" (newer)
        progress: regex::Regex::new(r"prompt processing\b.*?n_(?:tokens|past) = (\d+).*?progress = ([\d.]+)").unwrap(),
        cancel: regex::Regex::new(r"cancel task, id_task = (\d+)").unwrap(),
        launch: regex::Regex::new(r"processing task").unwrap(),
        // "n_gen =    130, tg =  42.07 t/s, tg_3s =  42.39 t/s"
        gen: regex::Regex::new(r"n_gen =\s*(\d+), tg =\s*([\d.]+) t/s(?:, tg_3s =\s*([\d.]+))?").unwrap(),
        task: regex::Regex::new(r"\btask (\d+)").unwrap(),
    })
}

impl App {
    /// llama-server log lines that matter only to the live dashboard (request in flight,
    /// prompt size, prompt progress, cancelled requests).
    pub fn dash_line(&mut self, p: usize, line: &str) {
        let re = dre();
        if re.launch.is_match(line) {
            self.panes[p].dash_active = Some((unix_now(), None, None));
            self.panes[p].dash_progress = Some(0.0);
            self.panes[p].dash_gen = None;
        } else if let Some(m) = re.new_prompt.captures(line) {
            let ctx = m[1].parse().ok();
            let n = m[2].parse().ok();
            self.panes[p].dash_slot_ctx = ctx;
            let t0 = self.panes[p].dash_active.map(|a| a.0).unwrap_or_else(unix_now);
            self.panes[p].dash_active = Some((t0, n, ctx));
        } else if let Some(m) = re.progress.captures(line) {
            let (t0, n0, ctx) = self.panes[p].dash_active.unwrap_or((unix_now(), None, None));
            let done: f64 = m[1].parse().unwrap_or(0.0);
            let frac: Option<f64> = m[2].parse().ok();
            // the prompt's size, when the log doesn't say it directly: done / progress
            let n = n0.or_else(|| frac.filter(|f| *f >= 0.05).map(|f| (done / f).round() as i64));
            self.panes[p].dash_active = Some((t0, n, ctx));
            self.panes[p].dash_progress = frac;
        } else if let Some(m) = re.gen.captures(line) {
            let n = m[1].parse().unwrap_or(0);
            let tps = m.get(3).or(m.get(2)).and_then(|x| x.as_str().parse().ok()).unwrap_or(0.0);
            self.panes[p].dash_gen = Some((n, tps));
            self.panes[p].dash_progress = Some(1.0);
            if self.panes[p].dash_active.is_none() {
                self.panes[p].dash_active = Some((unix_now(), None, None));
            }
        } else if let Some(m) = re.cancel.captures(line) {
            if let Ok(t) = m[1].parse() {
                let c = &mut self.panes[p].dash_cancelled;
                c.push(t);
                if c.len() > 64 {
                    c.remove(0);
                }
            }
        }
    }

    // ============================================================ sampling ==
    /// Read every GPU's sensors (every tick, whether a dashboard is open or not, so the
    /// graphs already have history when one is opened).
    pub fn sample_gpus(&mut self) {
        for d in self.devices.clone() {
            let (s, lim) = read_sensors(&d);
            let h = self.gpu_hist.entry(d.id.clone()).or_default();
            h.push_back(s);
            while h.len() > KEEP_SAMPLES {
                h.pop_front();
            }
            self.gpu_limits.insert(d.id.clone(), lim);
        }
    }

    pub fn dash_metrics(&mut self, p: usize, m: &HashMap<String, f64>) {
        let q = &mut self.panes[p].dash_metrics;
        q.push_back((unix_now(), m.clone()));
        while q.len() > KEEP_SAMPLES {
            q.pop_front();
        }
    }

    // ============================================================= history ==
    /// Append one event to history.jsonl (best effort: a full disk never breaks the app).
    pub fn history_event(&mut self, p: usize, kind: &str, extra: Value) {
        let Some((key, rel)) = self.panes[p].stats_ids_loaded.clone() else { return };
        let mut rec = json!({"t": (unix_now() * 1000.0).round() / 1000.0, "when": now_iso(), "kind": kind, "key": key, "model": rel});
        if let (Some(o), Value::Object(e)) = (rec.as_object_mut(), extra) {
            o.extend(e);
        }
        let path = history_path();
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{rec}");
        }
    }

    /// A finished request (llama.cpp): keep it for the live dashboard and the history.
    pub fn dash_request(&mut self, p: usize, r: &Opts, cache: Option<(i64, i64)>, aborted: bool) {
        let now = unix_now();
        let (start, prompt_n, slot_ctx) = self.panes[p].dash_active.take().unwrap_or((now, None, None));
        self.panes[p].dash_progress = None;
        let p_tok = gi(r, "p_tok", 0);
        let g_tok = gi(r, "g_tok", 0);
        let (p_ms, g_ms) = (gf(r, "p_ms", 0.0), gf(r, "g_ms", 0.0));
        let ctx = slot_ctx.or(self.panes[p].dash_slot_ctx).or_else(|| {
            self.panes[p].loaded.as_ref().map(|(_, o, _)| gi(o, "ctx", 0)).filter(|c| *c > 0)
        });
        let rec = json!({
            "t": (now * 1000.0).round() / 1000.0,
            "dur": ((now - start) * 100.0).round() / 100.0,
            "aborted": aborted,
            "pTok": p_tok, "pMs": p_ms, "pTps": o(rate(p_tok as f64, p_ms)),
            "gTok": g_tok, "gMs": g_ms, "gTps": o(rate(g_tok as f64, g_ms)),
            "acc": gi(r, "acc", 0), "gen": gi(r, "gen", 0),
            "mtp": o(pct(gi(r, "acc", 0) as f64, gi(r, "gen", 0) as f64)),
            "cached": cache.map(|c| c.0), "total": cache.map(|c| c.1).or(prompt_n),
            "cache": o(cache.and_then(|(c, t)| pct(c as f64, t as f64))),
            "nEnd": r.get("n_end").cloned().unwrap_or(Value::Null),
            "ctx": ctx,
        });
        let q = &mut self.panes[p].dash_reqs;
        q.push_back(rec.clone());
        while q.len() > KEEP_REQUESTS {
            q.pop_front();
        }
        self.history_event(p, if aborted { "abort" } else { "request" }, rec);
    }

    // ================================================================ live ==
    /// Delete one configuration from the history: its running totals (a current row) or its
    /// retired totals line (an earlier row), and every event recorded for it in that period.
    pub fn dash_delete(&mut self, arg: &Value) -> Value {
        let s = |k: &str| arg.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let (key, model, status, since, retired) = (s("key"), s("model"), s("status"), s("since"), s("retired"));
        if key.is_empty() || model.is_empty() {
            return json!({"error": "Nothing to delete."});
        }
        let current = status == "current";
        if current {
            let busy = self.panes.iter().any(|p| p.proc.is_some() && p.stats_ids_loaded.as_ref() == Some(&(key.clone(), model.clone())));
            if busy {
                return json!({"error": "That configuration is running now - stop its model first."});
            }
        }
        let mut removed_events = 0usize;
        let path = history_path();
        if let Ok(text) = std::fs::read_to_string(&path) {
            let mut kept = String::with_capacity(text.len());
            let mut dropped_retired = false;
            for line in text.lines() {
                let drop = match serde_json::from_str::<Value>(line) {
                    Ok(v) if v.get("key").and_then(|x| x.as_str()) == Some(key.as_str())
                        && v.get("model").and_then(|x| x.as_str()) == Some(model.as_str()) =>
                    {
                        let kind = v.get("kind").and_then(|x| x.as_str()).unwrap_or("");
                        let when = v.get("when").and_then(|x| x.as_str()).unwrap_or("");
                        if kind == "retired" {
                            // the earlier row itself (once), matched by when its totals started and ended
                            let st_since = v.get("stats").and_then(|x| x.get("since")).and_then(|x| x.as_str()).unwrap_or("");
                            let hit = !current && !dropped_retired && st_since == since && (retired.is_empty() || when == retired);
                            if hit {
                                dropped_retired = true;
                            }
                            hit
                        } else if current {
                            !since.is_empty() && when >= since.as_str()
                        } else {
                            !since.is_empty() && when >= since.as_str() && (retired.is_empty() || when < retired.as_str())
                        }
                    }
                    _ => false,
                };
                if drop {
                    removed_events += 1;
                } else {
                    kept.push_str(line);
                    kept.push('\n');
                }
            }
            if removed_events > 0 {
                let tmp = path.with_extension("jsonl.tmp");
                if std::fs::write(&tmp, kept).and_then(|_| std::fs::rename(&tmp, &path)).is_err() {
                    return json!({"error": format!("Could not rewrite {}", path.display())});
                }
            }
        }
        let had = current && self.settings.del_stats(&key, &model);
        for p in 0..2 {
            self.refresh_life(p);
        }
        json!({"ok": true, "stats": had, "events": removed_events})
    }

    /// Cards a pane runs on (both in Dual GPU mode; none for the right pane while Dual is on).
    fn pane_devs(&self, p: usize) -> Vec<Device> {
        if self.dual_active(p) {
            self.dual_devices()
        } else if p == 1 && self.dual_on() {
            vec![]
        } else {
            self.panes[p].device.clone().into_iter().collect()
        }
    }

    /// Everything on one screen: every card, and every pane that has a model (both servers
    /// side by side; one in Dual GPU mode).
    pub fn dash_live(&self, arg: &Value) -> Value {
        let minutes = arg.get("minutes").and_then(|v| v.as_f64()).unwrap_or(10.0).clamp(1.0, 30.0);
        let now = unix_now();
        let since = now - minutes * 60.0;
        let slot_of = |id: &str| self.devices.iter().position(|d| d.id == id).unwrap_or(0);

        // ---- GPUs (all of them, in detection order so each keeps its colour)
        let mut gpus = vec![];
        for (slot, d) in self.devices.iter().enumerate() {
            let users: Vec<&str> = (0..2).filter(|&i| self.pane_devs(i).iter().any(|x| x.id == d.id))
                .map(|i| if self.dual_active(i) { "dual" } else { self.panes[i].side }).collect();
            let empty = Default::default();
            let h = self.gpu_hist.get(&d.id).unwrap_or(&empty);
            let w: Vec<&GpuSample> = h.iter().filter(|s| s.t >= since).collect();
            let col = |f: &dyn Fn(&GpuSample) -> Option<f64>| -> Vec<Value> { w.iter().map(|s| o(f(s))).collect() };
            let lim = self.gpu_limits.get(&d.id).cloned().unwrap_or_default();
            let cur = h.back().cloned().unwrap_or_default();
            let busy_avg = {
                let b: Vec<f64> = w.iter().filter_map(|s| s.busy).collect();
                if b.is_empty() { None } else { Some(b.iter().sum::<f64>() / b.len() as f64) }
            };
            gpus.push(json!({
                "id": d.id, "name": d.name, "short": short_gpu(&d.name), "slot": slot,
                "usedBy": users.join(" + "),
                "live": d.sysfs.is_some(),
                "totalGiB": o(cur.total.map(|b| b / GIB)),
                "limits": {"edge": o(lim.edge), "junction": o(lim.junction), "mem": o(lim.mem), "power": o(lim.power_cap)},
                "cur": {
                    "vramGiB": o(cur.used.map(|b| b / GIB)), "gttGiB": o(cur.gtt.map(|b| b / GIB)), "busy": o(cur.busy),
                    "edge": o(cur.edge), "junction": o(cur.junction), "mem": o(cur.mem), "power": o(cur.power),
                    "sclk": o(cur.sclk), "mclk": o(cur.mclk), "fan": o(cur.fan),
                },
                "busyAvg": o(busy_avg),
                // highest readings in the shown time range
                "peak": {
                    "vramGiB": o(peak(&w, &|s| s.used.map(|b| b / GIB))), "busy": o(peak(&w, &|s| s.busy)),
                    "edge": o(peak(&w, &|s| s.edge)), "junction": o(peak(&w, &|s| s.junction)), "mem": o(peak(&w, &|s| s.mem)),
                    "power": o(peak(&w, &|s| s.power)), "sclk": o(peak(&w, &|s| s.sclk)), "mclk": o(peak(&w, &|s| s.mclk)),
                    "fan": o(peak(&w, &|s| s.fan)),
                },
                "series": {
                    "t": w.iter().map(|s| json!(((s.t - now) * 10.0).round() / 10.0)).collect::<Vec<_>>(),
                    "vram": col(&|s| s.used.map(|b| b / GIB)),
                    "busy": col(&|s| s.busy),
                    "edge": col(&|s| s.edge),
                    "junction": col(&|s| s.junction),
                    "power": col(&|s| s.power),
                },
            }));
        }

        // ---- servers
        let mut shown: Vec<usize> = (0..2)
            .filter(|&p| !(p == 1 && self.dual_on()))
            .filter(|&p| self.panes[p].proc.is_some() || !self.panes[p].model_var.is_empty())
            .collect();
        if shown.is_empty() {
            shown.push(0);
        }
        let mut servers = vec![];
        let mut insights: Vec<(u8, Value)> = vec![];
        let mut all_reqs: Vec<Value> = vec![];
        for &p in &shown {
            let (srv, mut ins, reqs) = self.dash_server(p, now, &gpus, &slot_of);
            servers.push(srv);
            insights.append(&mut ins);
            all_reqs.extend(reqs);
        }
        insights.extend(self.dash_gpu_insights(&gpus));
        insights.sort_by_key(|x| x.0);
        all_reqs.sort_by(|a, b| a["t"].as_f64().unwrap_or(0.0).partial_cmp(&b["t"].as_f64().unwrap_or(0.0)).unwrap());
        let keep = all_reqs.len().saturating_sub(300);
        json!({
            "now": now, "minutes": minutes,
            "servers": servers,
            "gpus": gpus,
            "requests": all_reqs.split_off(keep),
            "insights": insights.into_iter().map(|x| x.1).collect::<Vec<_>>(),
        })
    }

    /// One pane's server: its figures, findings and requests (tagged with the pane).
    fn dash_server(&self, p: usize, now: f64, gpus: &[Value], slot_of: &dyn Fn(&str) -> usize) -> (Value, Vec<(u8, Value)>, Vec<Value>) {
        let pane = &self.panes[p];
        let devs = self.pane_devs(p);
        let dual = self.dual_active(p);
        let cards: Vec<String> = devs.iter().map(|d| short_gpu(&d.name)).collect();
        let who = if dual { "Dual".to_string() } else { cap_side(pane.side) };
        let label = if cards.is_empty() { who.clone() } else { format!("{who} • {}", cards.join(" + ")) };
        let slot = devs.first().map(|d| slot_of(&d.id)).unwrap_or(p);

        let reqs: Vec<&Value> = pane.dash_reqs.iter().collect();
        let done: Vec<&&Value> = reqs.iter().filter(|r| !truthy(r.get("aborted"))).collect();
        let sum = |k: &str, list: &[&&Value]| -> f64 { list.iter().map(|r| r.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)).sum() };
        let big: Vec<&&Value> = done.iter().filter(|r| r.get("pTok").and_then(|v| v.as_i64()).unwrap_or(0) >= 1024).cloned().collect();
        let cached = sum("cached", &done);
        let ctotal: f64 = done.iter().filter(|r| r.get("cached").map(|v| !v.is_null()).unwrap_or(false))
            .map(|r| r.get("total").and_then(|v| v.as_f64()).unwrap_or(0.0)).sum();
        let busy_secs = sum("dur", &reqs.iter().collect::<Vec<_>>());
        let up_since = pane.dash_ready_t.filter(|_| pane.proc.is_some());
        let session = json!({
            "requests": done.len(), "aborted": reqs.len() - done.len(),
            "promptTokens": sum("pTok", &done), "genTokens": sum("gTok", &done), "cachedTokens": cached,
            "genTps": o(rate(sum("gTok", &done), sum("gMs", &done))),
            "promptTps": o(rate(sum("pTok", &big), sum("pMs", &big))),
            "mtp": o(pct(sum("acc", &done), sum("gen", &done))),
            "cache": o(pct(cached, ctotal)),
            "busyPct": o(up_since.and_then(|t| pct(busy_secs, now - t)).map(|v| v.min(100.0))),
        });
        let mlast = pane.dash_metrics.back().map(|(_, m)| m.clone()).unwrap_or_default();
        let mut mkeys: Vec<(&String, &f64)> = mlast.iter().collect();
        mkeys.sort_by(|a, b| a.0.cmp(b.0));
        // Radiance exports dozens of series: show the ones worth reading, without the family prefix
        let rad_shown: Vec<(String, f64)> = if pane.is_rad() {
            let mut v: Vec<(String, f64)> = RAD_DASH_METRICS
                .iter()
                .filter_map(|k| mlast.get(*k).map(|x| (k.split(':').nth(1).unwrap_or(k).to_string(), *x)))
                .collect();
            let (st, sd) = (mlast.get("radiance:engine_step_seconds_decode_total"), mlast.get("radiance:engine_steps_decode_total"));
            if let (Some(t), Some(n)) = (st, sd.filter(|n| **n > 0.0)) {
                v.insert(0, ("decode step ms".into(), 1000.0 * t / n));
            }
            v
        } else {
            vec![]
        };
        if pane.is_rad() {
            mkeys = rad_shown.iter().map(|(k, v)| (k, v)).collect();
        }
        let active = pane.dash_active.map(|(t0, n, c)| json!({"since": ((now - t0) * 10.0).round() / 10.0, "promptTokens": n, "ctx": c,
                                                              "progress": o(pane.dash_progress.map(|x| x * 100.0)),
                                                              "genTokens": pane.dash_gen.map(|g| g.0), "genTps": o(pane.dash_gen.map(|g| g.1))}));
        let opts = pane.loaded.as_ref().map(|(_, o, _)| o.clone()).unwrap_or_else(|| self.current_options(p));
        let hist = pane.stats_ids_loaded.clone().or_else(|| self.stats_ids(p)).and_then(|(k, r)| self.settings.get_stats(&k, &r));
        let mine: Vec<&Value> = gpus.iter().filter(|g| devs.iter().any(|d| Some(d.id.as_str()) == g["id"].as_str())).collect();
        let ins = self.dash_insights(p, &label, &mine, &reqs, &session, hist.as_ref(), &mlast);
        let (backend, build) = if pane.is_rad() {
            ("Radiance".to_string(), String::new())
        } else if pane.is_vllm() {
            ("vLLM".to_string(), String::new())
        } else if pane.is_kobold() {
            ("KoboldCpp".to_string(), self.kbuild(p))
        } else {
            ("llama.cpp".to_string(), self.kbuild(p))
        };
        let tagged: Vec<Value> = reqs.iter().map(|r| {
            let mut r = (*r).clone();
            r["side"] = json!(pane.side);
            r["slot"] = json!(slot);
            r["label"] = json!(label);
            r
        }).collect();
        let srv = json!({
            "side": pane.side, "label": label, "who": who, "slot": slot, "supported": pane.is_llama() || pane.is_rad(),
            "model": pane.model_var, "backend": backend, "build": build, "state": pane.state,
            "running": pane.proc.is_some(), "dual": dual,
            "uptime": pane.status.get("uptime").cloned().unwrap_or_default(),
            "ctx": gi(&opts, "ctx", 0),
            "kv": if pane.is_rad() { gs(&opts, "r_kv", "") } else { gs(&opts, "kv", "") },
            "mtp": if pane.is_rad() {
                match gs(&opts, "r_spec", "auto").as_str() { "0" => "(off)".to_string(), "auto" => "auto".to_string(), _ => "on".to_string() }
            } else { gs(&opts, "mtp", "") },
            "mtpN": if pane.is_rad() { gs(&opts, "r_spec", "0").parse::<i64>().unwrap_or(0) } else { gi(&opts, "mtp_n", 0) },
            "kvPool": if pane.is_rad() { json!(pane.r_kv_tokens) } else { Value::Null },
            "loadSecs": o(match (pane.dash_load_t, pane.dash_ready_t) { (Some(a), Some(b)) if b >= a => Some(b - a), _ => None }),
            "session": session,
            "history": hist.map(|st| {
                let f = |k: &str| gf(&st, k, 0.0);
                json!({"requests": gi(&st, "requests", 0), "runs": gi(&st, "loads", 0),
                       "genTps": o(rate(f("gen_tokens"), f("gen_ms"))), "promptTps": o(rate(f("big_prompt_tokens"), f("big_prompt_ms"))),
                       "mtp": o(pct(f("draft_acc"), f("draft_gen"))), "since": gs(&st, "since", "")})
            }),
            "active": active,
            "last": done.last().map(|r| (**r).clone()),
            "metrics": mkeys.iter().map(|(k, v)| json!([k, o(Some(**v))])).collect::<Vec<_>>(),
        });
        (srv, ins, tagged)
    }

    /// Card findings (temperatures, power, memory) for every card.
    fn dash_gpu_insights(&self, gpus: &[Value]) -> Vec<(u8, Value)> {
        let mut out = vec![];
        let num = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        // cards a running Radiance server holds: it claims all VRAM and pins host memory on
        // purpose, so "VRAM full" and GTT use are normal there
        let rad_cards: Vec<String> = (0..2)
            .filter(|&p| self.panes[p].is_rad() && self.panes[p].proc.is_some())
            .flat_map(|p| self.pane_devs(p).into_iter().map(|d| d.id))
            .collect();
        for g in gpus {
            let name = g["short"].as_str().unwrap_or("GPU").to_string();
            let id = g["id"].as_str().unwrap_or("");
            let in_use = !g["usedBy"].as_str().unwrap_or("").is_empty();
            let cur = &g["cur"];
            let lim = &g["limits"];
            if let (Some(j), Some(c)) = (num(cur, "junction"), num(lim, "junction")) {
                if j >= c {
                    out.push((0, insight(&format!("gpu:{id}:hot"), &name, "critical", "Hotspot at its limit",
                        format!("Junction {j:.0} °C against a {c:.0} °C limit - the card is throttling. Check airflow and fan curve."))));
                } else if j >= c - 5.0 {
                    out.push((2, insight(&format!("gpu:{id}:warm"), &name, "warning", "Running hot",
                        format!("Junction {j:.0} °C, {:.0} °C below the {c:.0} °C limit.", c - j))));
                }
            }
            if let (Some(m), Some(c)) = (num(cur, "mem"), num(lim, "mem")) {
                if m >= c - 5.0 {
                    out.push((1, insight(&format!("gpu:{id}:memtemp"), &name, "serious", "Memory temperature high",
                        format!("VRAM at {m:.0} °C (limit {c:.0} °C). Memory throttling lowers generation speed directly."))));
                }
            }
            if !in_use || rad_cards.iter().any(|c| c == id) {
                continue;
            }
            if let (Some(u), Some(t)) = (num(cur, "vramGiB"), g["totalGiB"].as_f64()) {
                if t > 0.0 && u / t > 0.985 {
                    out.push((2, insight(&format!("gpu:{id}:vram"), &name, "warning", "VRAM full",
                        format!("{u:.1} of {t:.1} GiB in use. Another program using the GPU could push the model into system RAM."))));
                }
            }
            if let Some(gtt) = num(cur, "gttGiB") {
                if gtt > 2.0 {
                    out.push((3, insight(&format!("gpu:{id}:gtt"), &name, "warning", &format!("{gtt:.1} GiB in GTT"),
                        "System RAM mapped for the GPU is in use. A little is normal; a lot usually means part of the model or cache spilled over.".into())));
                }
            }
        }
        out
    }

    /// One server's findings. Each has a stable id so the UI can keep it until dismissed.
    fn dash_insights(&self, p: usize, scope: &str, gpus: &[&Value], reqs: &[&Value], session: &Value, _hist: Option<&Opts>,
                     metrics: &HashMap<String, f64>) -> Vec<(u8, Value)> {
        let mut out: Vec<(u8, Value)> = vec![];
        let pane = &self.panes[p];
        let side = pane.side;
        let id = |k: &str| format!("{side}:{k}");
        let num = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let done: Vec<&&Value> = reqs.iter().filter(|r| !truthy(r.get("aborted"))).collect();

        if pane.spilled {
            out.push((0, insight(&id("spill"), scope, "critical", "Model is partly in system RAM",
                "VRAM use is lower than the model needs, so some of it runs from system memory. Expect a large slowdown; lower the context or GPU layers, or close other GPU users.".into())));
        }
        if gpus.len() == 2 && done.len() >= 2 && !pane.is_rad() {
            if let (Some(a), Some(b)) = (gpus[0]["busyAvg"].as_f64(), gpus[1]["busyAvg"].as_f64()) {
                let (hi, lo) = if a >= b { (0, 1) } else { (1, 0) };
                let (vh, vl) = (a.max(b), a.min(b));
                if vh > 30.0 && vl < vh * 0.5 {
                    out.push((5, insight(&id("uneven"), scope, "info", "Uneven load between the cards",
                        format!("{} averaged {vh:.0}% busy, {} {vl:.0}%. With a layer split the cards take turns, so the slower card sets the pace; more layers on the faster card helps.",
                                gpus[hi]["short"].as_str().unwrap_or("?"), gpus[lo]["short"].as_str().unwrap_or("?")))));
                }
            }
        }
        if let Some(m) = num(session, "mtp") {
            let rad = pane.is_rad();
            let n = pane.loaded.as_ref().map(|(_, o, _)| if rad { gs(o, "r_spec", "0").parse().unwrap_or(0) } else { gi(o, "mtp_n", 0) }).unwrap_or(0);
            let knob = if rad { "Spec tokens" } else { "Draft max" };
            if m < 35.0 {
                out.push((3, insight(&id("mtp-low"), scope, "serious", &format!("Draft acceptance only {m:.0}%"),
                    if rad {
                        "Most drafted tokens are thrown away. Try Draft window = adaptive (it stops drafting where acceptance falls off), a lower Spec tokens, or 0 to compare - Radiance's advice is to judge speculation by step time, not acceptance.".to_string()
                    } else {
                        format!("Most drafted tokens are thrown away, so MTP may be costing speed. Try {knob} {} or turn MTP off and compare generation speed.", (n - 1).max(1))
                    })));
            } else if m > 70.0 && n > 0 && n < 4 {
                out.push((6, insight(&id("mtp-high"), scope, "info", &format!("Draft acceptance {m:.0}%"),
                    format!("Drafts are accepted most of the time; a higher {knob} (now {n}) may be faster."))));
            }
        }
        let big_prompts = done.iter().filter(|r| r.get("total").and_then(|v| v.as_f64()).unwrap_or(0.0) > 8192.0).count();
        if let Some(c) = num(session, "cache") {
            if c < 30.0 && big_prompts >= 2 {
                let reread: f64 = done.iter().map(|r| r.get("pMs").and_then(|v| v.as_f64()).unwrap_or(0.0)).sum::<f64>() / 1000.0;
                out.push((2, insight(&id("cache-low"), scope, "warning", &format!("Prompt cache reuse {c:.0}%"),
                    format!("Each request re-reads most of its prompt ({reread:.0} s spent on prompts this run). Usually the chat app changes text near the start of the prompt (trimmed history, timestamps, lorebook entries); keeping the start stable lets the server reuse the cache."))));
            }
        }
        if let Some(last) = done.last() {
            if let (Some(n), Some(c)) = (num(last, "nEnd"), num(last, "ctx")) {
                if c > 0.0 && n / c > 0.85 {
                    out.push((2, insight(&id("ctx-full"), scope, "warning", &format!("Context {:.0}% full", 100.0 * n / c),
                        format!("The last request ended at {} of {} tokens. When it fills up, requests fail or old text is dropped.", fmt_thousands(n as i64), fmt_thousands(c as i64)))));
                }
            }
        }
        if pane.is_rad() {
            if let Some(pre) = metrics.get("radiance:num_requests_preempted").filter(|x| **x > 0.0) {
                out.push((2, insight(&id("preempted"), scope, "warning", &format!("{} request(s) preempted", *pre as i64),
                    "The KV pool ran out while requests were running, so Radiance paused one to make room. Lower Max seqs or Context, use an fp8 KV cache, or move the vision tower to RAM.".into())));
            }
            let seqs = pane.loaded.as_ref().map(|(_, o, _)| gi(o, "parallel", 1)).unwrap_or(1);
            if let (Some(b), true) = (pane.r_backs.filter(|b| *b < 99.5), seqs <= 1) {
                let lim = pane.r_kv_tokens.map(|t| format!(" (about {} tokens)", fmt_thousands(t))).unwrap_or_default();
                out.push((2, insight(&id("pool-short"), scope, "warning", &format!("Context is larger than the KV pool ({b:.0}%)"),
                    format!("A prompt longer than the pool{lim} is refused. Press Fit (it sets Context to what the pool holds), or use an fp8 KV cache."))));
            }
            let ev = metrics.get("radiance:prefix_cache_evictions_total").cloned().unwrap_or(0.0);
            let tier_off = pane.loaded.as_ref().map(|(_, o, _)| gi(o, "r_prefix_mib", 0) == 0).unwrap_or(true);
            if ev > 0.0 && tier_off {
                out.push((6, insight(&id("evictions"), scope, "info", &format!("{} prefix-cache evictions", ev as i64),
                    "The pool dropped cached prompts to make room; a later turn that wanted them re-reads its whole context. Prefix cache (MiB) keeps finished turns in system RAM.".into())));
            }
        }
        if let Some(d) = metrics.get("requests_deferred") {
            if *d > 0.0 {
                out.push((4, insight(&id("queue"), scope, "info", "Requests are queueing",
                    format!("{d:.0} request(s) waiting for a free slot. Raise Parallel if several clients use this server."))));
            }
        }
        out
    }

    // ============================================================= history ==
    pub fn dash_history(&self, arg: &Value) -> Value {
        let mut rows = vec![];
        let stats = self.settings.raw_section("stats");
        let mut current_ids: Vec<(String, String)> = vec![];
        for (k, v) in stats.iter() {
            let Some((key, rel)) = k.split_once('|') else { continue };
            let Some(st) = v.as_object() else { continue };
            current_ids.push((key.to_string(), rel.to_string()));
            rows.push(stats_row(key, rel, st, "current"));
        }
        // the history file: retired totals, and per-request records for the detail view
        let want = arg.get("id").and_then(|v| v.as_str()).map(String::from);
        let want_kr: Option<(String, String)> = arg.get("key").and_then(|v| v.as_str()).zip(arg.get("model").and_then(|v| v.as_str()))
            .map(|(a, b)| (a.to_string(), b.to_string()));
        let mut detail_reqs: Vec<Value> = vec![];
        let mut detail_events: Vec<Value> = vec![];
        let mut per_key_runs: HashMap<(String, String), i64> = HashMap::new();
        let mut first_t: Option<f64> = None;
        if let Ok(f) = std::fs::File::open(history_path()) {
            for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
                let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                let kind = v.get("kind").and_then(|x| x.as_str()).unwrap_or("");
                let key = v.get("key").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let rel = v.get("model").and_then(|x| x.as_str()).unwrap_or("").to_string();
                if first_t.is_none() {
                    first_t = v.get("t").and_then(|x| x.as_f64());
                }
                if kind == "retired" {
                    if let Some(st) = v.get("stats").and_then(|x| x.as_object()) {
                        if gi(st, "loads", 0) > 0 || gi(st, "requests", 0) > 0 {
                            let mut row = stats_row(&key, &rel, st, "earlier");
                            row["retired"] = v.get("when").cloned().unwrap_or(Value::Null);
                            rows.push(row);
                        }
                    }
                    continue;
                }
                if kind == "load" {
                    *per_key_runs.entry((key.clone(), rel.clone())).or_default() += 1;
                }
                if want_kr.as_ref() == Some(&(key.clone(), rel.clone())) {
                    match kind {
                        "request" | "abort" => detail_reqs.push(v),
                        _ => detail_events.push(v),
                    }
                }
            }
        }
        let keep = detail_reqs.len().saturating_sub(2000);
        let detail = want_kr.map(|(k, r)| json!({"key": k, "model": r, "id": want, "requests": detail_reqs.split_off(keep),
                                                 "events": detail_events}));
        // totals
        let tot = |k: &str| rows.iter().map(|r| r[k].as_f64().unwrap_or(0.0)).sum::<f64>();
        let models: std::collections::HashSet<&str> = rows.iter().filter_map(|r| r["model"].as_str()).collect();
        json!({
            "rows": rows,
            "totals": {"configs": rows.len(), "models": models.len(), "runs": tot("runs"), "requests": tot("requests"),
                       "aborts": tot("aborts"), "crashes": tot("crashes"), "loadFailed": tot("loadFailed"), "spills": tot("spills"),
                       "genTokens": tot("genTokens"), "promptTokens": tot("promptTokens")},
            "historySince": first_t,
            "detail": detail,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn helper_cancel_is_not_an_abort() {
        // real llama-server log (Oct 2026): the chat app cancels task 772 while task 771 runs
        let log = include_str!("../../tests/llama_log_cancel.txt");
        std::env::set_var("LLM_DUO_DIR", std::env::temp_dir().join(format!("llmduo-dash-{}", std::process::id())));
        let mut app = App::new();
        for l in log.lines() {
            app.parse_line(0, l);
        }
        let r = &app.panes[0].dash_reqs;
        assert_eq!(r.len(), 4, "{r:?}");
        assert!(r.iter().all(|x| !truthy(x.get("aborted"))), "{r:?}");
        assert_eq!(r[1]["gTok"], 1312);
        assert!(app.panes[0].dash_active.is_none());
    }
    #[test]
    fn keys() {
        assert_eq!(describe_key("Vulkan:DUAL:AMD Radeon AI PRO R9700+AMD Radeon RX 9070 XT"),
                   ("llama.cpp".into(), "Vulkan".into(), "Dual: AI PRO R9700 + RX 9070 XT".into(), true));
        assert_eq!(describe_key("AMD Radeon RX 9070 XT"), ("llama.cpp".into(), "ROCm".into(), "RX 9070 XT".into(), false));
        assert_eq!(describe_key("Kobold-Vulkan:AMD Radeon RX 9070 XT").1, "Vulkan");
        assert_eq!(describe_key("vLLM:DUAL:A+B").0, "vLLM");
    }
}
