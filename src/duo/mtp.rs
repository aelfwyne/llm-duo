//! MTP / draft helper discovery, relaxed chat templates and long-term stats.

use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::consts::STATS_IGNORE;
use super::gguf::{read_arch, ModelInfo, MtpCand};
use super::util::{dirs, fmt_thousands, gi, now_iso, py_json, Opts};

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).unwrap())
}

fn mtp_token() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?i)(^|[-_.\s])mtp([-_.\s]|$)")
}
fn assist_token() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(&R, r"(?i)(^|[-_.\s])assistant([-_.\s]|$)")
}

/// True for MTP drafter files that should not appear as main models.
pub fn is_helper_name(name: &str) -> bool {
    let n = name.to_lowercase();
    mtp_token().is_match(&n) || (n.contains("gemma") && assist_token().is_match(&n))
}

/// Size tokens like "31b", "e4b", "30b-a3b" (Python's lookbehind/lookahead regex, done by hand).
pub fn size_tokens(texts: &[Option<String>]) -> BTreeSet<String> {
    static FULL: OnceLock<Regex> = OnceLock::new();
    static BASE: OnceLock<Regex> = OnceLock::new();
    let full = re(&FULL, r"(?i)^e?\d+(?:\.\d+)?b-a\d+(?:\.\d+)?b");
    let base = re(&BASE, r"(?i)^e?\d+(?:\.\d+)?b");
    let mut out = BTreeSet::new();
    for t in texts.iter().flatten() {
        let b = t.as_bytes();
        let bad_before = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'.' || c.is_ascii_uppercase();
        let bad_after = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit() || c.is_ascii_uppercase();
        let mut i = 0;
        while i < b.len() {
            if i > 0 && bad_before(b[i - 1]) {
                i += 1;
                continue;
            }
            let rest = &t[i..];
            let mut hit: Option<usize> = None;
            for r in [full, base] {
                if let Some(m) = r.find(rest) {
                    let end = i + m.end();
                    if end >= b.len() || !bad_after(b[end]) {
                        hit = Some(end);
                        break;
                    }
                }
            }
            if let Some(end) = hit {
                out.insert(t[i..end].to_lowercase());
                i = end;
            } else {
                i += 1;
            }
            while i < b.len() && !t.is_char_boundary(i) {
                i += 1;
            }
        }
    }
    out
}

/// Drafter files near model_path, best first.
pub fn find_mtp_candidates(model_path: &Path, info: &ModelInfo) -> Vec<MtpCand> {
    let folder = model_path.parent().unwrap_or(Path::new("/")).to_path_buf();
    let mut files: BTreeSet<PathBuf> = BTreeSet::new();
    let gguf_in = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d)
            .map(|r| {
                r.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().map(|x| x == "gguf").unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default()
    };
    for p in gguf_in(&folder) {
        let nm = p.file_name().unwrap().to_string_lossy().into_owned();
        if p != model_path && is_helper_name(&nm) {
            files.insert(p);
        }
    }
    if let Ok(rd) = std::fs::read_dir(&folder) {
        for e in rd.filter_map(|e| e.ok()) {
            let sub = e.path();
            if sub.is_dir() && sub.file_name().map(|n| n.to_string_lossy().to_lowercase().contains("mtp")).unwrap_or(false) {
                files.extend(gguf_in(&sub));
            }
        }
    }
    let kvs = |k: &str| info.kv.get(k).map(|v| v.display());
    let target = size_tokens(&[
        model_path.file_name().map(|n| n.to_string_lossy().into_owned()),
        kvs("general.size_label"),
        kvs("general.name"),
        kvs("general.basename"),
        kvs("general.base_model.0.name"),
    ]);
    let target_arch = info.arch.clone();
    let mut out = Vec::new();
    for p in files {
        let arch = read_arch(&p).unwrap_or_else(|| "?".into());
        let parent_name = if p.parent() != Some(folder.as_path()) {
            p.parent().and_then(|x| x.file_name()).map(|n| n.to_string_lossy().into_owned())
        } else {
            Some(String::new())
        };
        let cand = size_tokens(&[p.file_name().map(|n| n.to_string_lossy().into_owned()), parent_name]);
        let arch_ok = arch == format!("{target_arch}-assistant")
            || arch == format!("{target_arch}_assistant")
            || arch.to_lowercase().contains("mtp");
        let inter: BTreeSet<String> = cand.intersection(&target).cloned().collect();
        let join = |s: &BTreeSet<String>| s.iter().cloned().collect::<Vec<_>>().join("/");
        let (note, ok) = if !arch_ok && arch != "?" {
            (format!("⚠ arch '{arch}' is not a {target_arch} drafter"), Some(false))
        } else if !cand.is_empty() && !target.is_empty() && inter.is_empty() {
            (format!("⚠ size mismatch: drafter {}, model {}", join(&cand), join(&target)), Some(false))
        } else if !inter.is_empty() {
            (format!("✓ matches {} ({arch})", join(&inter)), Some(true))
        } else {
            (format!("? size unknown ({arch}) - may not match"), None)
        };
        let size = std::fs::metadata(&p).map(|m| m.len() as f64).unwrap_or(0.0);
        let rel = p.strip_prefix(&folder).map(|r| r.to_string_lossy().into_owned()).unwrap_or_default();
        out.push(MtpCand { rel, arch, ok, note, size });
    }
    // best first: matching, then Q8_0, then smaller
    out.sort_by(|a, b| {
        let ka = (a.ok != Some(true), !a.rel.to_lowercase().contains("q8_0"));
        let kb = (b.ok != Some(true), !b.rel.to_lowercase().contains("q8_0"));
        ka.cmp(&kb).then(a.size.partial_cmp(&b.size).unwrap_or(std::cmp::Ordering::Equal))
    });
    out
}

// ------------------------------------------------------- relaxed template --
pub fn template_dir() -> PathBuf {
    dirs().data.join("templates")
}

fn raise_system() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    // Python used a backreference for the quote; spelled out for both quote kinds here.
    re(
        &R,
        r#"(?is)\{\{-?\s*raise_exception\(\s*(?:'[^']*?system message[^']*?'|"[^"]*?system message[^"]*?")\s*\)\s*-?\}\}"#,
    )
}

/// Turn 'system message must be first' errors into a normal system turn. Returns (text, count).
pub fn relax_template(tpl: &str) -> (String, usize) {
    static FOR: OnceLock<Regex> = OnceLock::new();
    static STOP: OnceLock<Regex> = OnceLock::new();
    static CONTENT: OnceLock<Regex> = OnceLock::new();
    let chatml = tpl.contains("<|im_start|>") && re(&FOR, r"for\s+message\s+in\s+messages").is_match(tpl);
    let repl = if chatml {
        "{{- '<|im_start|>system\\n' + (message.content if message.content is string else '') + '<|im_end|>\\n' }}"
    } else {
        ""
    };
    let stop = re(&STOP, r"\{%-?\s*(elif|else|endfor)\b");
    let content = re(&CONTENT, r"message\.content|render_content");
    let mut out = String::new();
    let mut last = 0;
    let mut n = 0;
    for m in raise_system().find_iter(tpl) {
        out.push_str(&tpl[last..m.start()]);
        n += 1;
        if !repl.is_empty() {
            // If the template already renders the system message after the check, only drop it.
            let rest = &tpl[m.end()..];
            let branch = match stop.find(rest) {
                Some(s) => &rest[..s.start()],
                None => rest,
            };
            if !(branch.contains("im_start|>system") || content.is_match(branch)) {
                out.push_str(repl);
            }
        }
        last = m.end();
    }
    out.push_str(&tpl[last..]);
    (out, n)
}

/// Write the relaxed template for a model to ./templates and return (path, n_fixed).
pub fn make_relaxed_template(tpl: Option<String>, model_path: &Path) -> Result<(PathBuf, usize), String> {
    let tpl = tpl.filter(|t| !t.is_empty()).ok_or("this model has no embedded chat template to relax")?;
    let (new, n) = relax_template(&tpl);
    let d = template_dir();
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    let stem = model_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    // a vLLM model is a folder: use its full name, like Path.stem would
    let stem = if model_path.is_dir() {
        model_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(stem)
    } else {
        stem
    };
    let out = d.join(format!("{stem}.relaxed.jinja"));
    std::fs::write(&out, new).map_err(|e| e.to_string())?;
    Ok((out, n))
}

// ------------------------------------------------------------------ stats --
/// Settings that define a 'setup'. Any change here starts the stats over.
pub fn stats_fingerprint(opts: &Opts) -> String {
    let mut m = serde_json::Map::new();
    for (k, v) in opts {
        if !STATS_IGNORE.contains(&k.as_str()) {
            m.insert(k.clone(), v.clone());
        }
    }
    py_json(&Value::Object(m))
}

pub fn new_stats(opts: &Opts) -> Opts {
    json!({"fp": stats_fingerprint(opts), "since": now_iso(), "last": null,
           "loads": 0, "crashes": 0, "spills": 0, "requests": 0,
           "prompt_tokens": 0, "prompt_ms": 0.0, "gen_tokens": 0, "gen_ms": 0.0,
           "draft_acc": 0, "draft_gen": 0,
           "gen_tps_min": null, "gen_tps_max": null,
           "big_prompt_tokens": 0, "big_prompt_ms": 0.0, "cache_tokens": 0, "cache_total": 0})
    .as_object()
    .unwrap()
    .clone()
}

fn f(st: &Opts, k: &str) -> f64 {
    super::util::gf(st, k, 0.0)
}

/// One-line summary for the Status panel.
pub fn stats_summary(st: Option<&Opts>) -> String {
    let since = |st: &Opts| st.get("since").and_then(|v| v.as_str()).unwrap_or("").chars().take(10).collect::<String>();
    let Some(st) = st else { return "—".into() };
    let req = gi(st, "requests", 0);
    let loads = gi(st, "loads", 0);
    if req == 0 {
        if loads != 0 {
            return format!("{loads} load(s), no completed requests yet  •  since {}", since(st));
        }
        return "—".into();
    }
    let mut parts = vec![format!("{} request{}", fmt_thousands(req), if req == 1 { "" } else { "s" })];
    if f(st, "big_prompt_ms") > 0.0 {
        parts.push(format!("prompt {:.0} t/s", f(st, "big_prompt_tokens") / (f(st, "big_prompt_ms") / 1000.0)));
    }
    if f(st, "gen_ms") > 0.0 {
        let mut g = format!("gen {:.1} t/s", f(st, "gen_tokens") / (f(st, "gen_ms") / 1000.0));
        if let (Some(mn), Some(mx)) = (
            st.get("gen_tps_min").and_then(|v| v.as_f64()),
            st.get("gen_tps_max").and_then(|v| v.as_f64()),
        ) {
            g += &format!(" ({:.1}–{:.1})", mn, mx);
        }
        parts.push(g);
    }
    if f(st, "cache_total") != 0.0 {
        parts.push(format!("cache {:.0}%", 100.0 * f(st, "cache_tokens") / f(st, "cache_total")));
    }
    if f(st, "draft_gen") != 0.0 {
        parts.push(format!("MTP {:.0}%", 100.0 * f(st, "draft_acc") / f(st, "draft_gen")));
    }
    let mut tail = format!("{loads} load{}", if loads == 1 { "" } else { "s" });
    let crashes = gi(st, "crashes", 0);
    if crashes != 0 {
        tail += &format!(", {crashes} crash{}", if crashes > 1 { "es" } else { "" });
    }
    let spills = gi(st, "spills", 0);
    if spills != 0 {
        tail += &format!(", {spills} RAM spill{}", if spills > 1 { "s" } else { "" });
    }
    parts.push(tail);
    parts.push(format!("since {}", since(st)));
    parts.join("  •  ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes() {
        let t = size_tokens(&[Some("gemma-4-31B-it-Q4_K_M.gguf".into())]);
        assert!(t.contains("31b"), "{t:?}");
        let t = size_tokens(&[Some("Qwen3-30B-A3B-Q8_0.gguf".into())]);
        assert!(t.contains("30b-a3b"), "{t:?}");
        let t = size_tokens(&[Some("gemma-3n-E4B".into())]);
        assert!(t.contains("e4b"), "{t:?}");
        let t = size_tokens(&[Some("x12b".into()), Some("3.5b".into())]);
        assert!(!t.contains("12b") && !t.contains("2b"), "{t:?}");
        assert!(t.contains("3.5b"));
        assert!(is_helper_name("gemma-4-31b-it-assistant-Q8_0.gguf"));
        assert!(is_helper_name("GLM-mtp-Q8_0.gguf"));
        assert!(!is_helper_name("Qwen3-30B.gguf"));
    }
    #[test]
    fn relax() {
        let tpl = "{%- for message in messages %}{%- if message.role == 'system' and not loop.first %}{{- raise_exception('System message must be at the beginning.') }}{%- elif x %}<|im_start|>{%- endif %}{%- endfor %}";
        let (out, n) = relax_template(tpl);
        assert_eq!(n, 1);
        assert!(out.contains("<|im_start|>system\\n"), "{out}");
        assert!(!out.contains("raise_exception"));
    }
}
