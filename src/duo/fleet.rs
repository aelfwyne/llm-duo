//! Fleet mode: Radiance 2.x serving several models from one process (docs/MULTIMODEL.md).
//!
//! A *pin* is a frozen copy of a Radiance pane's flags plus the fleet keys of that model
//! (aliases, priority, pin, ttl, park, flex). The fleet YAML is assembled from the pins: each
//! pin's flags go through `rad-yaml` (Radiance's own parser), the model entry is taken from its
//! output, and the process-wide parts (server, cards, memory, scheduler, groups) are written here.
//!
//! Everything in this file is plain text and JSON work, with no Qt and no processes, so it is
//! tested on its own. The App side (pinning, launching, polling) is in `fleet_app.rs`.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

type Opts = Map<String, Value>;

/// The tag of the generated file (`config_path("fleet")`).
pub const FLEET_TAG: &str = "fleet";

/// Top-level YAML sections a hot reload can't apply: Radiance refuses `server:`/`cards:` changes
/// (`Fleet::reload`), and the host budget and disk tier are read once at start.
pub const RESTART_SECTIONS: [&str; 3] = ["server", "cards", "memory"];

fn s_of(o: &Opts, k: &str) -> String {
    match o.get(k) {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}
fn i_of(o: &Opts, k: &str) -> i64 {
    match o.get(k) {
        Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)).unwrap_or(0),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(0),
        _ => 0,
    }
}
fn b_of(o: &Opts, k: &str, d: bool) -> bool {
    match o.get(k) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => matches!(s.trim(), "1" | "true" | "yes" | "on"),
        Some(Value::Number(n)) => n.as_i64().unwrap_or(0) != 0,
        _ => d,
    }
}

// ================================================================ names ==
/// A model id the fleet accepts: Radiance reserves ':' (base:adapter); LLM Duo keeps names to
/// letters, digits, '.', '_' and '-' so they need no YAML quoting.
pub fn clean_name(s: &str) -> String {
    let n: String = s
        .trim()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' })
        .collect();
    n.trim_matches('-').to_string()
}

/// Why a typed pin name can't be used, with a suggestion; None when it can. Pin names are the
/// model ids requests send and the YAML keys of the file: letters, digits, '.', '_' and '-'.
pub fn name_problem(n: &str) -> Option<String> {
    let n = n.trim();
    if n.is_empty() {
        return Some("A model name can't be empty.".into());
    }
    if clean_name(n) == n {
        return None;
    }
    let why = if n.chars().any(|c| c.is_whitespace()) {
        "has a space (Radiance refuses spaces in model names)"
    } else if n.contains(':') {
        "has a ':' (Radiance keeps it for base:adapter)"
    } else {
        "has a character other than letters, digits, '.', '_' and '-'"
    };
    let s = clean_name(n);
    Some(if s.is_empty() { format!("'{n}' {why}.") } else { format!("'{n}' {why}. For example: {s}") })
}

/// The default pin name for a model path: the container's stem, as the pane serves it.
pub fn pin_name_for(path: &str) -> String {
    let p = path.trim_end_matches('/');
    let n = p.rsplit('/').next().unwrap_or(p);
    clean_name(n.strip_suffix(".rad").unwrap_or(n))
}

/// The name a pane's model is pinned under: the name of an existing pin of the same model on the
/// same card(s), so re-pinning offers to replace it even after it was renamed, else `default`.
/// With several such pins, the one named `default` wins, then the first by name.
pub fn pin_target_name(pins: &Map<String, Value>, dev_key: &str, model: &str, default: &str) -> String {
    let src = |p: &Value, k: &str| p.pointer(&format!("/src/{k}")).and_then(|v| v.as_str()).map(String::from);
    let mut same: Vec<&String> = pins
        .iter()
        .filter(|(_, p)| src(p, "dev_key").as_deref() == Some(dev_key) && src(p, "model").as_deref() == Some(model))
        .map(|(n, _)| n)
        .collect();
    same.sort();
    if same.iter().any(|n| *n == default) {
        return default.to_string();
    }
    same.first().map(|n| n.to_string()).unwrap_or_else(|| default.to_string())
}

/// `base`, or `base-2`, `base-3` … whichever is not taken.
pub fn unique_name(base: &str, taken: &[String]) -> String {
    if !taken.iter().any(|t| t == base) {
        return base.to_string();
    }
    (2..).map(|i| format!("{base}-{i}")).find(|n| !taken.contains(n)).unwrap()
}

/// Aliases as typed, separated by commas ("default, chat"), each exactly as typed (trimmed).
/// Nothing is rewritten: one Radiance would refuse is reported by `bad_aliases`.
pub fn alias_list(s: &str) -> Vec<String> {
    s.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect()
}

/// The aliases Radiance refuses as model ids (`valid_name`: no ':', reserved for base:adapter,
/// no whitespace, no control characters), with why.
pub fn bad_aliases(s: &str) -> Vec<String> {
    alias_list(s)
        .into_iter()
        .filter_map(|a| {
            let why = if a.chars().any(|c| c.is_whitespace()) {
                "has a space"
            } else if a.contains(':') {
                "has a ':' (Radiance keeps it for base:adapter)"
            } else if a.chars().any(|c| c.is_control()) {
                "has a control character"
            } else {
                return None;
            };
            Some(format!("'{a}' {why}: Radiance refuses spaces and ':' in model names, e.g. '{}'", suggest_alias(&a)))
        })
        .collect()
}

fn suggest_alias(a: &str) -> String {
    let s: String = a.chars().map(|c| if c.is_whitespace() || c == ':' || c.is_control() { '-' } else { c }).collect();
    let mut out = String::new();
    for c in s.chars() {
        if !(c == '-' && out.ends_with('-')) {
            out.push(c);
        }
    }
    out.trim_matches('-').to_string()
}

// ================================================================= pins ==
/// A pane's radiance flag line -> the pin's flags: without the binary, without what the fleet
/// owns (`--host`, `--port`, the API key: it travels in the environment) or replaces with the
/// pin's name (`--served-model-name`).
pub fn pin_flags(cmd: &[String]) -> Vec<String> {
    const DROP: [&str; 4] = ["--host", "--port", "--served-model-name", "--api-key"];
    let mut out = vec![];
    let mut i = 1;
    while i < cmd.len() {
        let a = &cmd[i];
        if DROP.contains(&a.as_str()) {
            i += 2;
            continue;
        }
        if DROP.iter().any(|d| a.starts_with(&format!("{d}="))) {
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

/// The tensor-parallel degree a pin's flags ask for (`--tp N`, 1 without it).
pub fn pin_tp(flags: &[String]) -> i64 {
    flag_value(flags, "--tp").and_then(|v| v.parse().ok()).unwrap_or(1).max(1)
}

/// The value of `flag` in a flag line (`--f v` or `--f=v`).
pub fn flag_value(flags: &[String], flag: &str) -> Option<String> {
    for (i, a) in flags.iter().enumerate() {
        if a == flag {
            return flags.get(i + 1).cloned();
        }
        if let Some(v) = a.strip_prefix(&format!("{flag}=")) {
            return Some(v.to_string());
        }
    }
    None
}

/// One pin's part of `rad-yaml` output.
#[derive(Debug, Default, PartialEq)]
pub struct PinYaml {
    /// the model entry's keys, indented four spaces, as `rad-yaml` printed them; a per-card memory
    /// key from `memory:` (headroom / memory MiB) moved in as the model's own
    pub body: Vec<String>,
    /// settings that are process-wide in a fleet and so can't be part of one pin ("server.log: debug")
    pub dropped: Vec<String>,
}

/// Split `rad-yaml`'s one-model file. Keys the fleet itself writes (server host/port/api-key-env)
/// are left out without a note; other process-wide keys are listed in `dropped`.
pub fn split_rad_yaml(yaml: &str) -> PinYaml {
    const OWN_SERVER: [&str; 3] = ["host", "port", "api-key-env"];
    const PER_MODEL_MEMORY: [&str; 2] = ["gpu-headroom-mib", "gpu-memory-mib"];
    let mut out = PinYaml::default();
    let mut moved = vec![];
    let mut section = String::new();
    for line in yaml.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent == 0 {
            let (k, v) = line.split_once(':').unwrap_or((line, ""));
            section = k.trim().to_string();
            if !v.trim().is_empty() {
                out.dropped.push(format!("{section}: {}", v.trim())); // e.g. cards: [0]
            }
            continue;
        }
        let key = line.trim().split(':').next().unwrap_or("").trim().to_string();
        match section.as_str() {
            "models" => {
                if indent >= 4 {
                    out.body.push(line.to_string());
                }
            }
            "server" if indent == 2 && OWN_SERVER.contains(&key.as_str()) => {}
            "server" if indent == 2 && key == "api-key" => {
                out.dropped.push("server.api-key (the fleet takes the key from Settings, through the environment)".into())
            }
            "memory" if indent == 2 && PER_MODEL_MEMORY.contains(&key.as_str()) => {
                moved.push(format!("    {}", line.trim()));
            }
            _ if indent == 2 => out.dropped.push(format!("{section}.{}", line.trim())),
            _ => {}
        }
    }
    // the model's own keys first, as rad-yaml sorts them; the moved card keys after
    out.body.extend(moved);
    out
}

/// The fleet keys of one pin as YAML lines (four-space indent). Only what is set is written:
/// everything else stays Radiance's default.
pub fn fleet_key_lines(f: &Opts) -> Vec<String> {
    let mut l = vec![];
    let aliases: Vec<String> = alias_list(&s_of(f, "aliases")).iter().map(|a| yaml_scalar(a)).collect();
    if !aliases.is_empty() {
        l.push(format!("    aliases: [{}]", aliases.join(", ")));
    }
    let pr = s_of(f, "priority");
    if let Ok(n) = pr.parse::<i64>() {
        l.push(format!("    priority: {n}"));
    }
    if b_of(f, "pin", false) {
        l.push("    pin: true".into());
    }
    let ttl = s_of(f, "ttl");
    if !ttl.is_empty() {
        l.push(format!("    ttl: {}", yaml_scalar(&ttl)));
    }
    let mut park = vec![];
    let ram = s_of(f, "park_ram");
    if !ram.is_empty() && ram != "allowed" {
        park.push(format!("ram: {ram}"));
    }
    let restore = s_of(f, "park_restore");
    if !restore.is_empty() && restore != "auto" {
        park.push(format!("restore: {restore}"));
    }
    if !park.is_empty() {
        l.push(format!("    park: {{{}}}", park.join(", ")));
    }
    let mut flex = vec![];
    let kv_min = i_of(f, "kv_min_tokens");
    let kv_max = i_of(f, "kv_max_tokens");
    let mut kv = vec![];
    if kv_min > 0 {
        kv.push(format!("min-tokens: {kv_min}"));
    }
    if kv_max > 0 {
        kv.push(format!("max-tokens: {kv_max}"));
    }
    if !kv.is_empty() {
        flex.push(format!("kv: {{{}}}", kv.join(", ")));
    }
    if let Ok(n) = s_of(f, "experts_max_mib").parse::<i64>() {
        flex.push(format!("experts: {{max-mib: {n}}}"));
    }
    if !flex.is_empty() {
        l.push(format!("    flex: {{{}}}", flex.join(", ")));
    }
    l
}

/// A plain YAML scalar, double-quoted when it would not read back as the same string.
pub fn yaml_scalar(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | '~'))
        && !matches!(s, "true" | "false" | "yes" | "no" | "on" | "off" | "null" | "~");
    if plain {
        s.to_string()
    } else {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

/// One model of the fleet file.
pub struct FleetModel {
    pub name: String,
    pub body: Vec<String>,
    pub keys: Vec<String>,
}

/// The whole fleet file. `server` is the fleet section's settings (`port`, `host_mib`,
/// `disk_tier`, `load_timeout`, `warm_up`); `groups` its list of {kind, members}.
pub fn fleet_yaml(server: &Opts, host: &str, key_env: Option<&str>, models: &[FleetModel], groups: &[Value]) -> String {
    let mut y = String::from("# Written by LLM Duo's Fleet mode; edits here are replaced on the next Load or Apply.\n");
    y += "server:\n";
    y += &format!("  host: {}\n", yaml_scalar(host));
    y += &format!("  port: {}\n", port_of(server));
    if let Some(e) = key_env {
        y += &format!("  api-key-env: {e}\n");
    }
    y += "cards: all\n";
    let mut mem = vec![];
    let hm = i_of(server, "host_mib");
    if hm > 0 {
        mem.push(format!("  host-mib: {hm}"));
    }
    let dt = s_of(server, "disk_tier");
    if !dt.is_empty() {
        mem.push(format!("  disk-tier: {}", yaml_scalar(&dt)));
    }
    if !mem.is_empty() {
        y += "memory:\n";
        y += &(mem.join("\n") + "\n");
    }
    let mut sch = vec![];
    let lt = s_of(server, "load_timeout");
    if !lt.is_empty() {
        sch.push(format!("  load-timeout: {}", yaml_scalar(&lt)));
    }
    // off unless asked for: Radiance's warm-up fills the cards with models nobody asked for, and
    // a model that never served a request is never parked by its ttl (Fleet::ttl_tick)
    if !b_of(server, "warm_up", false) {
        sch.push("  warm-up: false".into());
    }
    if !sch.is_empty() {
        y += "scheduler:\n";
        y += &(sch.join("\n") + "\n");
    }
    y += "models:\n";
    for m in models {
        y += &format!("  {}:\n", m.name);
        for l in m.body.iter().chain(m.keys.iter()) {
            y += l;
            y += "\n";
        }
    }
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    let gl = group_lines(groups, &names);
    if !gl.is_empty() {
        y += "groups:\n";
        for l in gl {
            y += &l;
            y += "\n";
        }
    }
    y
}

pub fn port_of(server: &Opts) -> i64 {
    match i_of(server, "port") {
        p if (1..=65535).contains(&p) => p,
        _ => 8080,
    }
}

/// The groups whose members are all in the file (at least two of them), as YAML lines.
fn group_lines(groups: &[Value], names: &[&str]) -> Vec<String> {
    let mut out = vec![];
    let mut n = 0;
    for g in groups {
        let Some((kind, members)) = group_of(g) else { continue };
        if members.len() < 2 || !members.iter().all(|m| names.contains(&m.as_str())) {
            continue;
        }
        n += 1;
        out.push(format!("  {kind}-{n}:"));
        out.push(format!("    {kind}: [{}]", members.join(", ")));
    }
    out
}

/// (kind, members) of a stored group, when it is one.
pub fn group_of(g: &Value) -> Option<(String, Vec<String>)> {
    let kind = g.get("kind")?.as_str()?.to_string();
    if kind != "together" && kind != "exclusive" {
        return None;
    }
    let members: Vec<String> = match g.get("members")? {
        Value::Array(a) => a.iter().filter_map(|v| v.as_str()).map(clean_name).filter(|s| !s.is_empty()).collect(),
        Value::String(s) => alias_list(s),
        _ => return None,
    };
    Some((kind, members))
}

/// The top-level sections of a YAML text, each with its lines.
pub fn top_sections(yaml: &str) -> BTreeMap<String, String> {
    let mut m: BTreeMap<String, String> = BTreeMap::new();
    let mut cur = String::new();
    for line in yaml.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') {
            cur = line.split(':').next().unwrap_or("").trim().to_string();
        }
        let e = m.entry(cur.clone()).or_default();
        e.push_str(line);
        e.push('\n');
    }
    m
}

/// Which sections that need a restart differ between the running file and a new one.
pub fn restart_sections(old: &str, new: &str) -> Vec<String> {
    let (a, b) = (top_sections(old), top_sections(new));
    RESTART_SECTIONS.iter().filter(|s| a.get(**s) != b.get(**s)).map(|s| s.to_string()).collect()
}

// =============================================================== checks ==
/// What a pin brings to the checks before a Load.
pub struct PinCheck {
    pub name: String,
    pub aliases: Vec<String>,
    /// aliases Radiance would refuse, with why (`bad_aliases`)
    pub bad_aliases: Vec<String>,
    pub tp: i64,
    pub model_exists: bool,
    /// estimated claim per card in bytes (0: unknown)
    pub claim: f64,
    /// pinned host memory the pin's flags ask for (prefix cache + host pool), MiB
    pub host_mib: i64,
    pub priority: Option<i64>,
    /// a generator whose Batch tokens can't hold its largest request (radiance::gen_batch_problem)
    pub gen_note: Option<String>,
}

/// (errors, warnings) for a fleet about to start. `card_bytes`: the smallest card's VRAM;
/// `cards`: how many there are; `host_budget_mib`: memory.host-mib or the default 75 % of RAM.
pub fn check_pins(pins: &[PinCheck], cards: usize, card_bytes: f64, host_budget_mib: i64) -> (Vec<String>, Vec<String>) {
    let (mut err, mut warn) = (vec![], vec![]);
    if pins.is_empty() {
        err.push("No pinned model is enabled. Tick at least one in the list.".into());
        return (err, warn);
    }
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for p in pins {
        for id in std::iter::once(&p.name).chain(p.aliases.iter()) {
            if let Some(other) = seen.get(id) {
                err.push(format!("'{id}' names both {other} and {} - a model name or alias must be unique.", p.name));
            } else {
                seen.insert(id.clone(), p.name.clone());
            }
        }
    }
    if pins.len() > 1 && !seen.contains_key("default") {
        warn.push(
            "No model is named or aliased 'default': a request without a \"model\" field will be refused. Add the alias 'default' to the model such requests should reach."
                .into(),
        );
    }
    for p in pins {
        for b in &p.bad_aliases {
            err.push(format!("{}: alias {b}.", p.name));
        }
        if !p.model_exists {
            err.push(format!("{}: its model file is gone. Re-pin it from a pane, or untick it.", p.name));
        }
        if p.tp as usize > cards {
            err.push(format!("{}: tp {} needs {} cards; {} are detected.", p.name, p.tp, p.tp, cards));
        }
        if let Some(n) = &p.gen_note {
            warn.push(format!("{}: {n}. Change it on the pane and re-pin.", p.name));
        }
        if p.claim > 0.0 && card_bytes > 0.0 && p.claim > card_bytes {
            warn.push(format!(
                "{}: its estimated claim (≈ {:.1} GiB a card) is more than a card holds ({:.1} GiB). Lower its Context on the pane and re-pin, or it may never load.",
                p.name,
                p.claim / GIB,
                card_bytes / GIB
            ));
        }
    }
    let host: i64 = pins.iter().map(|p| p.host_mib).sum();
    if host_budget_mib > 0 && host > host_budget_mib {
        warn.push(format!(
            "The pins' prefix caches and host pools pin {:.1} GiB of host RAM together, over the fleet's host budget ({:.1} GiB). Parked copies share that budget, so models will unload rather than park; lower Prefix cache on some pins or raise Host memory.",
            host as f64 / 1024.0,
            host_budget_mib as f64 / 1024.0
        ));
    }
    let with = pins.iter().filter(|p| p.priority.is_some()).count();
    if with > 0 && with < pins.len() {
        warn.push(
            "Priority is set on some models and not on others. Radiance never evicts a higher-priority model for a lower one, so a low-priority model whose room only higher-priority models hold waits out the load timeout. Leave Priority empty on all of them for least-recently-used swapping (MULTIMODEL.md §6.4)."
                .into(),
        );
    }
    (err, warn)
}

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// =============================================================== status ==
/// One row of the fleet table, from `/admin/fleet`'s `models` array.
pub fn status_rows(v: &Value) -> Vec<Value> {
    let Some(models) = v.get("models").and_then(|m| m.as_array()) else { return vec![] };
    models
        .iter()
        .map(|m| {
            let st = m.get("state").and_then(|s| s.as_str()).unwrap_or("?").to_string();
            let state = match (st.as_str(), m.get("doing").and_then(|d| d.as_str())) {
                ("waiting", _) => format!("waiting for {}", m.get("waiting_for").and_then(|w| w.as_str()).unwrap_or("room")),
                (s, Some(d)) if !d.is_empty() => format!("{s} ({d})"),
                (s, _) => s.to_string(),
            };
            let cards: Vec<String> = m
                .get("cards")
                .and_then(|c| c.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_i64()).map(|x| x.to_string()).collect())
                .unwrap_or_default();
            let vram: Vec<String> = m
                .get("vram_bytes_by_card")
                .and_then(|c| c.as_object())
                .map(|o| {
                    o.iter().filter_map(|(k, b)| b.as_f64().map(|b| format!("{k}: {:.1}", b / GIB))).collect()
                })
                .unwrap_or_default();
            serde_json::json!({
                "id": m.get("id").and_then(|s| s.as_str()).unwrap_or(""),
                "state": state,
                "raw": st,
                "cards": cards.join(", "),
                "vram": if vram.is_empty() { String::new() } else { format!("{} GiB", vram.join("  ")) },
                "busy": m.get("in_flight").and_then(|x| x.as_i64()).unwrap_or(0),
                "queued": m.get("waiting").and_then(|x| x.as_i64()).unwrap_or(0),
                "pinned": m.get("pinned").and_then(|x| x.as_bool()).unwrap_or(false),
                "parks": m.get("parks").and_then(|x| x.as_i64()).unwrap_or(0),
                "restores": m.get("restores").and_then(|x| x.as_i64()).unwrap_or(0),
            })
        })
        .collect()
}

/// A one-line summary of the cards and host memory in `/admin/fleet`.
pub fn status_summary(v: &Value) -> String {
    let mut parts = vec![];
    if let Some(cards) = v.get("cards").and_then(|c| c.as_array()) {
        for c in cards {
            let (Some(n), Some(free)) = (c.get("card").and_then(|x| x.as_i64()), c.get("free_bytes").and_then(|x| x.as_f64())) else {
                continue;
            };
            parts.push(format!("card {n}: {:.1} GiB free", free / GIB));
        }
    }
    if let Some(h) = v.get("host") {
        if let (Some(held), Some(b)) = (h.get("held_bytes").and_then(|x| x.as_f64()), h.get("budget_bytes").and_then(|x| x.as_f64())) {
            parts.push(format!("host pinned {:.1} of {:.1} GiB", held / GIB, b / GIB));
        }
    }
    if let Some(q) = v.get("warm_up").and_then(|w| w.get("queued")).and_then(|x| x.as_i64()).filter(|q| *q > 0) {
        parts.push(format!("warm-up: {q} queued"));
    }
    parts.join("   •   ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn repin_finds_renamed_pin() {
        let pin = |dev: &str, model: &str| json!({"src": {"dev_key": dev, "model": model}});
        let mut pins = Map::new();
        pins.insert("Nemotron-Embed".into(), pin("R9700", "Nemotron-3.5-Embed-1B-BF16"));
        pins.insert("other".into(), pin("R9700", "x.rad"));
        let d = "Nemotron-3.5-Embed-1B-BF16";
        // renamed pin of the same model on the same card: its name, so Replace is offered
        assert_eq!(pin_target_name(&pins, "R9700", "Nemotron-3.5-Embed-1B-BF16", d), "Nemotron-Embed");
        // other card kind or model: the default name
        assert_eq!(pin_target_name(&pins, "DUAL:R9700", "Nemotron-3.5-Embed-1B-BF16", d), d);
        assert_eq!(pin_target_name(&pins, "R9700", "y.rad", "y"), "y");
        // a pin under the default name as well: that one
        pins.insert(d.into(), pin("R9700", "Nemotron-3.5-Embed-1B-BF16"));
        assert_eq!(pin_target_name(&pins, "R9700", "Nemotron-3.5-Embed-1B-BF16", d), d);
    }

    #[test]
    fn flags() {
        let cmd = v(&[
            "/r/radiance", "--model", "/m/x.rad", "--served-model-name", "x", "--host", "0.0.0.0", "--port=8080", "--max-model-len",
            "1000", "--api-key", "s3cret", "--tp", "2",
        ]);
        let f = pin_flags(&cmd);
        assert_eq!(f, v(&["--model", "/m/x.rad", "--max-model-len", "1000", "--tp", "2"]));
        assert_eq!(pin_tp(&f), 2);
        assert_eq!(pin_tp(&v(&["--model", "a"])), 1);
    }

    #[test]
    fn names() {
        assert_eq!(pin_name_for("/mnt/1TB/Radiance/Gemma-4:x.rad"), "Gemma-4-x");
        assert_eq!(pin_name_for("/m/ckpt-dir/"), "ckpt-dir");
        assert_eq!(unique_name("a", &v(&["a", "a-2"])), "a-3");
        assert_eq!(name_problem(" Qwen-3.8_27b "), None);
        assert_eq!(name_problem("Qwen Image").unwrap(), "'Qwen Image' has a space (Radiance refuses spaces in model names). For example: Qwen-Image");
        assert!(name_problem("a:b").unwrap().contains("base:adapter"));
        assert!(name_problem("a#b").unwrap().contains("other than"));
        assert!(name_problem("").is_some());
        assert_eq!(alias_list("default, Qwen Image ,, x:y"), v(&["default", "Qwen Image", "x:y"]));
        let b = bad_aliases("default, Qwen Image, x:y, Qwen-3.8");
        assert_eq!(b.len(), 2, "{b:?}");
        assert!(b[0].starts_with("'Qwen Image' has a space") && b[0].ends_with("'Qwen-Image'"), "{b:?}");
        assert!(b[1].starts_with("'x:y' has a ':'"), "{b:?}");
    }

    // what `rad-yaml --model /x.rad --served-model-name a --tp 2 --gpu-headroom-mib 256
    // --prefix-cache-mib 1000 --host 0.0.0.0 --port 8080 --max-model-len 1000` printed (2.1.0)
    const RAD_YAML: &str = "server:\n  host: 0.0.0.0\n  port: 8080\nmemory:\n  gpu-headroom-mib: 256\nmodels:\n  a:\n    model: /x.rad\n    tp: 2\n    prefix-cache-mib: 1000\n    max-model-len: 1000\n";

    #[test]
    fn split() {
        let p = split_rad_yaml(RAD_YAML);
        assert_eq!(
            p.body,
            v(&["    model: /x.rad", "    tp: 2", "    prefix-cache-mib: 1000", "    max-model-len: 1000", "    gpu-headroom-mib: 256"])
        );
        assert!(p.dropped.is_empty());
        let p = split_rad_yaml("server:\n  log: debug\n  api-key: s3cret\nscheduler:\n  stuck-exit: 0\nmodels:\n  a:\n    model: /x.rad\n");
        assert_eq!(p.body, v(&["    model: /x.rad"]));
        assert_eq!(p.dropped.len(), 3);
        assert!(p.dropped[0] == "server.log: debug");
        assert!(!p.dropped.iter().any(|d| d.contains("s3cret")));
        assert!(p.dropped[2] == "scheduler.stuck-exit: 0");
    }

    #[test]
    fn keys() {
        let f = json!({"aliases": "default", "priority": "", "pin": false, "ttl": "30m", "park_ram": "never",
                       "park_restore": "auto", "kv_max_tokens": 0, "experts_max_mib": ""});
        assert_eq!(fleet_key_lines(f.as_object().unwrap()), v(&["    aliases: [default]", "    ttl: 30m", "    park: {ram: never}"]));
        let f = json!({"priority": "100", "pin": true, "kv_min_tokens": 200000, "kv_max_tokens": 0, "experts_max_mib": "0"});
        assert_eq!(
            fleet_key_lines(f.as_object().unwrap()),
            v(&["    priority: 100", "    pin: true", "    flex: {kv: {min-tokens: 200000}, experts: {max-mib: 0}}"])
        );
        assert!(fleet_key_lines(&Map::new()).is_empty());
    }

    #[test]
    fn whole_file() {
        let a = FleetModel { name: "a".into(), body: split_rad_yaml(RAD_YAML).body, keys: v(&["    aliases: [default]"]) };
        let b = FleetModel { name: "b".into(), body: v(&["    model: /y.rad"]), keys: vec![] };
        let server = json!({"port": 8000, "host_mib": 40960, "disk_tier": "/mnt/1TB/rad cache", "load_timeout": "", "warm_up": false});
        let groups = vec![json!({"kind": "exclusive", "members": ["a", "b"]}), json!({"kind": "together", "members": ["a", "gone"]})];
        let y = fleet_yaml(server.as_object().unwrap(), "0.0.0.0", Some("RADIANCE_API_KEY"), &[a, b], &groups);
        let want = "# Written by LLM Duo's Fleet mode; edits here are replaced on the next Load or Apply.
server:
  host: 0.0.0.0
  port: 8000
  api-key-env: RADIANCE_API_KEY
cards: all
memory:
  host-mib: 40960
  disk-tier: \"/mnt/1TB/rad cache\"
scheduler:
  warm-up: false
models:
  a:
    model: /x.rad
    tp: 2
    prefix-cache-mib: 1000
    max-model-len: 1000
    gpu-headroom-mib: 256
    aliases: [default]
  b:
    model: /y.rad
groups:
  exclusive-1:
    exclusive: [a, b]
";
        assert_eq!(y, want);
        // warm-up is off unless asked for
        let bare = fleet_yaml(&Map::new(), "h", None, &[], &[]);
        assert!(bare.contains("scheduler:\n  warm-up: false\n"), "{bare}");
        let on = fleet_yaml(json!({"warm_up": true}).as_object().unwrap(), "h", None, &[], &[]);
        assert!(!on.contains("warm-up"), "{on}");
        let y2 = y.replace("port: 8000", "port: 8001");
        assert_eq!(restart_sections(&y, &y2), v(&["server"]));
        assert!(restart_sections(&y, &y.replace("model: /y.rad", "model: /z.rad")).is_empty());
    }

    #[test]
    fn checks() {
        let pc = |n: &str, al: &[&str], tp, claim, host, pr| PinCheck {
            name: n.into(),
            aliases: v(al),
            bad_aliases: vec![],
            tp,
            model_exists: true,
            claim,
            host_mib: host,
            priority: pr,
            gen_note: None,
        };
        let (e, w) = check_pins(&[pc("a", &["default"], 1, 0.0, 0, None), pc("b", &[], 2, 0.0, 0, None)], 2, 32.0 * GIB, 60000);
        assert!(e.is_empty() && w.is_empty(), "{e:?} {w:?}");
        let (e, w) = check_pins(&[pc("a", &["b"], 3, 40.0 * GIB, 40000, Some(1)), pc("b", &[], 1, 0.0, 30000, None)], 2, 32.0 * GIB, 60000);
        assert_eq!(e.len(), 2, "{e:?}"); // 'b' twice, tp 3
        assert_eq!(w.len(), 4, "{w:?}"); // no default, claim, host, priority
        assert!(!check_pins(&[], 2, 0.0, 0).0.is_empty());
        let mut g = pc("default", &[], 1, 0.0, 0, None);
        g.gen_note = Some("Batch tokens 16,384 is below".into());
        let (e, w) = check_pins(&[g], 2, 32.0 * GIB, 60000);
        assert!(e.is_empty() && w.len() == 1 && w[0].starts_with("default: Batch tokens"), "{e:?} {w:?}");
    }

    #[test]
    fn status() {
        let s = json!({"models": [
            {"id": "q", "state": "resident", "cards": [1], "vram_bytes_by_card": {"1": 16106127360u64}, "in_flight": 1, "waiting": 0, "pinned": false, "parks": 2, "restores": 1},
            {"id": "f", "state": "waiting", "waiting_for": "accelerator", "doing": "loading"},
            {"id": "g", "state": "loading", "doing": "restoring"}],
            "cards": [{"card": 0, "free_bytes": 2147483648u64}],
            "host": {"budget_bytes": 68719476736u64, "held_bytes": 10737418240u64},
            "warm_up": {"queued": 1}});
        let r = status_rows(&s);
        assert_eq!(r[0]["vram"], "1: 15.0 GiB");
        assert_eq!(r[0]["cards"], "1");
        assert_eq!(r[1]["state"], "waiting for accelerator");
        assert_eq!(r[2]["state"], "loading (restoring)");
        assert_eq!(status_summary(&s), "card 0: 2.0 GiB free   •   host pinned 10.0 of 64.0 GiB   •   warm-up: 1 queued");
    }
}
