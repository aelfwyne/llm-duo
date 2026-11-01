//! Small helpers that the Python original got from its standard library:
//! shlex, Python-style rounding / JSON dumps, local time stamps, subprocesses
//! with a timeout, and dynamic option maps.

use serde_json::{Map, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const MIB: f64 = 1024.0 * 1024.0;
pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Options are kept as a JSON object, exactly like the Python dicts, so that
/// settings files written by the Python version keep working.
pub type Opts = Map<String, Value>;

pub fn v_i64(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)),
        _ => None,
    }
}

pub fn v_f64(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(*b as i64 as f64),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

pub fn gi(o: &Opts, k: &str, d: i64) -> i64 {
    v_i64(o.get(k)).unwrap_or(d)
}
pub fn gf(o: &Opts, k: &str, d: f64) -> f64 {
    v_f64(o.get(k)).unwrap_or(d)
}
pub fn gb(o: &Opts, k: &str, d: bool) -> bool {
    match o.get(k) {
        None | Some(Value::Null) => d,
        v => truthy(v),
    }
}
pub fn gs(o: &Opts, k: &str, d: &str) -> String {
    match o.get(k) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => d.to_string(),
        Some(v) => py_str(v),
    }
}

/// `str(value)` the way Python prints a scalar.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Null => "None".into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_float(n.as_f64().unwrap_or(0.0))
            }
        }
        other => py_json(other),
    }
}

pub fn py_float(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{:.1}", f)
    } else if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else {
        format!("{}", f)
    }
}

/// Python's `json.dumps(v, sort_keys=True)` (default separators, ensure_ascii).
/// Used for the stats fingerprint so stats recorded by the Python app still match.
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    fn esc(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{:04x}", u));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    fn rec(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    out.push_str(&i.to_string())
                } else if let Some(u) = n.as_u64() {
                    out.push_str(&u.to_string())
                } else {
                    out.push_str(&py_float(n.as_f64().unwrap_or(0.0)))
                }
            }
            Value::String(s) => esc(s, out),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    rec(x, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    esc(k, out);
                    out.push_str(": ");
                    rec(&o[*k], out);
                }
                out.push('}');
            }
        }
    }
    rec(v, &mut out);
    out
}

/// Python 3 `round()` for floats: round half to even.
pub fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc();
        let up = t + x.signum();
        if (t as i64) % 2 == 0 { t as i64 } else { up as i64 }
    } else {
        r as i64
    }
}

/// Python floor division for possibly negative ints.
pub fn floordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

pub fn fmt_thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

// ------------------------------------------------------------------ shlex --
pub fn shlex_split(s: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_tok = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_tok {
                    out.push(std::mem::take(&mut cur));
                    in_tok = false;
                }
            }
            '\'' => {
                in_tok = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(x) => cur.push(x),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            '"' => {
                in_tok = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(x) if matches!(x, '\\' | '"' | '$' | '`' | '\n') => cur.push(x),
                            Some(x) => {
                                cur.push('\\');
                                cur.push(x);
                            }
                            None => return Err("No closing quotation".into()),
                        },
                        Some(x) => cur.push(x),
                        None => return Err("No closing quotation".into()),
                    }
                }
            }
            '\\' => {
                in_tok = true;
                match chars.next() {
                    Some(x) => cur.push(x),
                    None => return Err("No escaped character".into()),
                }
            }
            c => {
                in_tok = true;
                cur.push(c);
            }
        }
    }
    if in_tok {
        out.push(cur);
    }
    Ok(out)
}

pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c)) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

pub fn shlex_join<S: AsRef<str>>(v: &[S]) -> String {
    v.iter().map(|x| shlex_quote(x.as_ref())).collect::<Vec<_>>().join(" ")
}

// ------------------------------------------------------------------ time --
fn local_tm() -> libc::tm {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        tm
    }
}

/// `datetime.now().isoformat(timespec="seconds")`
/// A byte count as JSON: an integer when it is whole (as the Python version wrote it).
pub fn jnum(x: f64) -> serde_json::Value {
    if x.fract() == 0.0 && x.abs() < 9.0e15 { serde_json::Value::from(x as i64) } else { serde_json::Value::from(x) }
}

pub fn now_iso() -> String {
    let t = local_tm();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec
    )
}

/// `%Y%m%d-%H%M%S`
pub fn now_stamp() -> String {
    let t = local_tm();
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec
    )
}

// ------------------------------------------------------------- processes --
pub struct RunOut {
    pub stdout: String,
    pub stderr: String,
    pub status: Option<i32>,
}

pub enum RunErr {
    NotFound,
    PermissionDenied,
    Timeout,
    Other(String),
}

impl std::fmt::Display for RunErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunErr::NotFound => write!(f, "No such file or directory"),
            RunErr::PermissionDenied => write!(f, "Permission denied"),
            RunErr::Timeout => write!(f, "timed out"),
            RunErr::Other(s) => write!(f, "{s}"),
        }
    }
}

/// `subprocess.run(args, capture_output=True, text=True, timeout=…)`
pub fn run_capture(args: &[String], timeout: Duration, env: Option<&Vec<(String, String)>>) -> Result<RunOut, RunErr> {
    if args.is_empty() {
        return Err(RunErr::Other("empty command".into()));
    }
    let mut cmd = Command::new(&args[0]);
    cmd.args(&args[1..]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(env) = env {
        cmd.env_clear();
        for (k, v) in env {
            cmd.env(k, v);
        }
    }
    let mut child = cmd.spawn().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => RunErr::NotFound,
        std::io::ErrorKind::PermissionDenied => RunErr::PermissionDenied,
        _ => RunErr::Other(e.to_string()),
    })?;
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t1 = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = so.read_to_end(&mut b);
        b
    });
    let t2 = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = se.read_to_end(&mut b);
        b
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st.code(),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RunErr::Timeout);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(RunErr::Other(e.to_string())),
        }
    };
    let so = t1.join().unwrap_or_default();
    let se = t2.join().unwrap_or_default();
    Ok(RunOut {
        stdout: String::from_utf8_lossy(&so).into_owned(),
        stderr: String::from_utf8_lossy(&se).into_owned(),
        status,
    })
}

pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if is_executable(&p) {
            return Some(p);
        }
    }
    None
}

pub fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.is_file() && p.metadata().map(|m| m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}

pub fn access_x(p: &str) -> bool {
    let c = match std::ffi::CString::new(p) {
        Ok(c) => c,
        Err(_) => return false,
    };
    unsafe { libc::access(c.as_ptr(), libc::X_OK) == 0 }
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// `Path(p).expanduser()`
pub fn expanduser(p: &str) -> PathBuf {
    if p == "~" {
        home()
    } else if let Some(rest) = p.strip_prefix("~/") {
        home().join(rest)
    } else {
        PathBuf::from(p)
    }
}

pub fn file_key(p: &Path) -> Option<(f64, u64)> {
    let m = std::fs::metadata(p).ok()?;
    let mt = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Some((mt, m.len()))
}

pub fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

/// Last `n` characters of a string (Python `s[-n:]`).
pub fn tail_chars(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    if c.len() <= n { s.to_string() } else { c[c.len() - n..].iter().collect() }
}

// ------------------------------------------------------------ app dirs --
pub struct Dirs {
    pub config: PathBuf,
    pub data: PathBuf,
}

static DIRS: OnceLock<Dirs> = OnceLock::new();

/// Where settings, logs, templates and tools live.
///
/// * `$LLM_DUO_DIR` if set: everything in that one folder (like the Python app)
/// * else, if `llm_duo_settings.json` sits next to the executable: that folder ("portable")
/// * else settings in `$XDG_CONFIG_HOME/llm-duo`, logs/templates/tools in `$XDG_DATA_HOME/llm-duo`
pub fn dirs() -> &'static Dirs {
    DIRS.get_or_init(|| {
        if let Some(d) = std::env::var_os("LLM_DUO_DIR") {
            let d = PathBuf::from(d);
            return Dirs { config: d.clone(), data: d };
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                if dir.join("llm_duo_settings.json").is_file() {
                    return Dirs { config: dir.to_path_buf(), data: dir.to_path_buf() };
                }
            }
        }
        let cfg = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| home().join(".config"))
            .join("llm-duo");
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| home().join(".local/share"))
            .join("llm-duo");
        Dirs { config: cfg, data }
    })
}

pub fn log_dir() -> PathBuf {
    dirs().data.join("logs")
}

pub fn write_log(name: &str, text: &str) -> std::io::Result<PathBuf> {
    let d = log_dir();
    std::fs::create_dir_all(&d)?;
    let safe: String = {
        let mut s = String::new();
        let mut last_us = false;
        for c in name.chars() {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                s.push(c);
                last_us = false;
            } else if !last_us {
                s.push('_');
                last_us = true;
            }
        }
        s.chars().take(80).collect()
    };
    let p = d.join(format!("{}_{}.log", now_stamp(), safe));
    std::fs::write(&p, text)?;
    Ok(p)
}

pub fn open_url(url: &str) {
    let _ = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding() {
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(3.5), 4);
        assert_eq!(py_round(-2.5), -2);
        assert_eq!(py_round(2.4), 2);
        assert_eq!(py_round(2.6), 3);
        assert_eq!(floordiv(-7, 2), -4);
    }
    #[test]
    fn shlex() {
        assert_eq!(shlex_split(r#"--a "b c" 'd e' f\ g"#).unwrap(), vec!["--a", "b c", "d e", "f g"]);
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("blk\\.(1|2)"), "'blk\\.(1|2)'");
        assert!(shlex_split("'abc").is_err());
    }
    #[test]
    fn pyjson() {
        let v: Value = serde_json::from_str(r#"{"b": 1.0, "a": [true, null, "é"], "c": 0.9}"#).unwrap();
        assert_eq!(py_json(&v), "{\"a\": [true, null, \"\\u00e9\"], \"b\": 1.0, \"c\": 0.9}");
        assert_eq!(fmt_thousands(1234567), "1,234,567");
    }
}

/// The flags that carry a secret on a server command line.
pub const SECRET_FLAGS: [&str; 2] = ["--api-key", "--password"];

/// A command line for showing (log, reports): every API key / password replaced by ••••.
/// `K=V K=V` for display, with values that hold a secret masked.
pub fn env_text(env: &[(String, String)]) -> String {
    env.iter()
        .map(|(k, v)| if k.ends_with("API_KEY") { format!("{k}=••••") } else { format!("{k}={v}") })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn mask_cmd(cmd: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(cmd.len());
    let mut hide = false;
    for a in cmd {
        if hide {
            out.push("••••".to_string());
            hide = false;
            continue;
        }
        if let Some(f) = SECRET_FLAGS.iter().find(|f| a.starts_with(&format!("{f}="))) {
            out.push(format!("{f}=••••"));
            continue;
        }
        hide = SECRET_FLAGS.contains(&a.as_str());
        out.push(a.clone());
    }
    out
}

#[cfg(test)]
mod mask_tests {
    use super::*;
    #[test]
    fn masks() {
        let c: Vec<String> = ["srv", "--api-key", "abc", "--port", "1", "--password=x"].iter().map(|s| s.to_string()).collect();
        assert_eq!(mask_cmd(&c), vec!["srv", "--api-key", "••••", "--port", "1", "--password=••••"]);
        let e = [("A".to_string(), "1".to_string()), ("RADIANCE_API_KEY".to_string(), "abc".to_string())];
        assert_eq!(env_text(&e), "A=1 RADIANCE_API_KEY=••••");
    }
}
