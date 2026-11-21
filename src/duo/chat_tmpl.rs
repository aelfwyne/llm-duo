//! A checkpoint's own chat template, for calibration corpora written as chat rows
//! (`{"messages": [...], "tools": ..., "thinking": ...}`, the format of quant-calibration sets).
//!
//! Radiance's calibration mode reads only `prompt` strings, tokenised as the server tokenises a
//! rendered template (control tokens parsed, nothing prepended), so a chat row is rendered here
//! the way transformers' `apply_chat_template` renders it: the template from
//! `chat_template.jinja` (or `tokenizer_config.json`), trim_blocks + lstrip_blocks, the special
//! tokens from `tokenizer_config.json`, Python string methods, `raise_exception`,
//! `strftime_now`, and a `tojson` that writes what Python's `json.dumps` writes.
//!
//! Depends only on minijinja, minijinja-contrib, serde_json and libc.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value};
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
    /// `bos_token`, `eos_token`, … as `tokenizer_config.json` names them
    specials: Vec<(String, String)>,
    /// where the template came from, for the log
    pub source: String,
}

/// The template's text: `chat_template.jinja`, else `tokenizer_config.json`'s `chat_template`
/// (a string, or a list of named templates: "default", or "tool_use" when `tools` is wanted).
fn template_text(dir: &Path, cfg: &serde_json::Value) -> Result<(String, String), String> {
    let f = dir.join("chat_template.jinja");
    if let Ok(t) = std::fs::read_to_string(&f) {
        return Ok((t, f.display().to_string()));
    }
    let src = dir.join("tokenizer_config.json").display().to_string();
    match cfg.get("chat_template") {
        Some(serde_json::Value::String(t)) => Ok((t.clone(), src)),
        Some(serde_json::Value::Array(list)) => {
            let named = |n: &str| {
                list.iter()
                    .find(|e| e.get("name").and_then(|x| x.as_str()) == Some(n))
                    .and_then(|e| e.get("template").and_then(|x| x.as_str()))
                    .map(String::from)
            };
            named("default")
                .or_else(|| named("tool_use"))
                .map(|t| (t, format!("{src} (default)")))
                .ok_or_else(|| format!("{src}: chat_template lists no \"default\" template"))
        }
        _ => Err(format!("{} has no chat template (no chat_template.jinja, and tokenizer_config.json names none)", dir.display())),
    }
}

impl ChatTemplate {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let cfg: serde_json::Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let (text, source) = template_text(dir, &cfg)?;
        let mut specials = vec![];
        if let Some(o) = cfg.as_object() {
            for (k, v) in o {
                if !k.ends_with("_token") {
                    continue;
                }
                let s = v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str()));
                if let Some(s) = s {
                    specials.push((k.clone(), s.to_string()));
                }
            }
        }
        Self::from_text(text, specials, source)
    }

    pub fn from_text(text: String, specials: Vec<(String, String)>, source: String) -> Result<Self, String> {
        let mut env = Environment::new();
        // transformers renders with trim_blocks and lstrip_blocks
        let syntax = SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().map_err(|e| e.to_string())?;
        env.set_syntax(syntax);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        env.add_template_owned("chat", text).map_err(|e| format!("{source}: {}", err_text(&e)))?;
        Ok(ChatTemplate { env, specials, source })
    }

    /// Renders one chat row. `row` is the whole JSONL object, parsed in order
    /// (`serde_json::from_str::<minijinja::Value>`). The generation prompt is added when the
    /// conversation doesn't end with an assistant turn; `thinking` (if the row has it) becomes
    /// `enable_thinking`; tool-call arguments given as JSON strings become mappings.
    pub fn render(&self, row: &Value) -> Result<String, String> {
        let msgs = row.get_attr("messages").map_err(|e| err_text(&e))?;
        if msgs.kind() != ValueKind::Seq || msgs.len() == Some(0) {
            return Err("\"messages\" is not a non-empty list".into());
        }
        let msgs: Vec<Value> = msgs.try_iter().map_err(|e| err_text(&e))?.map(|m| fix_message(&m)).collect();
        let last_role = msgs.last().and_then(|m| m.get_attr("role").ok()).map(|r| r.to_string()).unwrap_or_default();
        let mut ctx: Vec<(String, Value)> = vec![
            ("messages".into(), Value::from(msgs)),
            ("add_generation_prompt".into(), Value::from(last_role != "assistant")),
        ];
        let tools = row.get_attr("tools").unwrap_or_default();
        let tools = match tools.as_str() {
            Some(s) if !s.trim().is_empty() => serde_json::from_str::<Value>(s).map_err(|e| format!("\"tools\" is not JSON: {e}"))?,
            Some(_) => Value::UNDEFINED,
            None => tools,
        };
        if tools.kind() == ValueKind::Seq {
            ctx.push(("tools".into(), tools));
        }
        let th = row.get_attr("thinking").unwrap_or_default();
        if th.kind() == ValueKind::Bool {
            ctx.push(("enable_thinking".into(), th));
        }
        for (k, v) in &self.specials {
            ctx.push((k.clone(), Value::from(v.as_str())));
        }
        let t = self.env.get_template("chat").map_err(|e| err_text(&e))?;
        t.render(Value::from_pairs(ctx)).map_err(|e| err_text(&e))
    }
}

/// The message of a template error: `raise_exception`'s text, or minijinja's description.
pub fn err_text(e: &Error) -> String {
    match (e.kind(), e.detail()) {
        (ErrorKind::InvalidOperation, Some(d)) => d.to_string(),
        _ => e.to_string(),
    }
}

/// A copy of `map` with `key` set to `val`, keeping the key order.
fn with_key(map: &Value, key: &str, val: Value) -> Value {
    let keys: Vec<Value> = map.try_iter().map(|i| i.collect()).unwrap_or_default();
    Value::from_pairs(keys.into_iter().map(|k| {
        let v = if k.as_str() == Some(key) { val.clone() } else { map.get_item(&k).unwrap_or_default() };
        (k, v)
    }))
}

/// Tool-call arguments as a mapping, as templates expect (`arguments|items`); rows store them as
/// a JSON string in the OpenAI format.
fn fix_args(call: &Value) -> Value {
    if call.kind() != ValueKind::Map {
        return call.clone();
    }
    let f = call.get_attr("function").unwrap_or_default();
    if f.kind() == ValueKind::Map {
        return with_key(call, "function", fix_args(&f));
    }
    match call.get_attr("arguments").ok().and_then(|a| a.as_str().map(String::from)) {
        Some(s) => match serde_json::from_str::<Value>(&s) {
            Ok(v) => with_key(call, "arguments", v),
            Err(_) => call.clone(),
        },
        None => call.clone(),
    }
}

fn fix_message(m: &Value) -> Value {
    let calls = m.get_attr("tool_calls").unwrap_or_default();
    if calls.kind() != ValueKind::Seq {
        return m.clone();
    }
    let fixed: Vec<Value> = calls.try_iter().map(|i| i.map(|c| fix_args(&c)).collect()).unwrap_or_default();
    with_key(m, "tool_calls", Value::from(fixed))
}

fn strftime_now(fmt: String) -> String {
    let Ok(cfmt) = std::ffi::CString::new(fmt) else { return String::new() };
    let mut buf = [0u8; 256];
    // SAFETY: localtime_r fills `tm`; strftime writes at most buf.len() bytes and returns the count
    let n = unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), cfmt.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// `tojson` as transformers has it: Python's `json.dumps(x, ensure_ascii=False)`, so `", "` and
/// `": "` between items, nothing HTML-escaped, keys in their order (`sort_keys` sorts them).
fn tojson(v: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<Value> = kwargs.get("indent")?;
    let sort: Option<bool> = kwargs.get("sort_keys")?;
    let _: Option<Value> = kwargs.get("ensure_ascii")?;
    let _: Option<Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(i) if i.is_none() => None,
        Some(i) => Some(usize::try_from(i)?),
    };
    let mut out = String::new();
    write_json(v, indent, sort.unwrap_or(false), 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

fn write_json(v: &Value, indent: Option<usize>, sort: bool, depth: usize, out: &mut String) -> Result<(), Error> {
    let nl = |out: &mut String, d: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * d));
        }
    };
    let sep = if indent.is_some() { "," } else { ", " };
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number if v.is_integer() => out.push_str(&v.to_string()),
        ValueKind::Number => out.push_str(&py_float(f64::try_from(v.clone())?)),
        ValueKind::String => write_str(v.as_str().unwrap_or_default(), out),
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if sort {
                keys.sort_by_key(|k| k.to_string());
            }
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                nl(out, depth + 1);
                write_str(&k.as_str().map(String::from).unwrap_or_else(|| k.to_string()), out);
                out.push_str(": ");
                write_json(&v.get_item(k)?, indent, sort, depth + 1, out)?;
            }
            nl(out, depth);
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                nl(out, depth + 1);
                write_json(x, indent, sort, depth + 1, out)?;
            }
            nl(out, depth);
            out.push(']');
        }
        _ => write_str(&v.to_string(), out),
    }
    Ok(())
}

fn py_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
    } else if f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpl(t: &str) -> ChatTemplate {
        ChatTemplate::from_text(t.into(), vec![("bos_token".into(), "<s>".into())], "test".into()).unwrap()
    }

    fn row(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn renders_like_transformers() {
        let t = tmpl(
            "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}",
        );
        let r = t.render(&row(r#"{"messages": [{"role": "user", "content": "hi"}]}"#)).unwrap();
        assert_eq!(r, "<s><user>hi\n<assistant>");
        let r = t.render(&row(r#"{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}"#)).unwrap();
        assert_eq!(r, "<s><user>hi\n<assistant>yo\n");
    }

    #[test]
    fn tools_args_and_json() {
        let t = tmpl(
            "{% for t in tools %}{{ t | tojson }}{% endfor %}|{% for m in messages %}{% for c in m.tool_calls or [] %}{% for k, v in c.function.arguments|items %}{{ k }}={{ v }};{% endfor %}{% endfor %}{% endfor %}|{{ enable_thinking }}",
        );
        let r = t
            .render(&row(r#"{"thinking": false, "tools": "[{\"name\": \"f\", \"a\": \"<x>\", \"n\": [1, 2.0]}]", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{\"z\": 1, \"a\": \"b\"}"}}]}]}"#))
            .unwrap();
        assert_eq!(r, r#"{"name": "f", "a": "<x>", "n": [1, 2.0]}|z=1;a=b;|False"#);
    }

    #[test]
    fn python_methods_and_raise() {
        let t = tmpl("{{ messages[0].content.startswith('a') }}{% if messages|length > 1 %}{{ raise_exception('too many') }}{% endif %}");
        assert_eq!(t.render(&row(r#"{"messages": [{"role": "user", "content": "abc"}]}"#)).unwrap(), "True");
        let e = t.render(&row(r#"{"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}"#)).unwrap_err();
        assert_eq!(e, "too many");
    }
}
