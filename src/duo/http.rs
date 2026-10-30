//! Minimal HTTP/1.1 GET for polling the local servers (/health, /metrics, /api/extra/perf).

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

pub struct Resp {
    pub status: u16,
    pub body: String,
}

pub fn get(host: &str, port: i64, path: &str, timeout: Duration) -> Result<Resp, String> {
    get_auth(host, port, path, timeout, None)
}

/// GET with `Authorization: Bearer <token>` when the server was started with an API key / password.
pub fn get_auth(host: &str, port: i64, path: &str, timeout: Duration, token: Option<&str>) -> Result<Resp, String> {
    let addr = (host, port as u16)
        .to_socket_addrs()
        .map_err(|e| e.to_string())?
        .next()
        .ok_or("no address")?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    s.set_read_timeout(Some(timeout)).ok();
    s.set_write_timeout(Some(timeout)).ok();
    let hosthdr = if host.contains(':') { format!("[{host}]:{port}") } else { format!("{host}:{port}") };
    let auth = token.filter(|t| !t.is_empty()).map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {hosthdr}\r\nUser-Agent: llm-duo\r\nAccept: */*\r\n{auth}Connection: close\r\n\r\n");
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let start = std::time::Instant::now();
    let mut chunk = [0u8; 65536];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                if buf.is_empty() {
                    return Err(e.to_string());
                }
                break;
            }
        }
        if start.elapsed() > timeout * 2 || buf.len() > 32 * 1024 * 1024 {
            break;
        }
    }
    let sep = buf.windows(4).position(|w| w == b"\r\n\r\n").ok_or("bad response")?;
    let head = String::from_utf8_lossy(&buf[..sep]).into_owned();
    let mut body = buf[sep + 4..].to_vec();
    let mut lines = head.lines();
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line.split_whitespace().nth(1).and_then(|x| x.parse().ok()).ok_or("bad status")?;
    let chunked = lines.any(|l| {
        let l = l.to_ascii_lowercase();
        l.starts_with("transfer-encoding:") && l.contains("chunked")
    });
    if chunked {
        body = dechunk(&body);
    }
    Ok(Resp { status, body: String::from_utf8_lossy(&body).into_owned() })
}

fn dechunk(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let Some(le) = b[i..].windows(2).position(|w| w == b"\r\n") else { break };
        let line = String::from_utf8_lossy(&b[i..i + le]).into_owned();
        let size = usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
        i += le + 2;
        if size == 0 {
            break;
        }
        let end = (i + size).min(b.len());
        out.extend_from_slice(&b[i..end]);
        i = end + 2;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunks() {
        assert_eq!(super::dechunk(b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n"), b"Wikipedia");
    }
}
