//! Server processes: spawning (pipe or pseudo-terminal), reader threads, stopping the whole
//! process group, plus the response chime.

use std::io::Read;
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::kobold::kre;
use super::util::{dirs, which};

pub enum ProcMsg {
    Line(String),
    Exit(u64, i32),
}

/// A running server. `id` identifies this launch (like `proc is self.proc` in Python).
pub struct Proc {
    pub id: u64,
    pub pid: i32,
    pub child: Arc<Mutex<Child>>,
}

fn new_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::Relaxed)
}

/// A pseudo-terminal for KoboldCpp's output -> (master, slave). Output processing is turned
/// off so lines arrive exactly as printed; a wide window keeps progress lines whole.
fn open_pty() -> std::io::Result<(OwnedFd, OwnedFd)> {
    unsafe {
        let mut master: libc::c_int = -1;
        let mut slave: libc::c_int = -1;
        if libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null(), std::ptr::null()) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(slave, &mut t) == 0 {
            t.c_oflag &= !libc::OPOST; // no \n -> \r\n translation
            t.c_lflag &= !(libc::ECHO | libc::ICANON);
            libc::tcsetattr(slave, libc::TCSANOW, &t);
        }
        let ws = libc::winsize { ws_row: 50, ws_col: 400, ws_xpixel: 0, ws_ypixel: 0 };
        libc::ioctl(slave, libc::TIOCSWINSZ, &ws);
        libc::fcntl(master, libc::F_SETFD, libc::FD_CLOEXEC);
        Ok((OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave)))
    }
}

/// Start `cmd` in its own session with the full environment `env`.
/// With `pty`, output goes through a pseudo-terminal (KoboldCpp line-buffers there).
pub fn spawn(cmd: &[String], env: &[(String, String)], pty: bool, tx: Sender<ProcMsg>) -> std::io::Result<Proc> {
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]).env_clear().envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    unsafe {
        c.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let id = new_id();
    let reader: Box<dyn Read + Send> = if pty {
        let (master, slave) = open_pty()?;
        let s1 = slave.try_clone()?;
        c.stdin(Stdio::null()).stdout(Stdio::from(slave)).stderr(Stdio::from(s1));
        Box::new(std::fs::File::from(master))
    } else {
        // one pipe for stdout + stderr, like Python's stderr=subprocess.STDOUT
        let (r, w) = std::io::pipe()?;
        let w2 = w.try_clone()?;
        c.stdin(Stdio::null()).stdout(w).stderr(w2);
        Box::new(r)
    };
    let child = c.spawn();
    drop(c); // close our copies of the write ends
    let child = child?;
    let pid = child.id() as i32;
    let child = Arc::new(Mutex::new(child));
    let ch = child.clone();
    std::thread::spawn(move || line_reader(reader, pty, tx, ch, id));
    Ok(Proc { id, pid, child })
}

fn wait_child(ch: &Arc<Mutex<Child>>) -> i32 {
    loop {
        {
            let mut c = ch.lock().unwrap();
            match c.try_wait() {
                Ok(Some(st)) => {
                    use std::os::unix::process::ExitStatusExt;
                    return st.code().unwrap_or_else(|| -st.signal().unwrap_or(1));
                }
                Ok(None) => {}
                Err(_) => return -1,
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Read server output: \r and \n both end a line (Python's universal newlines);
/// on a pseudo-terminal, terminal escape codes are dropped.
fn line_reader(mut f: Box<dyn Read + Send>, strip: bool, tx: Sender<ProcMsg>, ch: Arc<Mutex<Child>>, id: u64) {
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = [0u8; 65536];
    let ansi = &kre().ansi;
    let flush = |b: &[u8], tx: &Sender<ProcMsg>| {
        let s = String::from_utf8_lossy(b);
        let s = if strip { ansi.replace_all(&s, "").into_owned() } else { s.into_owned() };
        let _ = tx.send(ProcMsg::Line(s));
    };
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                pending.extend_from_slice(&buf[..n]);
                let mut start = 0;
                let mut i = 0;
                while i < pending.len() {
                    let c = pending[i];
                    if c == b'\n' || c == b'\r' {
                        let end = i;
                        // treat \r\n as one break
                        if c == b'\r' && i + 1 < pending.len() && pending[i + 1] == b'\n' {
                            i += 1;
                        } else if c == b'\r' && i + 1 == pending.len() {
                            // \r at the very end: wait for more data in case \n follows
                            break;
                        }
                        let line = pending[start..end].to_vec();
                        flush(&line, &tx);
                        start = i + 1;
                    }
                    i += 1;
                }
                pending.drain(..start);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break, // EIO: the process (and everything it started) closed it
        }
    }
    if !pending.is_empty() {
        let p = std::mem::take(&mut pending);
        let p: Vec<u8> = p.into_iter().filter(|c| *c != b'\r').collect();
        if !p.is_empty() {
            flush(&p, &tx);
        }
    }
    let rc = wait_child(&ch);
    let _ = tx.send(ProcMsg::Exit(id, rc));
}

/// SIGTERM the process group, SIGKILL after `grace` seconds.
pub fn stop_group(pid: i32, child: Arc<Mutex<Child>>, grace: u64, wait: bool) {
    let killer = move || {
        unsafe {
            if libc::killpg(pid, libc::SIGTERM) != 0 {
                return;
            }
        }
        let start = std::time::Instant::now();
        loop {
            if let Ok(mut c) = child.lock() {
                if let Ok(Some(_)) = c.try_wait() {
                    return;
                }
            }
            if start.elapsed() > Duration::from_secs(grace) {
                unsafe {
                    libc::killpg(pid, libc::SIGKILL);
                }
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    if wait {
        killer();
    } else {
        std::thread::spawn(killer);
    }
}

pub fn term_group(pid: i32) {
    unsafe {
        libc::killpg(pid, libc::SIGTERM);
    }
}

// ------------------------------------------------------------------ chime --
fn chime_file() -> std::path::PathBuf {
    dirs().data.join("chime.wav")
}

/// Write a soft two-note bell chime (E5 -> A5) as a 16-bit mono WAV.
fn make_chime(path: &std::path::Path) -> std::io::Result<()> {
    let rate = 44100usize;
    let mut samples: Vec<f64> = Vec::new();
    for (freq, start, dur) in [(659.25f64, 0.0f64, 0.9f64), (880.0, 0.16, 1.1)] {
        let n0 = (start * rate as f64) as usize;
        let n = (dur * rate as f64) as usize;
        if samples.len() < n0 + n {
            samples.resize(n0 + n, 0.0);
        }
        for i in 0..n {
            let t = i as f64 / rate as f64;
            let env = (t / 0.006).min(1.0) * (-t * 4.5).exp();
            let tau = 2.0 * std::f64::consts::PI;
            let v = ((tau * freq * t).sin() + 0.25 * (tau * freq * 2.0 * t).sin() + 0.08 * (tau * freq * 3.0 * t).sin()) * env;
            samples[n0 + i] += v;
        }
    }
    let peak = samples.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let peak = if peak == 0.0 { 1.0 } else { peak };
    let vol = 0.22; // polite: well below full scale
    let data: Vec<u8> = samples.iter().flat_map(|x| ((x / peak * vol * 32767.0) as i16).to_le_bytes()).collect();
    let mut w = Vec::with_capacity(44 + data.len());
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&(rate as u32).to_le_bytes());
    w.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(data.len() as u32).to_le_bytes());
    w.extend_from_slice(&data);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, w)
}

pub fn play_chime() {
    let f = chime_file();
    if !f.exists() && make_chime(&f).is_err() {
        return;
    }
    for player in ["paplay", "pw-play", "aplay"] {
        if let Some(exe) = which(player) {
            let mut c = Command::new(exe);
            if player == "aplay" {
                c.arg("-q");
            }
            c.arg(&f).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
            if let Ok(mut child) = c.spawn() {
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return;
            }
        }
    }
}
