//! GPU discovery (llama-server --list-devices) and live sysfs statistics.

use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use super::util::{run_capture, tail_chars, RunErr, MIB};

#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub backend: String,
    pub index: i64,
    pub id: String,
    pub name: String,
    pub total_mib: i64,
    pub sysfs: Option<String>,
}

impl Device {
    pub fn short_name(&self) -> String {
        self.name.replace("AMD Radeon ", "")
    }
}

fn dev_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)").unwrap())
}

/// Ask llama-server for its devices, then match each to a sysfs card for live stats.
pub fn detect_devices(server_bin: &str) -> (Vec<Device>, String) {
    let mut devices = Vec::new();
    let mut err = String::new();
    match run_capture(&[server_bin.to_string(), "--list-devices".into()], Duration::from_secs(90), None) {
        Ok(out) => {
            let text = format!("{}\n{}", out.stdout, out.stderr);
            for line in text.lines() {
                if let Some(m) = dev_re().captures(line) {
                    devices.push(Device {
                        backend: m[1].to_string(),
                        index: m[2].parse().unwrap_or(0),
                        id: format!("{}{}", &m[1], &m[2]),
                        name: m[3].trim().to_string(),
                        total_mib: m[4].parse().unwrap_or(0),
                        sysfs: None,
                    });
                }
            }
            if devices.is_empty() {
                err = format!("llama-server --list-devices returned no GPUs.\n\n{}", tail_chars(text.trim(), 3000));
            }
        }
        Err(RunErr::NotFound) => err = format!("llama-server not found at:\n{server_bin}"),
        Err(e) => err = format!("Failed to run llama-server --list-devices:\n{e}"),
    }
    (attach_sysfs(devices), err)
}

/// Match each device to a /sys/class/drm card (by VRAM size) for live stats.
pub fn attach_sysfs(mut devices: Vec<Device>) -> Vec<Device> {
    let mut cards: Vec<(PathBuf, f64)> = Vec::new();
    let card_re = Regex::new(r"^card\d+$").unwrap();
    // LLM_DUO_DRM_DIR: a stand-in for /sys/class/drm (testing without the hardware)
    let drm = std::env::var("LLM_DUO_DRM_DIR").unwrap_or_else(|_| "/sys/class/drm".into());
    if let Ok(rd) = std::fs::read_dir(&drm) {
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        for d in entries {
            let name = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if !card_re.is_match(&name) {
                continue;
            }
            let t = d.join("device").join("mem_info_vram_total");
            if let Some(v) = std::fs::read_to_string(&t).ok().and_then(|s| s.trim().parse::<f64>().ok()) {
                cards.push((d.join("device"), v));
            }
        }
    }
    let mut used = vec![false; cards.len()];
    let mut idx: Vec<usize> = (0..devices.len()).collect();
    idx.sort_by_key(|&i| -devices[i].total_mib);
    for di in idx {
        let mut best: Option<(usize, f64)> = None;
        for (i, c) in cards.iter().enumerate() {
            if used[i] {
                continue;
            }
            let diff = (c.1 / MIB - devices[di].total_mib as f64).abs();
            if best.map(|b| diff < b.1).unwrap_or(true) {
                best = Some((i, diff));
            }
        }
        if let Some((i, _)) = best {
            used[i] = true;
            devices[di].sysfs = Some(cards[i].0.to_string_lossy().into_owned());
        }
    }
    devices
}

fn rd(p: &Path, name: &str) -> Option<f64> {
    std::fs::read_to_string(p.join(name)).ok()?.trim().parse::<f64>().ok()
}

/// (used_bytes, total_bytes, busy_percent) from sysfs.
pub fn gpu_stats(dev: Option<&Device>) -> (Option<f64>, Option<f64>, Option<f64>) {
    let Some(d) = dev else { return (None, None, None) };
    let Some(s) = &d.sysfs else { return (None, Some(d.total_mib as f64 * MIB), None) };
    let p = Path::new(s);
    (rd(p, "mem_info_vram_used"), rd(p, "mem_info_vram_total"), rd(p, "gpu_busy_percent"))
}

/// GTT (system RAM mapped for the GPU) in use on this card, in bytes.
pub fn gpu_gtt(dev: Option<&Device>) -> Option<f64> {
    let s = dev?.sysfs.as_ref()?;
    rd(Path::new(s), "mem_info_gtt_used")
}

pub fn device_env(dev: &Device) -> Vec<(String, String)> {
    let idx = dev.index.to_string();
    match dev.backend.to_lowercase().as_str() {
        "rocm" | "hip" => vec![("HIP_VISIBLE_DEVICES".into(), idx)],
        "vulkan" => vec![("GGML_VK_VISIBLE_DEVICES".into(), idx)],
        "cuda" => vec![("CUDA_VISIBLE_DEVICES".into(), idx)],
        _ => vec![],
    }
}

pub fn dev_label(d: &Device, decimals: usize) -> String {
    format!("{}: {} ({:.*} GiB)", d.id, d.name, decimals, d.total_mib as f64 / 1024.0)
}
