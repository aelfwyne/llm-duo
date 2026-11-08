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

/// DRM render minors of the GPUs HIP can see, in HIP device order (HIP numbers its devices in
/// KFD topology order, which is not the /sys/class/drm cardN order). None if unknown.
fn hip_render_minors(kfd: &Path) -> Option<Vec<i64>> {
    let mut nodes: Vec<(u64, i64)> = vec![];
    for e in std::fs::read_dir(kfd.join("nodes")).ok()?.filter_map(|e| e.ok()) {
        let Ok(id) = e.file_name().to_string_lossy().parse::<u64>() else { continue };
        let Ok(props) = std::fs::read_to_string(e.path().join("properties")) else { continue };
        let prop = |k: &str| {
            props.lines().find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix(' ')).and_then(|v| v.trim().parse::<i64>().ok()))
        };
        // CPU nodes have no SIMDs
        if prop("simd_count").unwrap_or(0) > 0 {
            nodes.push((id, prop("drm_render_minor")?));
        }
    }
    nodes.sort();
    let mut minors: Vec<i64> = nodes.into_iter().map(|n| n.1).collect();
    // ROCR_VISIBLE_DEVICES filters the agents, then HIP_VISIBLE_DEVICES indexes what is left.
    // Only plain index lists are understood; UUIDs etc. fall back to matching by VRAM size.
    for var in ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"] {
        let Ok(v) = std::env::var(var) else { continue };
        if v.trim().is_empty() {
            continue;
        }
        let idx: Vec<usize> = v.split(',').map(|s| s.trim().parse::<usize>().ok()).collect::<Option<_>>()?;
        minors = idx.iter().map(|&i| minors.get(i).copied()).collect::<Option<_>>()?;
    }
    Some(minors)
}

/// Match each device to a /sys/class/drm card for live stats: ROCm devices through the KFD
/// topology (exact, also for identical cards), the rest by VRAM size.
pub fn attach_sysfs(devices: Vec<Device>) -> Vec<Device> {
    // LLM_DUO_DRM_DIR / LLM_DUO_KFD_DIR: stand-ins for /sys/class/drm and /sys/class/kfd/kfd
    // (testing without the hardware)
    let drm = std::env::var("LLM_DUO_DRM_DIR").unwrap_or_else(|_| "/sys/class/drm".into());
    let kfd = std::env::var("LLM_DUO_KFD_DIR").unwrap_or_else(|_| "/sys/class/kfd/kfd".into());
    attach_sysfs_in(devices, Path::new(&drm), Path::new(&kfd))
}

fn attach_sysfs_in(mut devices: Vec<Device>, drm: &Path, kfd: &Path) -> Vec<Device> {
    let mut cards: Vec<(PathBuf, f64)> = Vec::new();
    let card_re = Regex::new(r"^card\d+$").unwrap();
    if let Ok(rd) = std::fs::read_dir(drm) {
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
    let canon: Vec<Option<PathBuf>> = cards.iter().map(|c| std::fs::canonicalize(&c.0).ok()).collect();
    if let Some(minors) = hip_render_minors(kfd) {
        for d in devices.iter_mut() {
            if !matches!(d.backend.to_lowercase().as_str(), "rocm" | "hip") || d.index < 0 {
                continue;
            }
            let Some(minor) = minors.get(d.index as usize) else { continue };
            let Ok(p) = std::fs::canonicalize(drm.join(format!("renderD{minor}")).join("device")) else { continue };
            if let Some(i) = (0..cards.len()).find(|&i| !used[i] && canon[i].as_ref() == Some(&p)) {
                used[i] = true;
                d.sysfs = Some(cards[i].0.to_string_lossy().into_owned());
            }
        }
    }
    let mut idx: Vec<usize> = (0..devices.len()).filter(|&i| devices[i].sysfs.is_none()).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(i: i64) -> Device {
        Device {
            backend: "ROCm".into(),
            index: i,
            id: format!("ROCm{i}"),
            name: "AMD Radeon AI PRO R9700".into(),
            total_mib: 32624,
            sysfs: None,
        }
    }

    #[test]
    fn identical_cards_follow_hip_order_not_card_numbers() {
        // card0 = PCI 06:00.0 (renderD129), card1 = PCI 03:00.0 (renderD128); HIP device 0 is
        // the first KFD GPU node, renderD128 -> card1
        let root = std::env::temp_dir().join(format!("llmduo-devs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (drm, kfd, pci) = (root.join("drm"), root.join("kfd"), root.join("pci"));
        for (card, bdf, minor) in [("card0", "0000:06:00.0", 129), ("card1", "0000:03:00.0", 128)] {
            let d = pci.join(bdf);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("mem_info_vram_total"), "34208743424").unwrap();
            std::fs::create_dir_all(drm.join(card)).unwrap();
            std::os::unix::fs::symlink(&d, drm.join(card).join("device")).unwrap();
            std::fs::create_dir_all(drm.join(format!("renderD{minor}"))).unwrap();
            std::os::unix::fs::symlink(&d, drm.join(format!("renderD{minor}")).join("device")).unwrap();
        }
        for (n, simd, minor) in [(0, 0, 0), (1, 128, 128), (2, 128, 129)] {
            let d = kfd.join("nodes").join(n.to_string());
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("properties"), format!("cpu_cores_count 0\nsimd_count {simd}\ndrm_render_minor {minor}\n")).unwrap();
        }
        let out = attach_sysfs_in(vec![dev(0), dev(1)], &drm, &kfd);
        let end = |d: &Device| PathBuf::from(d.sysfs.clone().unwrap()).parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(end(&out[0]), "card1");
        assert_eq!(end(&out[1]), "card0");
        // without KFD topology: falls back to VRAM size (both cards still get one)
        let out = attach_sysfs_in(vec![dev(0), dev(1)], &drm, &root.join("none"));
        assert!(out.iter().all(|d| d.sysfs.is_some()));
        let _ = std::fs::remove_dir_all(&root);
    }
}
