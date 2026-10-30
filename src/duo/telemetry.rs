//! GPU sensors for the performance dashboard (amdgpu sysfs + hwmon), sampled every tick.

use std::path::{Path, PathBuf};

use super::devices::Device;

/// Samples kept per GPU: 30 minutes at the 1.5 s tick.
pub const KEEP_SAMPLES: usize = 1200;
/// Live per-request records kept per pane.
pub const KEEP_REQUESTS: usize = 500;

pub fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Default)]
pub struct GpuSample {
    pub t: f64,
    /// VRAM in bytes
    pub used: Option<f64>,
    pub total: Option<f64>,
    /// GTT (system RAM mapped for the GPU) in bytes
    pub gtt: Option<f64>,
    /// GPU busy in percent
    pub busy: Option<f64>,
    /// °C
    pub edge: Option<f64>,
    pub junction: Option<f64>,
    pub mem: Option<f64>,
    /// W
    pub power: Option<f64>,
    /// MHz
    pub sclk: Option<f64>,
    pub mclk: Option<f64>,
    /// RPM
    pub fan: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct GpuLimits {
    pub edge: Option<f64>,
    pub junction: Option<f64>,
    pub mem: Option<f64>,
    pub power_cap: Option<f64>,
}

fn rd(p: &Path, name: &str) -> Option<f64> {
    std::fs::read_to_string(p.join(name)).ok()?.trim().parse::<f64>().ok()
}
fn rds(p: &Path, name: &str) -> Option<String> {
    std::fs::read_to_string(p.join(name)).ok().map(|s| s.trim().to_lowercase())
}

/// The card's hwmon folder (…/device/hwmon/hwmonN).
fn hwmon(sysfs: &Path) -> Option<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(sysfs.join("hwmon")).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    v.sort();
    v.into_iter().next()
}

/// Read every sensor the card exposes. Missing ones stay None (other vendors, older kernels).
pub fn read_sensors(dev: &Device) -> (GpuSample, GpuLimits) {
    let mut s = GpuSample { t: unix_now(), total: Some(dev.total_mib as f64 * super::util::MIB), ..Default::default() };
    let mut lim = GpuLimits::default();
    let Some(sys) = dev.sysfs.as_ref().map(PathBuf::from) else { return (s, lim) };
    s.used = rd(&sys, "mem_info_vram_used");
    s.total = rd(&sys, "mem_info_vram_total").or(s.total);
    s.gtt = rd(&sys, "mem_info_gtt_used");
    s.busy = rd(&sys, "gpu_busy_percent");
    let Some(h) = hwmon(&sys) else { return (s, lim) };
    for i in 1..=4 {
        let Some(v) = rd(&h, &format!("temp{i}_input")) else { continue };
        let label = rds(&h, &format!("temp{i}_label")).unwrap_or_else(|| match i {
            1 => "edge".into(),
            2 => "junction".into(),
            _ => "mem".into(),
        });
        let crit = rd(&h, &format!("temp{i}_crit")).map(|c| c / 1000.0).filter(|c| *c > 0.0 && *c < 200.0);
        let v = v / 1000.0;
        match label.as_str() {
            "edge" => (s.edge, lim.edge) = (Some(v), crit),
            "junction" | "hotspot" => (s.junction, lim.junction) = (Some(v), crit),
            "mem" | "memory" => (s.mem, lim.mem) = (Some(v), crit),
            _ => {}
        }
    }
    s.power = rd(&h, "power1_average").or_else(|| rd(&h, "power1_input")).map(|u| u / 1e6);
    lim.power_cap = rd(&h, "power1_cap").map(|u| u / 1e6).filter(|w| *w > 0.0);
    for i in 1..=2 {
        let Some(v) = rd(&h, &format!("freq{i}_input")) else { continue };
        let label = rds(&h, &format!("freq{i}_label")).unwrap_or_else(|| if i == 1 { "sclk".into() } else { "mclk".into() });
        match label.as_str() {
            "sclk" => s.sclk = Some(v / 1e6),
            "mclk" => s.mclk = Some(v / 1e6),
            _ => {}
        }
    }
    s.fan = rd(&h, "fan1_input");
    (s, lim)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_fake_card() {
        let d = std::env::temp_dir().join(format!("llmduo-sens-{}", std::process::id()));
        let h = d.join("hwmon/hwmon3");
        std::fs::create_dir_all(&h).unwrap();
        let w = |p: &Path, n: &str, v: &str| std::fs::write(p.join(n), v).unwrap();
        w(&d, "mem_info_vram_used", "1073741824");
        w(&d, "mem_info_vram_total", "17179869184");
        w(&d, "gpu_busy_percent", "57");
        w(&h, "temp1_input", "61000");
        w(&h, "temp1_label", "edge");
        w(&h, "temp2_input", "78000");
        w(&h, "temp2_label", "junction");
        w(&h, "temp2_crit", "110000");
        w(&h, "power1_average", "212000000");
        w(&h, "power1_cap", "304000000");
        w(&h, "freq1_input", "2950000000");
        w(&h, "freq1_label", "sclk");
        let dev = Device {
            backend: "ROCm".into(),
            index: 0,
            id: "ROCm0".into(),
            name: "X".into(),
            total_mib: 16384,
            sysfs: Some(d.to_string_lossy().into_owned()),
        };
        let (s, l) = read_sensors(&dev);
        assert_eq!(s.busy, Some(57.0));
        assert_eq!(s.edge, Some(61.0));
        assert_eq!(s.junction, Some(78.0));
        assert_eq!(l.junction, Some(110.0));
        assert_eq!(s.power, Some(212.0));
        assert_eq!(l.power_cap, Some(304.0));
        assert_eq!(s.sclk, Some(2950.0));
        assert_eq!(s.mem, None);
        std::fs::remove_dir_all(&d).ok();
    }
}
