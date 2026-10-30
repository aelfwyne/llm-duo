//! KoboldCpp helpers, Vulkan device mapping and llama-server log regexes.

use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use super::consts::KOBOLD_BATCHES;
use super::devices::Device;
use super::util::{run_capture, shlex_quote, tail_chars, RunErr};

pub struct KRe {
    pub ready: Regex,
    pub done: Regex,
    pub progress: Regex,
    pub draft: Regex,
    pub flag: Regex,
    pub ansi: Regex,
}

pub fn kre() -> &'static KRe {
    static R: OnceLock<KRe> = OnceLock::new();
    R.get_or_init(|| KRe {
        ready: Regex::new(r"Please connect to custom endpoint at|Your remote tunnel is ready").unwrap(),
        done: Regex::new(
            r"CtxLimit:(\d+)/(\d+),.*?Processed:(\d+) in ([\d.]+)s \(([\d.]+)T/s\),\s*Generated:(\d+)/(\d+) in ([\d.]+)s \(([\d.]+)T/s\)",
        )
        .unwrap(),
        progress: Regex::new(
            r"^\s*(?:Processing Prompt(?: \[\w+\])? \((\d+) / (\d+) tokens\)|Generating \((\d+) / (\d+) tokens\))\s*$",
        )
        .unwrap(),
        draft: Regex::new(r"Draft Results - Success:(\d+), Failure:(\d+)").unwrap(),
        // Python: (?<![\w-])--[A-Za-z][\w-]*   (lookbehind done by hand in help_flags)
        flag: Regex::new(r"--[A-Za-z][\w-]*").unwrap(),
        ansi: Regex::new(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*\x07|[()][A-Za-z0-9])").unwrap(),
    })
}

pub fn extract_flags(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for m in kre().flag.find_iter(text) {
        let before = text[..m.start()].chars().last();
        if let Some(c) = before {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                continue;
            }
        }
        out.insert(m.as_str().to_string());
    }
    out
}

/// Run '<koboldcpp> --help' (a PyInstaller one-file binary unpacks first: slow) -> (flags, error).
pub fn kobold_help_flags(exe: &str, timeout: u64) -> Result<Vec<String>, String> {
    let out = match run_capture(&[exe.to_string(), "--help".into()], Duration::from_secs(timeout), None) {
        Ok(o) => o,
        Err(RunErr::PermissionDenied) => return Err(format!("{exe} is not executable - run:  chmod +x {}", shlex_quote(exe))),
        Err(RunErr::NotFound) => return Err(format!("not found: {exe}")),
        Err(RunErr::Timeout) => return Err(format!("{exe} --help took over {timeout} s")),
        Err(e) => return Err(format!("could not run {exe} --help: {e}")),
    };
    let all = format!("{}\n{}", out.stdout, out.stderr);
    let flags = extract_flags(&all);
    if !flags.contains("--contextsize") {
        return Err(format!(
            "{exe} --help doesn't look like KoboldCpp:\n{}",
            tail_chars(&format!("{}{}", out.stdout, out.stderr), 1500)
        ));
    }
    Ok(flags.into_iter().collect())
}

/// Largest batch size KoboldCpp accepts that is <= n.
pub fn snap_batch(n: i64) -> i64 {
    KOBOLD_BATCHES.iter().rev().find(|&&b| b <= n).cloned().unwrap_or(KOBOLD_BATCHES[0])
}

/// (domain, bus, device, function) of a detected GPU from its sysfs path.
pub fn pci_of(dev: &Device) -> Option<(u32, u32, u32, u32)> {
    let p = std::fs::canonicalize(dev.sysfs.as_ref()?).ok()?;
    let name = p.file_name()?.to_string_lossy().into_owned();
    let re = Regex::new(r"(?i)^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])$").unwrap();
    let m = re.captures(&name)?;
    let h = |i: usize| u32::from_str_radix(&m[i], 16).ok();
    Some((h(1)?, h(2)?, h(3)?, h(4)?))
}

#[derive(Clone, Debug)]
pub struct VkGpu {
    pub index: i64,
    pub name: String,
    pub discrete: bool,
    pub pci: Option<(u32, u32, u32, u32)>,
}

/// Vulkan physical devices in enumeration order (the index ggml/KoboldCpp uses).
pub fn vulkan_gpus() -> (Vec<VkGpu>, String) {
    let mut text = String::new();
    let mut err = String::new();
    let gpu_hdr = Regex::new(r"(?m)^GPU\d+:").unwrap();
    for args in [vec!["vulkaninfo".to_string()], vec!["vulkaninfo".into(), "--summary".into()]] {
        match run_capture(&args, Duration::from_secs(30), None) {
            Ok(o) => {
                text = o.stdout;
                if gpu_hdr.is_match(&text) {
                    break;
                }
            }
            Err(RunErr::NotFound) => {
                return (vec![], "vulkaninfo not found - install it with:  sudo apt install vulkan-tools".into())
            }
            Err(e) => err = format!("vulkaninfo failed: {e}"),
        }
    }
    let split = Regex::new(r"(?m)^GPU(\d+):\s*$").unwrap();
    let mut gpus: HashMap<i64, VkGpu> = HashMap::new();
    let heads: Vec<(i64, usize, usize)> =
        split.captures_iter(&text).map(|c| (c[1].parse().unwrap_or(0), c.get(0).unwrap().start(), c.get(0).unwrap().end())).collect();
    let name_re = Regex::new(r"deviceName\s*=\s*(.+)").unwrap();
    let pci_res: Vec<Regex> =
        ["Domain", "Bus", "Device", "Function"].iter().map(|k| Regex::new(&format!(r"\bpci{k}\s*=\s*(\d+)")).unwrap()).collect();
    for (i, (idx, _s, e)) in heads.iter().enumerate() {
        let end = heads.get(i + 1).map(|h| h.1).unwrap_or(text.len());
        let body = &text[*e..end];
        let g = gpus.entry(*idx).or_insert(VkGpu { index: *idx, name: String::new(), discrete: false, pci: None });
        if let Some(m) = name_re.captures(body) {
            if g.name.is_empty() {
                g.name = m[1].trim().to_string();
            }
        }
        if body.contains("PHYSICAL_DEVICE_TYPE_DISCRETE_GPU") {
            g.discrete = true;
        }
        let pc: Vec<Option<u32>> = pci_res.iter().map(|r| r.captures(body).and_then(|c| c[1].parse().ok())).collect();
        if pc.iter().all(|x| x.is_some()) {
            g.pci = Some((pc[0].unwrap(), pc[1].unwrap(), pc[2].unwrap(), pc[3].unwrap()));
        }
    }
    if gpus.is_empty() && err.is_empty() {
        err = "vulkaninfo listed no GPUs".into();
    }
    let mut keys: Vec<i64> = gpus.keys().cloned().collect();
    keys.sort();
    (keys.into_iter().map(|k| gpus.remove(&k).unwrap()).collect(), err)
}

fn norm_gpu_name(n: &str) -> String {
    let re = Regex::new(r"\s*\(.*?\)\s*").unwrap();
    re.replace_all(n, " ").trim().to_lowercase()
}

/// Match each detected GPU to its Vulkan index: PCI address first, then name, then order.
pub fn map_vulkan(devices: &[Device], vk: &[VkGpu]) -> (HashMap<String, i64>, HashMap<String, String>) {
    let mut out = HashMap::new();
    let mut how = HashMap::new();
    let mut taken: BTreeSet<i64> = BTreeSet::new();
    for d in devices {
        let pci = pci_of(d);
        if let Some(g) = vk.iter().find(|g| pci.is_some() && g.pci == pci && !taken.contains(&g.index)) {
            out.insert(d.id.clone(), g.index);
            how.insert(d.id.clone(), "PCI address".to_string());
            taken.insert(g.index);
        }
    }
    for d in devices {
        if out.contains_key(&d.id) {
            continue;
        }
        let nm = norm_gpu_name(&d.name);
        let hit: Vec<&VkGpu> = vk
            .iter()
            .filter(|g| {
                let gn = norm_gpu_name(&g.name);
                !taken.contains(&g.index) && !nm.is_empty() && (gn.starts_with(&nm) || nm.starts_with(&gn))
            })
            .collect();
        if hit.len() == 1 {
            out.insert(d.id.clone(), hit[0].index);
            how.insert(d.id.clone(), "name".to_string());
            taken.insert(hit[0].index);
        }
    }
    let mut rest: Vec<&VkGpu> = vk.iter().filter(|g| g.discrete && !taken.contains(&g.index)).collect();
    let mut left: Vec<&Device> = devices.iter().filter(|d| !out.contains_key(&d.id)).collect();
    left.sort_by_key(|d| d.index);
    for d in left {
        if !rest.is_empty() {
            let g = rest.remove(0);
            out.insert(d.id.clone(), g.index);
            how.insert(d.id.clone(), "order (guess)".to_string());
        }
    }
    (out, how)
}

pub fn kobold_failure_hint(lines: &[String]) -> String {
    let text = lines.join("\n");
    let m = |p: &str| Regex::new(p).unwrap().is_match(&text);
    if m("Permission denied") {
        return "the KoboldCpp binary isn't executable - run chmod +x on it.".into();
    }
    if m(r"(?i)address already in use|Errno 98|port .* in use") {
        return "the port is already in use (another server, or a KoboldCpp that didn't exit).".into();
    }
    if m(r"(?i)hipErrorNoBinaryForGpu|no kernel image|invalid device function|Cannot find Symbol|not compiled for") {
        return "this ROCm build has no GPU code for your cards (gfx1201) - re-download the rocm-rolling build, or use the Vulkan build.".into();
    }
    if m(r"(?i)out of memory|ErrorOutOfDeviceMemory|failed to allocate|cudaMalloc failed|hipMalloc failed|unable to allocate") {
        return "out of VRAM - lower the context, GPU layers or KV cache type, or press Fit.".into();
    }
    if m(r"(?i)unknown model architecture|unknown architecture") {
        return "this KoboldCpp build doesn't know the model's architecture yet - update KoboldCpp.".into();
    }
    if text.contains("Could not load text model") {
        return "KoboldCpp couldn't load the model - the lines above it in the log say why.".into();
    }
    if m(r"error: unrecognized arguments|invalid choice") {
        return "this KoboldCpp build doesn't accept one of the arguments (see the log) - check Extra args, or update KoboldCpp.".into();
    }
    String::new()
}

// ------------------------------------------------------ llama-server log --
pub struct LRe {
    pub prompt: Regex,
    pub req_done: Regex,
    pub eot_missing: Regex,
    pub accept: Regex,
    pub eval: Regex,
    pub buf: Regex,
    pub ready: Regex,
    pub metric: Regex,
    pub busy: Regex,
    pub idle_one: Regex,
    pub idle_all: Regex,
}

pub fn lre() -> &'static LRe {
    static R: OnceLock<LRe> = OnceLock::new();
    R.get_or_init(|| LRe {
        prompt: Regex::new(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second").unwrap(),
        req_done: Regex::new(r"release:.*stop processing(?:: n_tokens = (\d+))?").unwrap(),
        eot_missing: Regex::new(r"special_eot_id is not in special_eog_ids").unwrap(),
        accept: Regex::new(r"draft acceptance(?: rate)?\s*=\s*([\d.]+)(?:\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated)?").unwrap(),
        // Python: (?<!prompt )\beval time = …   (lookbehind checked in eval_match)
        eval: Regex::new(r"\beval time =\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second").unwrap(),
        buf: Regex::new(r"\b(?:ROCm|Vulkan|CUDA)\d+\S*\s+(model|KV|compute|RS)\s+buffer size\s*=\s*([\d.]+) MiB").unwrap(),
        ready: Regex::new(r"listening on http|server is listening on|model loaded").unwrap(),
        metric: Regex::new(r"(?m)^llamacpp:(\w+)\s+([-\d.eE+]+)").unwrap(),
        busy: Regex::new(r"launch_slot_.*processing task|processing task").unwrap(),
        idle_one: Regex::new(r"stop processing").unwrap(),
        idle_all: Regex::new(r"all slots are idle").unwrap(),
    })
}

/// RE_EVAL.search with the `(?<!prompt )` lookbehind: (ms, tokens, tps).
pub fn eval_match(line: &str) -> Option<(f64, i64, f64)> {
    let re = &lre().eval;
    let mut start = 0;
    while start <= line.len() {
        let c = re.captures_at(line, start)?;
        let m = c.get(0).unwrap();
        if !line[..m.start()].ends_with("prompt ") {
            return Some((c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?));
        }
        start = m.start() + 1;
        while start < line.len() && !line.is_char_boundary(start) {
            start += 1;
        }
    }
    None
}

pub fn path_exists(p: &str) -> bool {
    !p.is_empty() && Path::new(p).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eval_lookbehind() {
        assert!(eval_match("prompt eval time =     100.00 ms /    10 tokens (  10.00 ms per token,   100.00 tokens per second)").is_none());
        let m = eval_match("       eval time =    2000.00 ms /   100 tokens (   20.00 ms per token,    50.00 tokens per second)").unwrap();
        assert_eq!(m.1, 100);
        assert_eq!(m.2, 50.0);
        assert_eq!(snap_batch(700), 512);
        assert_eq!(snap_batch(5), 16);
        let f = extract_flags("usage: --model M [--contextsize N] a--b --usecuda\n  --quantkv");
        assert!(f.contains("--contextsize") && f.contains("--usecuda") && !f.contains("--b"));
    }
    #[test]
    fn kobold_done() {
        let l = "[12:00:00] CtxLimit:1234/8192, Amt:100/512, Init:0.01s, Process:1.00s (1000.0T/s), Generate:2.00s (50.0T/s), Total:3.00s";
        assert!(kre().done.captures(l).is_none());
        let l = "CtxLimit:1234/8192, Amt:100/512, Process:1.00s, Processed:1134 in 1.00s (1134.00T/s), Generated:100/512 in 2.00s (50.00T/s)";
        let c = kre().done.captures(l).unwrap();
        assert_eq!(&c[3], "1134");
        assert!(kre().progress.is_match("Processing Prompt [BLAS] (512 / 1134 tokens)"));
        assert!(kre().progress.is_match("Generating (12 / 512 tokens)"));
    }
}
