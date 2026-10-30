//! llama.cpp / KoboldCpp VRAM estimate and the dual-GPU layer-split planner.

use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use super::consts::cache_bytes;
use super::devices::Device;
use super::gguf::ModelInfo;
use super::util::{gb, gf, gi, gs, py_round, Opts, GIB, MIB};

#[derive(Clone, Debug, Default)]
pub struct Est {
    pub weights: f64,
    pub kv: f64,
    pub state: f64,
    pub compute: f64,
    pub mmproj: f64,
    pub overhead: f64,
    pub total: f64,
}

/// Very rough VRAM estimate in bytes.
/// `layer_range`/`with_output` describe one device's share of a multi-GPU layer split.
/// `cpu_set`: explicit set of layers whose experts stay in RAM (else the first `n_cpu_moe`).
#[allow(clippy::too_many_arguments)]
pub fn estimate_vram(
    info: &ModelInfo,
    s: &Opts,
    mmproj_bytes: f64,
    overhead_mib: f64,
    adjust: f64,
    layer_range: Option<(i64, i64)>,
    with_output: Option<bool>,
    cpu_set: Option<&BTreeSet<usize>>,
) -> Est {
    // MTP / next-token-prediction layers at the end are not loaded by llama.cpp
    let n = info.n_layer as i64 - info.nextn as i64;
    let n = n.max(0);
    let split = layer_range.is_some();
    let gpu_layers: Vec<usize>;
    let mut ngl = 0i64;
    if let Some((a, b)) = layer_range {
        let lo = a.max(0);
        let hi = b.min(n);
        gpu_layers = if hi > lo { (lo as usize..hi as usize).collect() } else { vec![] };
    } else {
        let s_ngl = gi(s, "ngl", 999);
        ngl = if s_ngl < 0 { n + 1 } else { s_ngl.min(n + 1).max(0) };
        let lo = n - ngl.min(n);
        gpu_layers = (lo as usize..n as usize).collect(); // llama.cpp offloads the last layers
    }
    let ncm = gi(s, "n_cpu_moe", 0);

    let mut weights = 0.0;
    let mut cpu_expert_layers = Vec::new();
    for &i in &gpu_layers {
        weights += info.layer_bytes[i];
        let on_cpu = match cpu_set {
            Some(set) => set.contains(&i),
            None => (i as i64) < ncm,
        };
        if on_cpu {
            weights -= info.expert_bytes[i];
            if info.expert_bytes[i] > 0.0 {
                cpu_expert_layers.push(i);
            }
        }
    }
    let out_on_gpu = if split { with_output.unwrap_or(false) } else { ngl > n };
    if out_on_gpu {
        weights += info.output_bytes;
    }

    let kvt = gs(s, "kv", "f16");
    let bk = cache_bytes(&kvt);
    let bv = bk;
    let ub = gi(s, "ubatch", 512).max(1) as f64;
    let ctx = gi(s, "ctx", 8192) as f64;
    let parallel = gi(s, "parallel", 1).max(1) as f64;
    let mut swa_cells = ctx;
    if info.swa != 0.0 && !gs(s, "extra", "").contains("--swa-full") {
        // llama.cpp keeps only window * slots + ubatch cells (padded to 256) for SWA layers
        let need = info.swa * parallel + ub;
        swa_cells = ctx.min((need / 256.0).ceil() * 256.0);
    }
    let mut kv = 0.0;
    for &i in &gpu_layers {
        let h = info.kv_heads[i];
        if h == 0.0 {
            continue;
        }
        if info.swa_layers[i] {
            kv += h * (info.k_len_swa * bk + info.v_len_swa * bv) * swa_cells;
        } else {
            kv += h * (info.k_len * bk + info.v_len * bv) * ctx;
        }
    }
    // recurrent state (f32, one per parallel sequence)
    let state: f64 = gpu_layers.iter().filter(|&&i| info.rs_layers[i]).map(|_| info.rs_bytes_per_layer).sum::<f64>() * parallel;

    let mut compute = ub * info.n_embd * 4.0 * 12.0;
    // output logits are computed on the GPU even when the output layer lives in RAM
    if out_on_gpu || !split {
        compute += info.n_vocab * ub * 4.0;
    }
    if !out_on_gpu && !split {
        // partial offload: prompt processing streams CPU-resident layers' weights through the GPU
        compute += (0..n as usize).map(|i| info.layer_bytes[i]).fold(0.0, f64::max);
    }
    if !cpu_expert_layers.is_empty() {
        compute += 0.55 * cpu_expert_layers.iter().map(|&i| info.expert_bytes[i]).fold(0.0, f64::max);
    }
    // attention masks (f16, ctx x ubatch); a second one for SWA layers
    compute += ctx * ub * 2.0 * if info.swa != 0.0 { 2.0 } else { 1.0 };
    if gs(s, "fa", "on") == "off" {
        compute += info.n_head * ctx * ub * 4.0;
    } else if kvt != "f16" {
        // flash attention converts one layer's quantized K/V to f16 scratch
        let mut worst: f64 = 0.0;
        for &i in &gpu_layers {
            let h = info.kv_heads[i];
            if h == 0.0 {
                continue;
            }
            if info.swa_layers[i] {
                worst = worst.max(h * (info.k_len_swa + info.v_len_swa) * 2.0 * swa_cells);
            } else {
                worst = worst.max(h * (info.k_len + info.v_len) * 2.0 * ctx);
            }
        }
        compute += worst;
    }

    if gs(s, "mtp", "") == "(built-in)" && out_on_gpu {
        let nx = info.nextn;
        weights += info.layer_bytes[info.n_layer.saturating_sub(nx)..].iter().sum::<f64>();
        let (k, c) = mtp_ctx_bytes(info, s);
        kv += k;
        compute += c;
    }
    let overhead = overhead_mib * MIB;
    let total = weights + kv + state + compute + mmproj_bytes + overhead + adjust;
    Est { weights, kv, state, compute, mmproj: mmproj_bytes, overhead, total }
}

/// A built-in MTP head's draft context: (KV cache, compute buffer) bytes.
pub fn mtp_ctx_bytes(info: &ModelInfo, s: &Opts) -> (f64, f64) {
    let bk = cache_bytes(&gs(s, "kv", "f16"));
    let ub = gi(s, "ubatch", 512).max(1) as f64;
    let ctx = gi(s, "ctx", 8192) as f64;
    let h_mtp = info.kv_heads.iter().cloned().fold(0.0, f64::max);
    (info.nextn as f64 * h_mtp * (info.k_len * bk + info.v_len * bk) * ctx, ub * info.n_embd * 4.0 * 4.0)
}

/// Simple single-GPU form.
pub fn est_simple(info: &ModelInfo, s: &Opts, mm: f64, oh: f64, adjust: f64) -> Est {
    estimate_vram(info, s, mm, oh, adjust, None, None, None)
}

/// (gen, width) of the slot link for a GPU, read from sysfs.
///
/// Radeon cards sit behind an on-board PCIe switch whose own link always reports full
/// speed, so the slot's real link is read from the device right below the root port.
pub fn pcie_link(dev: &Device) -> (Option<u32>, Option<u32>) {
    let Some(sys) = &dev.sysfs else { return (None, None) };
    let f = || -> Option<(u32, u32)> {
        let path = std::fs::canonicalize(sys).ok()?;
        let parts: Vec<_> = path.components().collect();
        let idx = parts.iter().position(|c| c.as_os_str().to_string_lossy().starts_with("pci"))?;
        let slot: PathBuf = if parts.len() > idx + 2 { parts[..idx + 3].iter().collect() } else { path.clone() };
        let speed = std::fs::read_to_string(slot.join("current_link_speed")).ok()?;
        let width: u32 = std::fs::read_to_string(slot.join("current_link_width")).ok()?.trim().parse().ok()?;
        let gts: f64 = speed.split_whitespace().next()?.parse().ok()?;
        let generation = match gts {
            x if x == 2.5 => 1,
            x if x == 5.0 => 2,
            x if x == 8.0 => 3,
            x if x == 16.0 => 4,
            x if x == 32.0 => 5,
            x if x == 64.0 => 6,
            _ => return Some((0, width)),
        };
        Some((generation, width))
    };
    match f() {
        Some((0, w)) => (None, Some(w)),
        Some((g, w)) => (Some(g), Some(w)),
        None => (None, None),
    }
}

/// Split modes listed for --split-mode in llama-server --help.
pub fn split_modes_supported(help_text: &str) -> Vec<String> {
    let re = Regex::new(r"--split-mode[^\n]*\n?(?:[^\n]*\n){0,6}").unwrap();
    let chunk = re.find(help_text).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
    let mut modes = Vec::new();
    for x in ["none", "layer", "row", "tensor"] {
        if Regex::new(&format!(r"\b{x}\b")).unwrap().is_match(&chunk) {
            modes.push(x.to_string());
        }
    }
    if modes.is_empty() { vec!["none".into(), "layer".into()] } else { modes }
}

/// Pick `ncm` expert layers to keep in system RAM, split evenly between the two devices of
/// a layer split (first device: layers < k_first). Only layers that have experts.
pub fn spread_cpu_moe(info: &ModelInfo, ncm: i64, k_first: usize) -> BTreeSet<usize> {
    let n_eff = info.n_layer.saturating_sub(info.nextn);
    let a: Vec<usize> = (0..k_first.min(n_eff)).filter(|&i| info.expert_bytes[i] > 0.0).collect();
    let b: Vec<usize> = (k_first..n_eff).filter(|&i| info.expert_bytes[i] > 0.0).collect();
    let ncm = ncm.max(0) as usize;
    let na = a.len().min(ncm.div_ceil(2));
    let nb = b.len().min(ncm.saturating_sub(na));
    let na = a.len().min(ncm.saturating_sub(nb));
    a[..na].iter().chain(b[..nb].iter()).cloned().collect()
}

fn spread_on(s: &Opts, info: &ModelInfo) -> bool {
    gi(s, "n_cpu_moe", 0) > 0 && gb(s, "dual_moe_spread", false) && info.n_expert != 0
}

/// Per-device estimates for a layer split: order[0] gets layers [0,k_first), order[1] the rest
/// + output layer + MTP drafter. mmproj counted on order[0].
pub fn dual_estimate(
    info: &ModelInfo,
    s: &Opts,
    order: &[Device],
    k_first: usize,
    side_bytes: (f64, f64),
    oh: f64,
    adjust: &HashMap<String, f64>,
) -> [Est; 2] {
    let (mm_b, mut mtp_b) = side_bytes;
    // the drafter goes on the second card unless the user put it on the first
    let mut mm_b = mm_b;
    if gs(s, "mtp_dev", "(auto)") == order[0].id {
        mm_b += mtp_b;
        mtp_b = 0.0;
    }
    let set = if spread_on(s, info) { Some(spread_cpu_moe(info, gi(s, "n_cpu_moe", 0), k_first)) } else { None };
    let mut a = estimate_vram(
        info, s, mm_b, oh, adjust.get(&order[0].id).cloned().unwrap_or(0.0),
        Some((0, k_first as i64)), Some(false), set.as_ref(),
    );
    let mut b = estimate_vram(
        info, s, mtp_b, oh, adjust.get(&order[1].id).cloned().unwrap_or(0.0),
        Some((k_first as i64, info.n_layer as i64)), Some(true), set.as_ref(),
    );
    // a built-in MTP head's weights stay with the last layers, but its draft context (its own KV
    // cache and compute buffer) is made on --spec-draft-device
    if gs(s, "mtp", "") == "(built-in)" && gs(s, "mtp_dev", "(auto)") == order[0].id {
        let (kv, comp) = mtp_ctx_bytes(info, s);
        b.kv -= kv;
        b.compute -= comp;
        b.total -= kv + comp;
        a.kv += kv;
        a.compute += comp;
        a.total += kv + comp;
    }
    [a, b]
}

#[derive(Clone, Debug)]
pub struct DualPlan {
    pub strategy: String,
    pub order: Vec<Device>,
    pub mode: String,
    pub k: Option<usize>,
    /// [first, second]; single-GPU plans fill only one slot
    pub est: [Option<Est>; 2],
    pub fits: bool,
    pub note: String,
    pub ts: Option<String>,
    pub single: Option<Device>,
    pub cpu_set: Option<BTreeSet<usize>>,
}

pub struct Manual {
    pub mode: String,
    pub ts: String,
    pub first: String,
}

/// Choose how to span a model over two GPUs (layer split for the Auto strategies).
#[allow(clippy::too_many_arguments)]
pub fn plan_dual(
    info: &ModelInfo,
    s: &Opts,
    devs: &[Device],
    budgets: &HashMap<String, f64>,
    strategy: &str,
    side_bytes: (f64, f64),
    oh: f64,
    manual: Option<&Manual>,
    adjust: &HashMap<String, f64>,
) -> DualPlan {
    let n_eff = info.n_layer.saturating_sub(info.nextn);
    let mut sorted = devs.to_vec();
    sorted.sort_by_key(|d| (d.total_mib, d.index));
    let (mut small, mut big) = (sorted[0].clone(), sorted[1].clone());
    if small.total_mib == big.total_mib {
        let mut by_i = devs.to_vec();
        by_i.sort_by_key(|d| d.index);
        small = by_i[0].clone();
        big = by_i[1].clone();
    }
    // Normally the big card goes last (output layer, logits, MTP). When experts live in system
    // RAM, llama.cpp streams them through the FIRST device during prompt processing, so the
    // card holding more of the model goes first instead.
    let big_first = gi(s, "n_cpu_moe", 0) > 0 && info.n_expert != 0;
    let mut order = if big_first { vec![big.clone(), small.clone()] } else { vec![small.clone(), big.clone()] };
    let mut bud: Vec<f64> = order.iter().map(|d| budgets.get(&d.id).cloned().unwrap_or(0.0)).collect();
    let mut tot: Vec<f64> = order.iter().map(|d| d.total_mib as f64 * MIB).collect();

    let mut plan = DualPlan {
        strategy: strategy.into(),
        order: order.clone(),
        mode: "layer".into(),
        k: None,
        est: [None, None],
        fits: false,
        note: String::new(),
        ts: None,
        single: None,
        cpu_set: None,
    };
    let lay_ts = |k: usize| format!("{},{}", k, info.n_layer as i64 + 1 - k as i64);

    if let (Some(m), "Manual") = (manual, strategy) {
        let mode = if m.mode.is_empty() { "layer".to_string() } else { m.mode.clone() };
        let want_big_first = if !m.first.is_empty() { m.first == big.id } else { big_first };
        if want_big_first != big_first {
            order.reverse();
            bud.reverse();
            tot.reverse();
        }
        plan.order = order.clone();
        plan.mode = mode.clone();
        if mode == "none" {
            let dev = order[0].clone();
            let mut s1 = s.clone();
            s1.insert("ngl".into(), (gi(s, "ngl", 999).max(n_eff as i64 + 1)).into());
            let e1 = estimate_vram(info, &s1, side_bytes.0 + side_bytes.1, oh, 0.0, None, None, None);
            plan.fits = e1.total <= bud[0];
            plan.note = format!("single GPU: {}", dev.name);
            plan.est = [Some(e1), None];
            plan.single = Some(dev);
            return plan;
        }
        let parsed: Option<(f64, f64)> = {
            let parts: Vec<&str> = m.ts.split(',').take(2).collect();
            if parts.len() == 2 {
                match (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                    (Ok(a), Ok(b)) => Some((a, b)),
                    _ => None,
                }
            } else {
                None
            }
        };
        let (ra, rb) = parsed.unwrap_or((bud[0], bud[1]));
        let frac = if ra + rb > 0.0 { ra / (ra + rb) } else { 0.5 };
        if mode == "layer" {
            let k = py_round(frac * (info.n_layer as f64 + 1.0)).min(n_eff as i64).max(0) as usize;
            let e = dual_estimate(info, s, &order, k, side_bytes, oh, adjust);
            if spread_on(s, info) {
                plan.cpu_set = Some(spread_cpu_moe(info, gi(s, "n_cpu_moe", 0), k));
            }
            plan.fits = e[0].total <= bud[0] && e[1].total <= bud[1];
            plan.k = Some(k);
            plan.ts = Some(lay_ts(k));
            plan.est = [Some(e[0].clone()), Some(e[1].clone())];
        } else {
            // row / tensor: every tensor split by ratio
            let mut s1 = s.clone();
            s1.insert("ngl".into(), (n_eff as i64 + 1).into());
            let whole = estimate_vram(info, &s1, side_bytes.0 + side_bytes.1, 0.0, 0.0, None, None, None);
            let share = [frac, 1.0 - frac];
            let mut e: Vec<Est> = Vec::new();
            for (i, sh) in share.iter().enumerate() {
                let mm = if i == 1 { whole.mmproj } else { 0.0 };
                let t = (whole.weights + whole.kv + whole.state) * sh + whole.compute + mm + oh * MIB;
                e.push(Est {
                    weights: whole.weights * sh,
                    kv: whole.kv * sh,
                    state: whole.state * sh,
                    compute: whole.compute,
                    mmproj: mm,
                    overhead: oh * MIB,
                    total: t,
                });
            }
            plan.fits = e[0].total <= bud[0] && e[1].total <= bud[1];
            plan.ts = Some(if !m.ts.is_empty() {
                m.ts.clone()
            } else {
                format!("{},{}", (bud[0] / GIB) as i64, (bud[1] / GIB) as i64)
            });
            plan.est = [Some(e[0].clone()), Some(e[1].clone())];
        }
        return plan;
    }

    // ---- Auto strategies ----
    let est = |k: usize| dual_estimate(info, s, &order, k, side_bytes, oh, adjust);
    let fits = |e: &[Est; 2]| e[0].total <= bud[0] && e[1].total <= bud[1];
    if strategy == "Auto Speed" {
        let mut s1 = s.clone();
        s1.insert("ngl".into(), (n_eff as i64 + 1).into());
        let single = estimate_vram(
            info, &s1, side_bytes.0 + side_bytes.1, oh,
            adjust.get(&format!("{}:single", big.id)).cloned().unwrap_or(0.0), None, None, None,
        );
        if single.total <= bud[1] {
            plan.mode = "none".into();
            plan.note = format!("fits on {} alone - single GPU is fastest (no PCIe hand-off)", big.name);
            plan.single = Some(big.clone());
            plan.est = [None, Some(single)];
            plan.fits = true;
            return plan;
        }
    }
    let mut candidates: Vec<(usize, [Est; 2])> = Vec::new();
    for k in 0..=n_eff {
        let e = est(k);
        if fits(&e) {
            candidates.push((k, e));
        }
    }
    let spread_set = |k: usize| if spread_on(s, info) { Some(spread_cpu_moe(info, gi(s, "n_cpu_moe", 0), k)) } else { None };
    if candidates.is_empty() {
        // nothing fits: report the most even split so the user sees by how much
        let mut best_k = 0;
        let mut best_v = f64::NEG_INFINITY;
        for k in 0..=n_eff {
            let e = est(k);
            let v = (bud[0] - e[0].total).min(bud[1] - e[1].total);
            if v > best_v {
                best_v = v;
                best_k = k;
            }
        }
        let e = est(best_k);
        plan.k = Some(best_k);
        plan.est = [Some(e[0].clone()), Some(e[1].clone())];
        plan.fits = false;
        plan.ts = Some(lay_ts(best_k));
        plan.note = "exceeds the safety margins at this context".into();
        plan.cpu_set = spread_set(best_k);
        return plan;
    }
    let (k, e, mut note) = if strategy == "Auto Speed" {
        // least work on the smaller card
        let c = if big_first { candidates.last().unwrap() } else { &candidates[0] };
        (c.0, c.1.clone(), "layer split, as little as possible on the smaller card".to_string())
    } else if strategy == "Auto VRAM" {
        let mut best = &candidates[0];
        let mut bv = f64::NEG_INFINITY;
        for c in &candidates {
            let v = (bud[0] - c.1[0].total).min(bud[1] - c.1[1].total);
            if v > bv {
                bv = v;
                best = c;
            }
        }
        (best.0, best.1.clone(), "layer split, maximum free room on both cards".to_string())
    } else {
        let mut best = &candidates[0];
        let mut bv = f64::INFINITY;
        for c in &candidates {
            let v = (c.1[0].total / tot[0]).max(c.1[1].total / tot[1]);
            if v < bv {
                bv = v;
                best = c;
            }
        }
        (best.0, best.1.clone(), "layer split, both cards filled to the same percentage".to_string())
    };
    if k == 0 && !big_first {
        plan.mode = "none".into();
        plan.single = Some(big.clone());
        plan.est = [None, Some(e[1].clone())];
        plan.fits = true;
        plan.note = format!("fits on {} alone - single GPU", big.name);
        return plan;
    }
    if big_first {
        note += &format!("; {} first (handles RAM-expert streaming)", big.name.replace("AMD Radeon ", ""));
    }
    plan.k = Some(k);
    plan.est = [Some(e[0].clone()), Some(e[1].clone())];
    plan.fits = true;
    plan.ts = Some(lay_ts(k));
    plan.note = note;
    plan.cpu_set = spread_set(k);
    plan
}

/// Architecture signature: models with the same one (e.g. fine-tunes of one base) need the
/// same extra buffers, so they can share a calibration offset.
pub fn model_sig(info: &ModelInfo) -> String {
    format!("{}:{}:{}:{}:{}", info.arch, info.n_layer, info.n_embd as i64, info.n_expert, info.nextn)
}

pub fn file_size(p: &Path) -> f64 {
    std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0)
}

#[allow(dead_code)]
pub fn opt_f(s: &Opts, k: &str) -> f64 {
    gf(s, k, 0.0)
}
