//! Defaults, constants and tooltip texts.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

use super::util::{home, Opts};

pub const APP_NAME: &str = "LLM Duo";

pub const BACKENDS: [&str; 4] = ["llama.cpp", "KoboldCpp", "vLLM", "Radiance"];
pub const LLAMA: &str = "llama.cpp";
pub const VLLM: &str = "vLLM";
pub const KOBOLD: &str = "KoboldCpp";
pub const RADIANCE: &str = "Radiance";
pub const RADIANCE_FG: &str = "#6250d6";
pub const EXPERIMENTAL_FG: &str = "#c62828";

pub fn backend_label(b: &str) -> &'static str {
    match b {
        "llama.cpp" => "llama.cpp",
        "KoboldCpp" => "KoboldCpp",
        "vLLM" => "vLLM (Experimental)",
        "Radiance" => "Radiance",
        _ => "llama.cpp",
    }
}

pub const KOBOLD_BUILDS: [&str; 2] = ["ROCm", "Vulkan"];
pub const KOBOLD_BATCHES: [i64; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

pub const KV_TYPES: [&str; 3] = ["f16", "q8_0", "q4_0"];

pub fn cache_bytes(t: &str) -> f64 {
    match t {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        _ => 2.0,
    }
}

pub const DUAL_STRATEGIES: [&str; 4] = ["Auto Balance", "Auto Speed", "Auto VRAM", "Manual"];
pub const VLLM_STRATEGIES: [&str; 3] = ["Auto Speed", "Auto VRAM", "Manual"];
pub const VLLM_SPLIT_MODES: [&str; 2] = ["pipeline", "tensor"];
pub const VLLM_KV_TYPES: [&str; 2] = ["auto", "fp8"];
pub const VLLM_DTYPES: [&str; 3] = ["auto", "bfloat16", "float16"];
pub const VLLM_ONLINE_QUANT_KEYS: [&str; 5] = ["(none)", "fp8_per_block", "fp8_per_tensor", "mxfp8", "mxfp4"];

pub fn vllm_online_quant(k: &str) -> Option<f64> {
    match k {
        "(none)" => Some(1.0),
        "fp8_per_block" | "fp8_per_tensor" => Some(0.5),
        "mxfp8" => Some(0.516),
        "mxfp4" => Some(0.266),
        _ => None,
    }
}

pub const VLLM_REASONING: [&str; 9] = [
    "(none)", "qwen3", "deepseek_r1", "deepseek_v3", "glm45", "openai_gptoss", "mistral", "granite", "hunyuan_a13b",
];

pub const STATS_IGNORE: [&str; 3] = ["port", "kquiet", "kbuild"];

pub fn default_global() -> Opts {
    let h = home();
    let hs = |p: &str| h.join(p).to_string_lossy().into_owned();
    let v = json!({
        "models_dir": hs("models"),
        "server_bin": hs("llama.cpp/build/bin/llama-server"),
        "server_bin_vulkan": hs("llama.cpp/build-vulkan/bin/llama-server"),
        "host": "127.0.0.1",
        "left_device": "",
        "right_device": "",
        "left_port": 8080,
        "right_port": 8081,
        "overhead_mib": 300,
        "fit_margin_mib": 1536,
        "auto_load": false,
        "dual": false,
        "chime": false,
        "chime_min_secs": 5,
        "left_backend": "llama.cpp",
        "right_backend": "llama.cpp",
        "vllm_bin": hs("vllm/.venv/bin/vllm"),
        "vllm_models_dir": "/mnt/1TB/LLM-VLLM",
        "vllm_env": "HF_HUB_CACHE={models}/.hf-cache",
        "vllm_left_port": 8042,
        "vllm_right_port": 8043,
        "kobold_rocm_bin": "",
        "kobold_vulkan_bin": "",
        "kobold_left_port": 5001,
        "kobold_right_port": 5002,
        "kobold_flags": {},
        "radiance_bin": hs("radiance/build/bin/radiance"),
        "radiance_home": hs("radiance/build/radiance_home"),
        "radiance_models_dir": "/mnt/1TB/Radiance",
        "radiance_recipes_dir": hs("radiance/data/recipes"),
        "radiance_env": "",
        "llama_api_key": "",
        "kobold_password": "",
        "vllm_api_key": "",
        "radiance_api_key": "",
        "radiance_left_port": 8080,
        "radiance_right_port": 8081,
        "last": {},
    });
    v.as_object().unwrap().clone()
}

pub fn default_model() -> Opts {
    json!({
        "ctx": 8192,
        "ngl": 999,
        "fa": "on",
        "kv": "f16",
        "n_cpu_moe": 0,
        "parallel": 1,
        "threads": 0,
        "batch": 2048,
        "ubatch": 512,
        "jinja": true,
        "relax_tpl": false,
        "dual_strategy": "Auto Balance",
        "dual_mode": "layer",
        "dual_ts": "",
        "dual_first": "",
        "dual_moe_spread": false,
        "mmproj": "(none)",
        "mtp": null,
        "mtp_n": 4,
        "mtp_dev": "(auto)",
        "port": null,
        "extra": "",
    })
    .as_object()
    .unwrap()
    .clone()
}

pub fn default_vllm_model() -> Opts {
    json!({
        "ctx": 8192,
        "gpu_util": 0.90,
        "kv_dtype": "auto",
        "dtype": "auto",
        "quant": "(none)",
        "parallel": 1,
        "max_batched": 0,
        "cpu_offload": 0,
        "eager": false,
        "text_only": false,
        "reasoning": null,
        "relax_tpl": false,
        "dual_strategy": "Auto VRAM",
        "dual_mode": "pipeline",
        "dual_ts": "",
        "dual_first": "",
        "mtp": "(off)",
        "mtp_n": 3,
        "port": null,
        "extra": "",
    })
    .as_object()
    .unwrap()
    .clone()
}

/// Radiance load options. Max seqs and Batch tokens default low: the engine's own defaults
/// (256 / 8192) are for many-user serving and give most of the card to the activation arena.
pub fn default_rad_model() -> Opts {
    json!({
        "ctx": 32768,
        "parallel": 1,
        "r_kv": "fp8",
        "r_batched": 2048,
        "r_headroom": 96,
        "r_spec": "auto",
        "r_draft_window": "fixed",
        "r_vision": "(default)",
        "r_embed": "(default)",
        "r_placement": "auto",
        "r_host_pool": 0,
        "r_prefix_mib": 0,
        "r_prefill_span": 1,
        "r_reasoning": "(default)",
        "r_tp_wire": "exact",
        "dual_strategy": "Tensor parallel",
        "dual_mode": "tensor",
        "port": null,
        "extra": "",
    })
    .as_object()
    .unwrap()
    .clone()
}

pub fn kobold_defaults() -> Opts {
    json!({"kbuild": null, "kquiet": true}).as_object().unwrap().clone()
}

/// Tooltip texts (same as the Python app's HELP dict).
pub fn help() -> &'static HashMap<String, String> {
    static H: OnceLock<HashMap<String, String>> = OnceLock::new();
    H.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../../assets/help.json")).expect("help.json");
        v.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
            .collect()
    })
}

pub fn help_text(k: &str) -> String {
    help().get(k).cloned().unwrap_or_default()
}
