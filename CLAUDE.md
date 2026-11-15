# LLM Duo

Qt/Kirigami desktop app (Rust + cxx-qt) that launches llama.cpp, KoboldCpp, vLLM and Radiance
servers, one per GPU. README.md has the layout, build steps, data paths and the changelog: read
the section you need, not the whole file.

## Machine

2× AMD Radeon AI PRO R9700 32 GB (gfx1201), PCIe 3.0 x8 each, no P2P; i9-10850K (AVX2, no
AVX-512); 91 GB RAM. The user usually runs one model per card.

- Radiance checkout: `~/radiance` (binary `~/radiance/build/bin/radiance`, update with
  `~/radiance/update.sh`). Containers: `/mnt/1TB/Radiance`. vLLM models: `/mnt/1TB/LLM-VLLM`.
- Settings: `~/.config/llm-duo/llm_duo_settings.json` (holds a Radiance API key: never print it).
- Recipe guide the user keeps: `/mnt/1TB/Radiance/RECIPES-Gemma4-Qwen3.8-27B.md`.

## Build, test, install

- `cargo test --release`: unit tests in `src/duo` (no Qt runtime needed, but the crate links Qt).
- `scripts/install-local.sh`: release build + install to `~/.local/bin/llm-duo`. **Run it after
  every change the user will try.** The window title shows the build stamp; an old installed
  binary has caused confusion before.

## Code map (`src/duo/`)

| File | What |
|---|---|
| `pane.rs` | pane state and options (all backends) |
| `pane_run.rs` | Load / Stop / Copy command, pre-Load checks and confirmations |
| `pane_r.rs` | Radiance options, Fit, launch command (`r_launch_cmd`) |
| `radiance.rs` | Radiance feature detection, YAML (`rad-yaml`), log hints, container info |
| `rad_convert.rs` / `calib.rs` | Convert window; calibration Hessians (record + RADGRAM2 files) |
| `pane_v.rs`, `vllm.rs` / `pane_k.rs`, `kobold.rs` / `estimate.rs`, `gguf.rs` | vLLM / KoboldCpp / llama.cpp |
| `app.rs` | app state, `Pending` confirmations, poll loop |
| `bench.rs`, `prefix.rs`, `dash.rs`, `telemetry.rs`, `devices.rs`, `proc.rs` | BetterBench, prefix-cache calc, dashboards, sensors, GPUs, processes |

`src/bridge.rs` is the single QML singleton; UI in `qml/`; tooltip texts in `assets/help.json`.

## Radiance

- Versions are detected from `radiance --help` with `binary_has_flag`, not a version number:
  `--config` → 2.0 (YAML-only config, built from the flag list via `rad-yaml`); `stuck-exit` →
  2.1 (i8 KV + DFlash2 draft trees fixed, #454). 1.x still gets flags.
- 2.x KV cache types: bf16 and i8 (fp8 is GLM-5.3 only). Check `~/radiance` sources and
  `data/recipes/` rather than assuming; Radiance changes fast.

## Rules from the user

- **Never silently change, force or substitute the user's options** as a workaround (e.g. forcing
  bf16). Warn, explain, and let them choose.
- **Tell the user up front** what a choice needs (e.g. recipes needing Hessians are marked in the
  list and alerted on selection), not when they press Start.
- **Never patch Radiance.** Report upstream bugs as a write-up for the user to file.
- Don't convert, download or run models unless asked. Before any GPU test, check which card the
  user's servers are on and leave those cards alone.
- API keys go in environment variables only: never on command lines, in YAML, or in logs.
- Recipe guides reference **official Radiance containers and recipes only**. Recommend only
  setups with ≥100K context, preferring full context; state whether Hessians are needed and
  whether they can be made on 2× 32 GB.
- Verify claims (measure or read the source) before stating them; say plainly what was not tested.

## Releases

Changelog in README.md (`## Changelog`, newest first, Added/Changed/Fixed), version in
`Cargo.toml`, then commit, push `main`, tag `vX.Y.Z`, `gh release create` with notes from the
changelog. Commit messages: one imperative summary line, short body explaining why.
