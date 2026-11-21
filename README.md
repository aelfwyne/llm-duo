# LLM Duo (Rust + Qt 6 / Kirigami)

Run one **llama-server**, **KoboldCpp**, **vLLM** or **Radiance** server per GPU from a two-pane
KDE Plasma GUI.

- one pane per GPU, each with its own backend (llama.cpp, KoboldCpp ROCm/Vulkan, Radiance, vLLM — experimental)
- per-(GPU, model) options saved automatically; live VRAM estimate while you change them; **Fit**
- Dual GPU mode (one model across both cards: layer/row/tensor split for llama.cpp/KoboldCpp,
  pipeline/tensor parallel for vLLM)
- MTP drafter detection, relaxed chat templates, calibration from real loads, long-term speed stats
- live status (state, prompt/gen speed, prompt-cache reuse, GPU buffers, VRAM/busy from sysfs)
- detection of models pushed out of VRAM, crash reports saved to `logs/`, optional chime
- FP8 quantize tool for vLLM models
- **Dashboard** (live performance) and **History** (every configuration) windows - see below

![LLM Duo running llama.cpp on one card and vLLM on the other](docs/screenshot.png)

*(Captured headless against stand-in servers, so it shows Qt's Fusion style; on Plasma it uses Breeze.)*

## Layout

| Path | What it is |
|---|---|
| `src/duo/` | all application logic in plain Rust (no Qt): GGUF/safetensors readers, VRAM estimates, dual-GPU and vLLM planners, process control, log/metrics parsing, settings, stats |
| `src/bridge.rs` | the cxx-qt bridge: one QML singleton (`Backend`) that forwards to `src/duo` |
| `cpp/launcher.cpp` | creates the `QApplication` + QML engine (needed for the Plasma style) |
| `qml/` | the Kirigami / Qt Quick Controls user interface |
| `assets/` | tooltip texts and the FP8 quantizer script (embedded into the binary) |
| `packaging/` | `.desktop` file and icon |

## Build on Ubuntu 26.04 LTS

These steps work on Kubuntu 26.04 and on Ubuntu 26.04 with any desktop (the KDE look needs the
packages from step 1; on GNOME the window simply uses the Qt Fusion style if the KDE desktop
style isn't installed).

### 1. System packages

```bash
sudo apt update
sudo apt install build-essential pkg-config git curl lld clang-format \
    qt6-base-dev qt6-base-dev-tools qmake6 \
    qt6-declarative-dev qt6-declarative-dev-tools \
    qml6-module-qtquick qml6-module-qtquick-controls qml6-module-qtquick-layouts \
    qml6-module-qtquick-templates qml6-module-qtquick-window qml6-module-qtquick-dialogs \
    qml6-module-qtqml-workerscript \
    qml6-module-org-kde-kirigami qml6-module-org-kde-desktop \
    vulkan-tools
```

- `qmake6` is how the build finds Qt; `qt6-declarative-dev-tools` provides `qmltyperegistrar` and
  `qmlcachegen`, which the build runs for the QML module.
- `qml6-module-org-kde-kirigami` and `qml6-module-org-kde-desktop` give the Plasma look
  (already installed on Kubuntu).
- `vulkan-tools` (`vulkaninfo`) is only needed for KoboldCpp's / llama.cpp's Vulkan builds; it maps
  each GPU to its Vulkan device number.
- The chime uses `paplay`, `pw-play` or `aplay`, whichever is installed (Ubuntu desktops have one).

### 2. Rust toolchain

cxx-qt 0.10 needs Rust 1.85 or newer. Use rustup (Ubuntu packages it):

```bash
sudo apt install rustup
rustup default stable
rustc --version        # 1.85 or newer
```

(or install rustup from https://rustup.rs).

### 3. Build and run

```bash
cd llm-duo
export QMAKE=/usr/bin/qmake6      # tells cxx-qt which Qt to use
cargo build --release
./target/release/llm-duo
```

The first build downloads the crates (cxx, cxx-qt, serde_json, regex, libc) and compiles the
C++ side; it takes a few minutes. `qmake6 -query QT_VERSION` shows which Qt is used (6.10 on 26.04).

### 4. Install for your user (optional)

```bash
scripts/install-local.sh            # builds, then installs to ~/.local/bin + menu entry + icon
scripts/install-local.sh --uninstall
```

After installing, LLM Duo appears in the application launcher (Development / Utilities).

## Where settings and data live

Settings live in `~/.config/llm-duo/llm_duo_settings.json`; logs, relaxed templates, the
quantizer script and the chime sound in `~/.local/share/llm-duo/`.

To keep everything (settings, `logs/`, `templates/`, `tools/`) in one folder instead, set
`LLM_DUO_DIR=/path/to/folder`, or put the binary next to an existing `llm_duo_settings.json`
(portable mode).

## Performance dashboards

**Dashboard** (toolbar) shows both servers and every card on one screen (sized for a 1440-pixel-high
display). Request figures come from llama.cpp for now; for KoboldCpp and vLLM only the GPU graphs fill in.

- one strip per server: last and average generation / prompt speed, draft (MTP) acceptance, prompt-cache
  reuse, context in use, requests and aborts, plus what it is doing right now (prompt progress)
- per-request generation and prompt speed and speed against context, coloured by server
- GPU memory, load, hotspot temperature and power over the last 5 / 10 / 30 minutes, one line per card;
  per card: edge / hotspot / memory temperature, fan and clocks with their peaks (amdgpu sysfs; sampled
  even while the window is closed)
- one request log for both servers, and each server's llama-server `/metrics`
- analysis: low draft acceptance, prompt cache not reused, context nearly full, cards hot or power-limited,
  VRAM full or spilled, uneven load in Dual GPU mode, slower or faster than the setup's history. Findings stay
  until you dismiss them (marked "cleared" once the condition is gone); with none left, the status lines show.

![Live dashboard (Dual GPU mode, test data)](docs/dashboard.png)

**History** lists every configuration (GPU/backend + model + load options) with runs, failed and
cancelled loads, requests, aborted requests, crashes, RAM spills, speeds and draft / cache figures,
charts the fastest setups, and shows the request-by-request trend of the one you click.

The totals come from the `stats` section of the settings file. LLM Duo also appends every load,
request, abort and crash to `~/.local/share/llm-duo/history.jsonl`, and keeps the totals of a
configuration there when its load options change (the settings file starts those totals over).

![History (with real-world settings, test machine)](docs/history.png)

## Radiance backend

Pick **Radiance** for a pane in Settings and fill in the Radiance section:

| setting | typical value |
|---|---|
| Radiance binary | `~/radiance/build/bin/radiance` (rad-info and rad-convert are found beside it) |
| Radiance home | `~/radiance/build/radiance_home` (the engine's plugins) |
| Radiance models folder | `/mnt/1TB/Radiance` |
| Recipes folder | `~/radiance/data/recipes` |

The model list shows `.rad` containers and checkpoint folders (served at their own precision).
`RADIANCE_HOME` and `HIP_VISIBLE_DEVICES` go on Radiance's own command line only, never into your
shell, so the other card stays visible to everything else.

**Radiance 2.0** reads its settings from a YAML file instead of command-line flags. LLM Duo still
builds the familiar flag list (Extra args included), turns it into that file with `rad-yaml` from
the same folder as the radiance binary, and starts `radiance --config FILE`. The files live in the
data folder under `radiance/` (`left.yaml`, `right.yaml`, `*-plan.yaml` for Fit, `*-copy.yaml` for
*Copy command*), and the log shows the command and the YAML. A 1.x binary still gets flags.
*Copy command* gives a line that runs as is:

```
RADIANCE_HOME=… HIP_VISIBLE_DEVICES=0 RADIANCE_API_KEY=… radiance --config ~/.local/share/llm-duo/radiance/left-copy.yaml
```

**Memory.** Radiance claims the whole card at startup and splits it between weights and its KV
pool itself, so a full VRAM bar is normal. LLM Duo doesn't guess the split. It shows what Radiance
reported for the same memory settings: KV cache type, Max seqs, Batch tokens, headroom,
placements, host pool and spec tokens. **Fit** reads that figure. The first time for a set of
memory settings, Fit runs a short *plan check*: it starts radiance on a spare port, reads the
`kv cache … tokens of context` line and stops it. If the start fails, it bisects the context.
Every real load records the figure as well. Spill detection and VRAM calibration are off for
Radiance, because it pins host memory on purpose (the embedding, and the vision tower set to
`ram`).

**Defaults.** Max seqs is 1, because the engine's own 256 leaves only a few thousand tokens of
context on a 32 GB card. KV cache defaults to i8 (Radiance 2.0's 8-bit cache, which replaced
fp8); the engine's default is bf16. Host pool is the model's total, split among its cards.

**Prompt lookup** (Radiance 1.15) drafts from the sequence's own n-grams with no draft model. Set
Prompt lookup to the most drafts a step; it replaces Spec tokens while it is on. A container with
a draft head refuses it, so use Spec tokens there, and lookup fills in where the head declines.

**Embedding, rerank and decision models** load like any other and show their endpoints in the
model summary. Embedding and rerank models read each input whole, so Batch tokens has to reach the
longest input. A model with no KV cache has nothing for Fit to measure, and Fit says so.

**API key.** With Radiance 1.15.1 or later, the key from Settings reaches the server through
`--api-key-env` (`server.api-key-env` in 2.0), so it doesn't show in `ps` or in the YAML file.
Older builds get `--api-key` on the command line.

**Gemma 4 31B with a DFlash2 drafter** fails its first request in Radiance 2.0.x with the i8 KV
cache (its draft trees can't move i8 rows). Radiance 2.1.0 fixes this (#454). On 2.0.x LLM Duo
warns before Load and changes nothing; there it runs with the bf16 cache, or with Spec tokens 0.

**Dual GPU mode** runs `--tp 2` (tensor parallel), because Radiance has no layer split. The KV
pool is sized to the tighter card, so an R9700 + 9070 XT pair usually holds less context than the
R9700 alone. The estimate says so.

**Status and dashboards.** These come from the startup log (KV pool, full-length sessions,
arena, the `settled at … free` line) and from `/metrics` and `/stats` (decode step time and
best/recent floor, throughput, KV in use, draft acceptance, preemptions, evictions). Each request
in the dashboard and history is derived from counter deltas between idle polls. A client
disconnect counts as an abort. *Open dashboard* opens Radiance's own web dashboard.

**Convert…** (top bar, shown while a pane uses Radiance) runs `rad-convert` on a checkpoint
folder. It takes a recipe from the recipes folder, read each time the window opens, and fills in
the `--set` keys the recipe's header asks for. It can also add a draft model, a max context, an
environment (such as `CALIB=`) and plan-only. The new container appears in the model list when it
finishes.

**Calibration Hessians.** Recipes fitted by GPTQ (`calib=$CALIB`) are marked "⚠ needs Hessians"
in the Recipe list. Picking one pops up a warning, a banner holds the *Calibration data* folder
(`gram.r*.bin` files) with a live check, and Start asks before converting without them.
*Create Hessians…* records them for dense Qwen3.5 / Qwen3.8: it serves the bf16 checkpoint with
Radiance 2.0's calibration mode over a corpus (a `.jsonl` file of `prompt` / `text` rows or chat
rows with `messages`, which are rendered with the checkpoint's own chat template, or a folder of
text files), in layer slices when host RAM is short, converts the recordings into the files rad-convert reads, and
sets Calibration data. The bf16 weights must fit the chosen cards (Qwen3.8-27B needs two 32 GB
cards); Gemma 4 records its Hessians another way and isn't supported.

Recipes that use repository-relative codebooks (`vq=`, `tcq_lut=`) run from the Radiance
checkout, so the recipes folder has to stay inside it. The output is named after the recipe's
format and the merged drafter (`-dflash2`, `-dflash`, `-assistant`, …). Decision
checkpoints are recognised: clef gets `decision.type=clef`, `--arch clef` and its
`joint_head.safetensors`, and Laya gets `decision.type=laya`. Laya folders with no `config.json`
are listed too. Naming `--arch` in Extra args turns the clef additions off.

The log parsing is written from Radiance's documentation, not a live log. If a status line stays
empty or Fit says it couldn't read the pool, send the startup log and the parser can be adjusted.

### Fleet mode (several models, one process)

Radiance 2.0 serves any number of models from one process (`docs/MULTIMODEL.md` in the Radiance checkout): a request picks its
model by its `model` field, and Radiance loads, parks (to host RAM) and unloads models as requests
arrive. A parked model comes back in a few seconds instead of a cold load.

1. Set a model up on a Radiance pane as usual (Fit, KV cache, prefix cache, speculative decoding,
   Extra args) and press **Pin to fleet**. The pin is a copy of those settings. On a single pane it
   becomes `tp: 1`, and Radiance picks the card when it loads. In Dual GPU mode it becomes `tp: 2`
   and takes both cards whenever it is loaded. No setting puts a model on one particular card.
2. Tick **Fleet mode** in the toolbar, with both panes stopped: the fleet uses both cards
   (`cards: all`). Each pin has the fleet's per-model keys: aliases (`default` takes requests that
   name no model), priority, keep in VRAM (`pin`), idle TTL, park to RAM / restore, and KV / expert
   flex limits. Below them are the fleet settings (port, host memory, disk tier, load timeout,
   warm-up) and `together` / `exclusive` groups. Warm-up is off by default: Radiance's warm-up
   loads models nobody asked for, an idle TTL never parks a model that hasn't served a request,
   and a model needing room may evict one you are using instead.
3. **Check** builds the file and runs `radiance --config FILE --check`, which only parses and
   validates and does not check that the models fit. **Load** starts the fleet. The table shows
   each model's state from `/admin/fleet`, with Load / Park / Unload / Keep buttons.
4. Edits while it runs light up **Apply**, which sends `POST /admin/reload`: changed models reload
   and the others keep serving. A change to the port, host memory or disk tier needs a restart,
   and Apply asks before restarting.

Each pin's flags go through `rad-yaml`, so Radiance's own parser builds the model entry.
Process-wide settings found in a pane's Extra args (`server.log`, …) are listed when you pin and
are left out. The file is `~/.local/share/llm-duo/radiance/fleet.yaml`, and the API key reaches it
through `RADIANCE_API_KEY` only. A model's claim in a fleet is its weights plus one full-length
context of KV, so the Context you Fit on a pane is what it asks for. When you pin, the claim is
estimated from the pane's last load of that model. A context Fit tightly while the model ran
alone may need a little trimming beside other models.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not find qmake` / wrong Qt picked | `export QMAKE=/usr/bin/qmake6` before `cargo build` |
| `module "org.kde.kirigami" is not installed` at start | `sudo apt install qml6-module-org-kde-kirigami` |
| `module "QtQuick.Dialogs" is not installed` | `sudo apt install qml6-module-qtquick-dialogs` |
| Looks like plain Fusion, not Breeze | `sudo apt install qml6-module-org-kde-desktop` (or force a style: `QT_QUICK_CONTROLS_STYLE=org.kde.desktop llm-duo`) |
| Linker errors mentioning `cxx_qt_init_...` or `ld.bfd` | `sudo apt install lld` and build with `RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build --release` |
| Need the QML engine's own messages | run from a terminal: `QT_LOGGING_RULES="qml.debug=true" llm-duo` |

## Development

The logic in `src/duo` has unit tests that run without Qt:

```bash
cargo test     # needs the Qt dev packages too, since the crate links Qt
```

## Changelog

### 1.1.2

Added:

- **Fleet mode** (Radiance 2.0+): pin a Radiance pane's configuration with *Pin to fleet*, then
  serve every pin from one radiance process on both cards (multi-model serving, hot swapping by
  the request's `model` field). It edits the fleet keys per model (aliases, priority, pin, ttl,
  park, flex) and the fleet's port, host memory, disk tier, load timeout, warm-up and groups. It
  checks with `radiance --check`, shows each model's state from `/admin/fleet` with Load / Park /
  Unload / Keep, and applies edits by hot reload (`/admin/reload`). Radiance's warm-up is off
  unless you turn it on, so only models that requests name take up a card.
- *Create Hessians…* reads chat rows (`{"messages": […], "tools": …, "thinking": …}`, the format
  of quant-calibration datasets) and renders them with the checkpoint's own chat template, the way
  transformers' `apply_chat_template` does, so such a corpus no longer needs rendering in Python
  first. Rows the template refuses are left out and counted in the log.
- **N-gram table** option on the Radiance pane (`--ngram-placement`: vram, ram or disk) for
  models with an n-gram embedding table (Qwen3.8-Flash-Next's is 47.7 GiB). The estimate shows
  the table's size and where it goes, leaves it out of the card's weights when it is in RAM or on
  disk, and warns before Load when Radiance's own choice could pin it in RAM beside a Host pool,
  when ram needs more memory than is free, or when vram cannot hold it.
- Radiance pane: an **Advanced** section (control vector, RoPE scaling / YaRN, state cache f16, KV
  resident, weight prefetch, prefill interleave, prefix cache TTL, weights disk tier) and a
  **Sampling defaults** section (temperature, top-k/p, min-p, typical-p, the presence, frequency
  and repetition penalties, DRY, XTC, default max tokens), both folded, with a header that counts
  the options set. Dual GPU gains *Attention split* and *Peer access*. Each has a tooltip on what
  it does and which models take it. Before Load the estimate flags in red what Radiance would refuse
  at startup: a control vector on a model other than Flash-Next, an f16 state cache outside the qwen35
  plugins, the row attention split outside GLM-5.3, and sampling values out of range. With RoPE
  scaling set, a Context past the trained length is measured against what YaRN reaches.
- Recipes whose GPTQ rules name a fixed Hessians folder (`calib=/calib`, as the recipes taken from
  keptn's Gemma 4 31B containers do) are marked "⚠ needs Hessians (calib=/calib)" and explained
  when picked. With Calibration data set, Start asks whether to run a copy of the recipe that reads
  it as `$CALIB` (the recipe file is not changed) or the recipe as it is.
- `hf-model-finder.html` finds Radiance containers: add `radiance` or `rad` to the query (or pick
  *radiance (.rad)* under Format) to list only repos that ship a `.rad` file, filter by container
  quant (`w4a8`, `i8g`, `iq4r`, …) and by drafter (MTP, DFlash, DFlash2, DSpark), and copy a
  resumable curl command for each `.rad` file.

Fixed:

- Radiance **Fit** reported a full-context fit for a model larger than the card (Gemma 4
  26B-A4B w8a8, 35 GiB, on one 32 GB card). Radiance states the KV pool before it places the
  weights, and the plan check stopped right there, so it never saw the `DID NOT FIT` that follows.
  It now waits until every card has passed placement. When Radiance warns that the experts do not
  fit beside one full-length sequence, Fit tries the context that frees enough for them, or, when
  none does, says so with the Host pool that would make room (or Dual GPU mode). Plans recorded
  the old way are no longer used; press Fit again.
- After Fit, Load could say the settings don't fit: the plan check was still exiting and holding
  its VRAM (≈ 18 GiB a card), which the estimate read as "already in use". Load, the estimate and
  Fit's next try now wait until the stopped plan check has exited.
- Image generators: LLM Duo always passed Batch tokens, which overrides the step size Radiance
  takes from the model's plugin, so the largest images were refused at request time ("an image
  with its conditioning needs 16525 rows a step and a step carries 16384"). LLM Duo now knows each
  generator plugin's step (krea2, anima, sdxl, qwen-image 2.1 and flux3 exactly; video, audio and
  3D generators by name). The estimate says before Load what the largest image needs (twice with
  guidance) and what fits, the Fleet's checks name a pin that would refuse its largest image, and
  *Radiance sizes it* (generators only) leaves Batch tokens, Max seqs and Context to Radiance;
  a generator picked for the first time starts with it ticked, and saved settings are kept.
- A fleet pin's estimated claim counted the whole container as weights on the cards, so a model
  that keeps experts in the host pool or its n-gram table on disk (Qwen3.8-Flash-Next at tp 2:
  ≈ 59 GiB a card) was reported as larger than a card and "may never load", though it loads. The
  claim now leaves out the host pool and an n-gram table kept off the cards. Re-pin to update it.
- A Radiance load that stopped with "out of memory" because the n-gram table was pinned in RAM and
  left the Host pool full now says so and suggests N-gram table → disk. It used to say something
  else was holding VRAM. A container on a filesystem without direct I/O (the disk tier needs it)
  gets its own hint.
- Servers (panes, Fit's plan check, the fleet) stop when LLM Duo ends in any way, not only when
  its window is closed: closing the terminal it was started from, Ctrl+C, a kill or a crash used
  to leave them running, holding the cards.
- *Pin to fleet* on a pane whose model is already pinned under another name (a pin renamed, say
  `Nemotron-Embed`) asks "This model is already pinned as Nemotron-Embed. Replace that pin?". It
  used to look only for a pin with the file's own name, find none, and add a second pin, so the
  new settings went to a copy that requests for the renamed pin never reached.
- The `--set quantization_config.quant_method=compressed-tensors` key is filled in for dense
  Gemma 4 int5 and int8 recipes too (keptn's `w5` and `i8g`), not only int4. Gemma 4 26B-A4B is
  told apart as a mixture of experts and still gets none.

### 1.1.1

Changed:

- The warning before Load for Gemma 4 + DFlash2 with the i8 KV cache only appears on Radiance
  2.0.x. Radiance 2.1.0 fixed the crash (#454); LLM Duo recognises 2.1 by the `stuck-exit`
  setting in `radiance --help`. The warning on 2.0.x now points at `~/radiance/update.sh`.

### 1.1.0

Radiance 2.0 support, BetterBench from a pane, and a fuller load history.

Added:

- *BetterBench…* next to *Open web UI*: runs `betterbench run` in the background against the
  pane's endpoint and served model, with its options in a window, and opens the HTML report when
  done. The API key goes in `BETTERBENCH_API_KEY`, not on the command line.
- The history window lists every recorded load option (tooltips, the All configurations table and
  the Details pane, with differences first), and *Restore these settings* saves a configuration's
  options as the model's settings on the matching pane.

- Radiance 2.0 launches: the flag list (Extra args included) becomes the YAML file 2.0 reads, via
  `rad-yaml`, and radiance starts with `--config`. Load, Fit's plan check and *Copy command* all
  use it; the log shows the YAML. A 1.x binary still gets flags.
- Calibration Hessians in Convert: recipes that need them are marked in the Recipe list, picking
  one or opening Convert with one selected pops up a warning, a banner holds the Calibration data
  folder with a live check, and Start (Plan only included) asks before running without them.
- *Create Hessians…*: records GPTQ Hessians for dense Qwen3.5 / Qwen3.8 with Radiance 2.0's
  calibration mode, in layer slices when host RAM is short, and converts them into the
  `gram.r0.<weight>.bin` files rad-convert reads.
- A prefix-cache calculator (*Calc*) for llama.cpp `--cache-ram`, vLLM `--kv-offloading-size` and
  Radiance `--prefix-cache-mib`: the system RAM for N finished conversations of a given length.
- A warning before Load for Gemma 4 + DFlash2 with the i8 KV cache, which Radiance 2.0 loads and
  then fails on the first request. Nothing is changed for you.
- Error hints for Radiance 2.0's refusals: weight formats it no longer serves (block FP8, MXFP4, …)
  and removed KV cache types.

Changed:

- KV cache types are bf16 and i8 (2.0 removed fp8, fp8_v4 and k6_v4); i8 is the default, and a
  saved fp8 setting becomes i8.
- Host pool is the model's total, split among its cards, as in 2.0 (1.x pinned it on every card).
- With 2.0 the API key always travels in the environment, so it is never written to the YAML file.
- A served model name has `:` replaced (2.0 reserves it for adapters).
- Convert names the output after the merged drafter (`-dflash2`, `-dflash`, `-assistant`, …)
  instead of `-mtp` for every drafter.

Fixed:

- Identical GPUs are matched to their sysfs card through the KFD topology, so each pane reads its
  own card's memory (before, on a system where HIP's device 0 is card1, the panes swapped).
- A model's Context is kept when switching models; the typed value was saved against the new model.
- Readiness on Radiance 2.0, which no longer logs "serving on".
- Recipes whose codebooks are named by `tcq_lut=` (IQR-V2 to V4) run from the Radiance checkout.
- Number boxes no longer rewrite the number while it is being typed.

### 1.0.0

First release.

Added:

- Prompt lookup, N-gram min and N-gram max controls for Radiance 1.15.
- Prompt lookup and forced preemption figures in the Radiance status line.
- Model kind and endpoints in the model summary, for embedding, rerank and decision models.
- clef and Laya recognised in Convert, with Laya folders listed as sources.
- A Batch tokens limit of 262144, and a warning when it's too low for an embedding or rerank model.

Changed:

- With Radiance 1.15.1 or later, the API key goes through `--api-key-env`, so it doesn't show in
  `ps`. It is also masked in the log and reports.

Fixed:

- Recipes with repository-relative codebooks run from the Radiance checkout, instead of failing
  with `vq=… cannot be opened`.
- The `CALIB` warning appears for recipes that need it.
- Startup refusals show the real reason: CWSR, plugin refusals, `--spec-ngram` and `--api-key-env`.
- Fit says when a model has no KV cache, instead of asking for a startup log.
- Prefix-cache log lines no longer overwrite the KV pool figure.
- Stale Radiance help text.
