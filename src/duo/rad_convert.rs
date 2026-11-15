//! Radiance convert tool: runs `rad-convert` on a Hugging Face checkpoint folder with a recipe
//! from Radiance's recipes folder (read when the window opens, so new recipes just appear) and
//! writes the `.rad` container into the Radiance models folder.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

use super::app::{App, Pending};
use super::proc::{self, Proc, ProcMsg};
use super::radiance::{
    bf16_plugin_mismatch, checkpoint_kind, decision_kind, known_sets, list_recipes, recipe_kind, recipe_needs_calib, recipe_needs_repo_cwd,
    recipe_sets, repo_of_recipe, scan_laya, sets_text, tool_path, Recipe,
};
use super::util::*;

pub const NO_RECIPE: &str = "(none: keep the checkpoint's own encodings)";

pub struct Conv {
    pub visible: bool,
    pub sources: Vec<String>,
    pub recipes: Vec<Recipe>,
    pub src: String,
    pub recipe: String,
    pub out: String,
    pub sets: String,
    pub draft: String,
    pub max_ctx: String,
    pub extra: String,
    pub env: String,
    /// the calibration folder (RADGRAM2 Hessians) a GPTQ recipe reads as $CALIB
    pub calib: String,
    /// "start without calibration data" was confirmed for the next start
    calib_confirmed: bool,
    /// the Create-calibration panel and its run
    pub cal: super::calib::Calib,
    pub plan_only: bool,
    /// what the Metadata box was last auto-filled with: it is replaced only while it still says that
    sets_auto: String,
    /// why the Metadata box holds what it holds (auto-fill), shown under it
    pub set_note: String,
    /// after a failed plan or convert: what went wrong, and the --set values that fix it
    pub hint: String,
    pub fix_sets: Option<String>,
    /// the output name LLM Duo suggested (replaced only while the user hasn't edited it)
    out_auto: String,
    /// GPU for rad-convert ("" = every card)
    pub gpu: String,
    pub info: String,
    pub about: String,
    pub status: String,
    pub proc: Option<Proc>,
    user_stopped: bool,
    t0: Option<Instant>,
    out_path: Option<PathBuf>,
    out_existed: bool,
    pub lines: VecDeque<String>,
    tx: Sender<ProcMsg>,
    rx: Receiver<ProcMsg>,
}

impl Conv {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Conv {
            visible: false,
            sources: vec![],
            recipes: vec![],
            src: String::new(),
            recipe: NO_RECIPE.into(),
            out: String::new(),
            sets: String::new(),
            draft: String::new(),
            max_ctx: String::new(),
            extra: String::new(),
            env: String::new(),
            calib: String::new(),
            calib_confirmed: false,
            cal: super::calib::Calib::new(),
            plan_only: false,
            sets_auto: String::new(),
            set_note: String::new(),
            hint: String::new(),
            fix_sets: None,
            out_auto: String::new(),
            gpu: "(auto)".into(),
            info: String::new(),
            about: String::new(),
            status: "Idle".into(),
            proc: None,
            user_stopped: false,
            t0: None,
            out_path: None,
            out_existed: false,
            lines: VecDeque::new(),
            tx,
            rx,
        }
    }
}

impl Default for Conv {
    fn default() -> Self {
        Self::new()
    }
}

/// A checkpoint's weights; a Laya release keeps its own in `encoder/`.
fn dir_size(d: &std::path::Path) -> f64 {
    let mut t = 0.0;
    if d.join("encoder").is_dir() {
        t += dir_size(&d.join("encoder"));
    }
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.filter_map(|e| e.ok()) {
            let n = e.file_name().to_string_lossy().to_lowercase();
            if n.ends_with(".safetensors") || n.ends_with(".gguf") {
                t += e.metadata().map(|m| m.len() as f64).unwrap_or(0.0);
            }
        }
    }
    t
}

/// A short name for a drafter repository, for output file names, read from its config.json:
/// "dflash2", "dflash", "dspark", "assistant", "eagle", else "draft".
pub fn drafter_tag(path: &std::path::Path) -> String {
    let dir = if path.is_file() { path.parent().unwrap_or(path) } else { path };
    let cfg: Value = std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let lc = |v: Option<&Value>| v.and_then(|x| x.as_str()).unwrap_or("").to_lowercase();
    let arch = cfg.get("architectures").and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" ").to_lowercase()).unwrap_or_default();
    let mt = lc(cfg.get("model_type"));
    let spec = lc(cfg.get("speculators_model_type"));
    let name = dir.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let has = |k: &str| arch.contains(k) || mt.contains(k) || spec.contains(k);
    if has("dflash2") {
        "dflash2"
    } else if has("dflash") || cfg.get("dflash_config").is_some() {
        "dflash"
    } else if has("dspark") {
        "dspark"
    } else if mt.ends_with("_assistant") || arch.contains("assistant") {
        "assistant"
    } else if has("eagle") {
        "eagle"
    } else if cfg.is_null() {
        // no readable config: fall back on the folder's own name
        if name.contains("dflash2") { "dflash2" } else if name.contains("dflash") { "dflash" } else if name.contains("dspark") { "dspark" } else if name.contains("assistant") { "assistant" } else { "draft" }
    } else {
        "draft"
    }
    .to_string()
}

#[cfg(test)]
mod drafter_tests {
    use super::drafter_tag;

    fn with_cfg(name: &str, cfg: &str) -> String {
        let d = std::env::temp_dir().join(format!("llmduo-drafter-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        if !cfg.is_empty() {
            std::fs::write(d.join("config.json"), cfg).unwrap();
        }
        let t = drafter_tag(&d);
        let _ = std::fs::remove_dir_all(&d);
        t
    }

    #[test]
    fn tags() {
        // RedHatAI's speculators-format DFlash2 (Gemma 4 31B)
        assert_eq!(with_cfg("a", r#"{"architectures":["DFlash2DraftModel"],"speculators_model_type":"dflash2"}"#), "dflash2");
        // z-lab's DFlash2 (Qwen3.8): a qwen3 model_type with a dflash_config
        assert_eq!(with_cfg("b", r#"{"model_type":"qwen3","architectures":["DFlash2DraftModel"],"dflash_config":{"block_size":8}}"#), "dflash2");
        // z-lab's DFlash
        assert_eq!(with_cfg("c", r#"{"model_type":"qwen3","architectures":["DFlashDraftModel"],"dflash_config":{"block_size":16}}"#), "dflash");
        assert_eq!(with_cfg("d", r#"{"model_type":"gemma4_assistant","architectures":["Gemma4AssistantForCausalLM"]}"#), "assistant");
        assert_eq!(with_cfg("e", r#"{"architectures":["DSparkModel"]}"#), "dspark");
        assert_eq!(with_cfg("f", r#"{"architectures":["SomethingElse"]}"#), "draft");
        assert_eq!(with_cfg("my-dflash2-drafter", ""), "dflash2");
    }
}

/// The empty folder a plan-only run gets as $CALIB when no calibration data is set.
fn plan_calib_placeholder() -> PathBuf {
    dirs().data.join("calib-plan-placeholder")
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '-' }).collect()
}

impl App {
    fn conv_out_root(&self) -> PathBuf {
        PathBuf::from(gs(self.settings.g(), "radiance_models_dir", ""))
    }

    pub fn conv_ui(&self) -> Value {
        let c = &self.conv;
        let mut names = vec![NO_RECIPE.to_string()];
        names.extend(c.recipes.iter().map(|r| r.name.clone()));
        // what the dropdown shows: a recipe that needs calibration Hessians says so in the list
        let mut labels = vec![NO_RECIPE.to_string()];
        labels.extend(c.recipes.iter().map(|r| {
            if recipe_needs_calib(&r.text) { format!("{}   ⚠ needs Hessians (CALIB)", r.name) } else { r.name.clone() }
        }));
        json!({
            "visible": c.visible, "sources": c.sources, "recipes": names, "recipeLabels": labels, "src": c.src, "recipe": c.recipe,
            "out": c.out, "sets": c.sets, "draft": c.draft, "maxCtx": c.max_ctx, "extra": c.extra, "env": c.env,
            "planOnly": c.plan_only, "info": c.info, "about": c.about, "status": c.status, "running": c.proc.is_some(),
            "outRoot": self.conv_out_root().display().to_string(),
            "setNote": c.set_note, "hint": c.hint, "fixSets": c.fix_sets.clone().unwrap_or_default(),
            "gpus": self.conv_gpu_choices(), "gpu": c.gpu,
            "recipesDir": gs(self.settings.g(), "radiance_recipes_dir", ""),
            "needsCalib": self.conv_needs_calib(), "calib": c.calib, "calibStatus": self.conv_calib_status().1,
            "calibOk": self.conv_calib_status().0, "calibWhy": self.conv_calib_why(),
            "cal": {
                "shown": c.cal.shown, "src": c.cal.src, "corpus": c.cal.corpus, "out": c.cal.out, "gpu": c.cal.gpu,
                "gpus": self.cal_gpu_choices(), "maxLen": c.cal.max_len, "keepRaw": c.cal.keep_raw,
                "running": c.cal.busy(), "status": c.cal.status, "info": self.cal_info(),
            },
        })
    }

    /// The selected recipe reads `$CALIB` (GPTQ against calibration Hessians).
    pub fn conv_needs_calib(&self) -> bool {
        self.conv.recipes.iter().find(|r| r.name == self.conv.recipe).is_some_and(|r| recipe_needs_calib(&r.text))
    }

    /// The calibration folder rad-convert will read: Environment's CALIB=, else the field.
    fn conv_calib_dir(&self) -> Option<PathBuf> {
        let from_env = shlex_split(&self.conv.env).unwrap_or_default().into_iter().find_map(|p| p.strip_prefix("CALIB=").map(String::from));
        let d = from_env.unwrap_or_else(|| self.conv.calib.clone());
        let d = d.trim();
        if d.is_empty() { None } else { Some(expanduser(d)) }
    }

    /// (usable, what the calibration folder holds).
    fn conv_calib_status(&self) -> (bool, String) {
        let Some(d) = self.conv_calib_dir() else { return (false, "no calibration folder set".into()) };
        if !d.is_dir() {
            return (false, format!("{} does not exist", d.display()));
        }
        match super::calib::grams_in(&d) {
            (0, 0) => (false, "no Hessian files (gram.r*.bin) in that folder".into()),
            (0, r) => (false, format!("{r} raw calibration files but no gram.r*.bin - point it at the grams/ folder of a finished run")),
            (g, _) => (true, format!("✓ {g} Hessian files")),
        }
    }

    /// Why Create Hessians can't run for the Convert source (empty: it can).
    fn conv_calib_why(&self) -> String {
        if self.conv.src.is_empty() {
            return String::new();
        }
        super::calib::unsupported(&PathBuf::from(&self.conv.src)).unwrap_or_default()
    }

    /// The alert when a GPTQ recipe is picked without calibration data.
    fn conv_calib_alert(&mut self) {
        if !self.conv_needs_calib() || self.conv_calib_status().0 {
            return;
        }
        let why = self.conv_calib_why();
        let how = if why.is_empty() {
            "LLM Duo can record them: press \"Create Hessians…\" in the Convert window. That serves the bf16 checkpoint with Radiance's calibration mode over a text corpus (an hour or more, and the bf16 weights must fit your cards), then writes the files rad-convert reads."
                .to_string()
        } else {
            format!("LLM Duo can't record them for this checkpoint: {why}.")
        };
        self.msg(
            "warning",
            &format!(
                "The recipe \"{}\" is fitted by GPTQ against calibration Hessians: the statistics of every layer's input over a text corpus (CALIB).\n\n\
                 Without them rad-convert quantises those weights uncalibrated, which gives a worse model than the recipe's published numbers.\n\n\
                 If you already have them, set \"Calibration data\" to their folder.\n\n{how}\n\n\
                 Without calibration data, a round-to-nearest recipe (e.g. q38-27b-iq8r-rtn) needs none.",
                self.conv.recipe
            ),
        );
    }

    pub fn conv_log_pub(&mut self, line: &str) {
        self.conv_log(line);
    }

    pub fn conv_show(&mut self) {
        let was = self.conv.visible;
        self.conv.visible = true;
        if self.conv.proc.is_none() {
            self.conv_refresh();
        }
        if !was {
            self.conv_calib_alert();
        }
    }

    /// Checkpoint folders from the vLLM and Radiance model folders, and the recipes.
    pub fn conv_refresh(&mut self) {
        let g = self.settings.g().clone();
        let mut src: Vec<String> = vec![];
        for key in ["vllm_models_dir", "radiance_models_dir"] {
            let root = gs(&g, key, "");
            if root.trim().is_empty() {
                continue;
            }
            for rel in super::vllm::scan_vllm_models(&root) {
                let full = PathBuf::from(&root).join(&rel).to_string_lossy().into_owned();
                if !src.contains(&full) {
                    src.push(full);
                }
            }
            for laya in scan_laya(&root) {
                // the folder itself, not the encoder inside it
                src.retain(|s| !std::path::Path::new(s).starts_with(&laya));
                src.push(laya.to_string_lossy().into_owned());
            }
        }
        if !self.conv.src.is_empty() && !src.contains(&self.conv.src) {
            src.insert(0, self.conv.src.clone()); // a folder picked by hand stays in the list
        }
        self.conv.sources = src;
        self.conv.recipes = list_recipes(&gs(&g, "radiance_recipes_dir", ""));
        if self.conv.recipe != NO_RECIPE && !self.conv.recipes.iter().any(|r| r.name == self.conv.recipe) {
            self.conv.recipe = NO_RECIPE.into();
        }
        self.conv_update_info();
    }

    pub fn conv_set_src(&mut self, v: &str) {
        self.conv.src = v.trim().trim_end_matches('/').to_string();
        if !self.conv.src.is_empty() && !self.conv.sources.contains(&self.conv.src) {
            self.conv.sources.insert(0, self.conv.src.clone());
        }
        self.conv.hint.clear();
        self.conv.fix_sets = None;
        self.conv_auto_sets();
        self.conv_default_out();
        self.conv_update_info();
    }

    pub fn conv_set_recipe(&mut self, v: &str) {
        self.conv.recipe = v.to_string();
        self.conv.hint.clear();
        self.conv.fix_sets = None;
        self.conv_auto_sets();
        self.conv_default_out();
        self.conv.calib_confirmed = false;
        self.conv_update_info();
        self.conv_calib_alert();
    }

    /// What the Metadata (--set) box should hold for this checkpoint and recipe, and why:
    /// the recipe header's own keys first; else, for a bf16 checkpoint whose trunk the recipe
    /// quantises, the keys that select the family's quantised serving plugin.
    fn conv_suggest_sets(&self) -> (Option<Vec<String>>, String) {
        let (sets, note) = self.conv_recipe_sets();
        let Some(kind) = decision_kind(&PathBuf::from(&self.conv.src)) else { return (sets, note) };
        let mut all = sets.unwrap_or_default();
        if !all.iter().any(|s| s.starts_with("decision.type=")) {
            all.insert(0, format!("decision.type={kind}"));
        }
        let what = format!("a {kind} decision checkpoint: served on /v1/systemone");
        (Some(all), if note.is_empty() { what } else { format!("{what}; {note}") })
    }

    fn conv_recipe_sets(&self) -> (Option<Vec<String>>, String) {
        let Some(r) = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe) else { return (None, String::new()) };
        let header = recipe_sets(&r.about);
        if !header.is_empty() {
            return (Some(header), "Metadata filled in from the recipe's header".into());
        }
        let rk = recipe_kind(&r.text);
        let src = PathBuf::from(&self.conv.src);
        if !rk.quantises_trunk() || !src.is_dir() {
            return (None, String::new());
        }
        let ck = checkpoint_kind(&src);
        if ck.quantised != Some(false) {
            return (None, String::new()); // a quantised release names its own format in its config
        }
        match known_sets(ck.family(), &rk) {
            Some((sets, why)) => (
                Some(sets),
                format!(
                    "Metadata auto-filled: the checkpoint is bf16 but this recipe writes {} weights - these keys {why}",
                    rk.tag(&r.name).to_uppercase()
                ),
            ),
            None => (
                None,
                format!(
                    "The checkpoint is bf16 and this recipe quantises it{} - the recipe may need quantization_config --set keys (Check plan will tell)",
                    if ck.model_type.is_empty() { String::new() } else { format!(" ({})", ck.model_type) }
                ),
            ),
        }
    }

    /// Fill the Metadata box unless the user has typed something of their own.
    fn conv_auto_sets(&mut self) {
        if self.conv.proc.is_some() {
            return;
        }
        let (sets, note) = self.conv_suggest_sets();
        let mine = self.conv.sets.trim().is_empty() || self.conv.sets == self.conv.sets_auto;
        if mine {
            let t = sets.as_deref().map(sets_text).unwrap_or_default();
            self.conv.sets = t.clone();
            self.conv.sets_auto = t;
            self.conv.set_note = note;
        } else {
            self.conv.set_note = match sets {
                Some(v) if sets_text(&v) != self.conv.sets => format!("{note} - kept what you typed instead: {}", sets_text(&v)),
                _ => String::new(),
            };
        }
    }

    fn conv_default_out(&mut self) {
        if self.conv.proc.is_some() || self.conv.src.is_empty() {
            return;
        }
        if !self.conv.out.trim().is_empty() && self.conv.out != self.conv.out_auto {
            return; // the user named it
        }
        let name = PathBuf::from(&self.conv.src).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut tag = String::new();
        if let Some(r) = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe) {
            // the format the trunk ends up in, not the recipe's name (an "nvfp4" recipe writes MXFP4)
            let t = recipe_kind(&r.text).tag(&r.name);
            tag = format!("-{}", if t.is_empty() { r.name.clone() } else { t });
        }
        if !self.conv.draft.trim().is_empty() {
            // what the merged drafter is (dflash2, dflash, assistant, …), not a blanket "-mtp"
            tag += &format!("-{}", drafter_tag(&expanduser(self.conv.draft.trim())));
        }
        self.conv.out = sanitize(&format!("{name}{tag}")) + ".rad";
        self.conv.out_auto = self.conv.out.clone();
    }

    /// "(auto)" = the card a Radiance pane uses; "all"; or one card.
    fn conv_gpu_choices(&self) -> Vec<String> {
        let mut v = vec!["(auto)".to_string(), "all cards".into()];
        v.extend(self.devices.iter().map(|d| format!("{}: {}", d.id, d.name)));
        v
    }

    fn conv_gpu_env(&self) -> Option<String> {
        let pick = |id: &str| self.devices.iter().find(|d| d.id == id).map(|d| d.index.to_string());
        match self.conv.gpu.as_str() {
            "all cards" => None,
            "(auto)" | "" => {
                // the card a Radiance pane runs on, else the largest card
                let rad = self.panes.iter().find(|p| p.is_rad()).and_then(|p| p.device.clone());
                let d = rad.or_else(|| self.devices.iter().max_by_key(|d| d.total_mib).cloned())?;
                Some(d.index.to_string())
            }
            other => pick(other.split(':').next().unwrap_or("")),
        }
    }

    fn conv_target(&self) -> Option<PathBuf> {
        let o = self.conv.out.trim();
        if o.is_empty() {
            return None;
        }
        let p = expanduser(o);
        let p = if p.is_absolute() { p } else { self.conv_out_root().join(o) };
        Some(if p.extension().is_none() { p.with_extension("rad") } else { p })
    }

    pub fn conv_update_info(&mut self) {
        let mut parts = vec![];
        let src = PathBuf::from(&self.conv.src);
        if !self.conv.src.is_empty() {
            if src.is_dir() || src.is_file() {
                let sz = if src.is_dir() { dir_size(&src) } else { std::fs::metadata(&src).map(|m| m.len() as f64).unwrap_or(0.0) };
                parts.push(format!("source {:.1} GiB", sz / GIB));
                if src.is_dir() && !src.join("config.json").is_file() && decision_kind(&src).is_none() {
                    parts.push("⚠ no config.json in that folder".into());
                }
            } else {
                parts.push("⚠ the source does not exist".into());
            }
        }
        if let Some(t) = self.conv_target() {
            if let Some(f) = t.parent().and_then(super::quantize::disk_free) {
                parts.push(format!("free on the output drive {:.0} GiB", f / GIB));
            }
            if t.exists() && !self.conv.plan_only {
                parts.push("⚠ the output file exists - it will be replaced".into());
            }
            parts.push(format!("→ {}", t.display()));
        }
        if self.conv.recipe == NO_RECIPE {
            parts.push("no recipe: bf16 / f16 / block-fp8 weights are kept as they are".into());
        }
        let recipe = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe);
        if recipe.is_some_and(|r| recipe_needs_calib(&r.text)) && !self.conv_calib_status().0 {
            parts.push("⚠ this recipe needs calibration Hessians: set Calibration data, or Create Hessians…".into());
        }
        if recipe.is_some_and(|r| recipe_needs_repo_cwd(&r.text) && repo_of_recipe(&r.path).is_none()) {
            parts.push("⚠ this recipe reads data/codebooks/… and that folder isn't beside the recipes folder".into());
        }
        if decision_kind(&src) == Some("clef") && !self.conv_has_arch() {
            parts.push("clef: --arch clef and its joint_head.safetensors are added".into());
        }
        self.conv.info = parts.join("   •   ");
        self.conv.about = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe).map(|r| r.about.clone()).unwrap_or_default();
    }

    /// Extra args name the architecture themselves: LLM Duo adds none of its own.
    fn conv_has_arch(&self) -> bool {
        shlex_split(&self.conv.extra).unwrap_or_default().iter().any(|a| a == "--arch")
    }

    /// Where a recipe's repository-relative data files can be opened from.
    fn conv_cwd(&self) -> Option<PathBuf> {
        let r = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe)?;
        if recipe_needs_repo_cwd(&r.text) { repo_of_recipe(&r.path) } else { None }
    }

    fn conv_command(&self) -> Result<(Vec<String>, Vec<(String, String)>), String> {
        let g = self.settings.g();
        let bin = gs(g, "radiance_bin", "");
        let tool = tool_path(&bin, "rad-convert");
        let c = &self.conv;
        if c.src.trim().is_empty() {
            return Err("Pick a source checkpoint folder.".into());
        }
        let out = if c.plan_only { PathBuf::from("/dev/null") } else { self.conv_target().ok_or("Name the output file.")? };
        // absolute, because a recipe with repository-relative files runs from the repository
        let abs = |p: PathBuf| std::path::absolute(&p).unwrap_or(p).to_string_lossy().into_owned();
        let mut cmd = vec![tool.to_string_lossy().into_owned(), abs(expanduser(c.src.trim())), "-o".into(), abs(out)];
        if c.recipe != NO_RECIPE {
            let r = c.recipes.iter().find(|r| r.name == c.recipe).ok_or("That recipe is no longer in the recipes folder.")?;
            cmd.extend(["--recipe".into(), r.path.to_string_lossy().into_owned()]);
        }
        let home = gs(g, "radiance_home", "");
        if !home.trim().is_empty() {
            cmd.extend(["--home".into(), expanduser(home.trim()).to_string_lossy().into_owned()]);
        }
        for kv in shlex_split(&c.sets)? {
            if !kv.contains('=') {
                return Err(format!("'{kv}' in Metadata (--set) is not KEY=VALUE"));
            }
            cmd.extend(["--set".into(), kv]);
        }
        if !c.draft.trim().is_empty() {
            cmd.extend(["--draft-model".into(), abs(expanduser(c.draft.trim()))]);
        }
        if !c.max_ctx.trim().is_empty() {
            let n: i64 = c.max_ctx.trim().replace(['_', ','], "").parse().map_err(|_| "Max context must be a number.")?;
            cmd.extend(["--max-ctx".into(), n.to_string()]);
        }
        if c.plan_only {
            cmd.extend(["--plan-only".into(), "-v".into()]);
        }
        if decision_kind(&PathBuf::from(&c.src)) == Some("clef") && !self.conv_has_arch() {
            cmd.extend(["--arch".into(), "clef".into()]);
            let head = PathBuf::from(&c.src).join("joint_head.safetensors");
            if head.is_file() {
                cmd.extend(["--add".into(), abs(head)]);
            }
        }
        cmd.extend(shlex_split(&c.extra)?);
        let mut env: Vec<(String, String)> = std::env::vars().collect();
        let mut set = |k: String, v: String| {
            env.retain(|e| e.0 != k);
            env.push((k, v));
        };
        if !home.trim().is_empty() {
            set("RADIANCE_HOME".into(), expanduser(home.trim()).to_string_lossy().into_owned());
        }
        if let Some(i) = self.conv_gpu_env() {
            set("HIP_VISIBLE_DEVICES".into(), i);
        }
        if !c.calib.trim().is_empty() {
            set("CALIB".into(), abs(expanduser(c.calib.trim())));
        }
        for part in shlex_split(&c.env)? {
            match part.split_once('=') {
                Some((k, v)) if !k.is_empty() => set(k.into(), v.into()),
                _ => return Err(format!("'{part}' in Environment is not KEY=VALUE")),
            }
        }
        // a plan reads no Hessians, but rad-convert refuses a recipe whose $CALIB is unset:
        // give it an empty folder so the plan can be checked before any calibration exists
        if c.plan_only && c.calib_confirmed && self.conv_needs_calib() && !env.iter().any(|e| e.0 == "CALIB" && !e.1.trim().is_empty()) {
            let ph = plan_calib_placeholder();
            let _ = std::fs::create_dir_all(&ph);
            env.push(("CALIB".into(), ph.to_string_lossy().into_owned()));
        }
        Ok((cmd, env))
    }

    pub fn conv_start(&mut self, replace_ok: bool) {
        if self.conv.proc.is_some() {
            return;
        }
        if self.conv.cal.busy() {
            self.msg("info", "Hessians are being recorded - wait for that run (or stop it) before converting.");
            return;
        }
        if self.conv_needs_calib() && !self.conv_calib_status().0 && !self.conv.calib_confirmed {
            let status = self.conv_calib_status().1;
            let recipe = self.conv.recipe.clone();
            if self.conv.plan_only {
                self.ask(
                    &format!(
                        "HESSIANS NEEDED: the recipe \"{recipe}\" is fitted by GPTQ against calibration Hessians, and none are set ({status}).\n\n\
                         A plan check reads no Hessians, so it can run with an empty stand-in folder as CALIB. \
                         Converting for real will still need them: set Calibration data, or Create Hessians….\n\n\
                         Check the plan with an empty stand-in for CALIB?"
                    ),
                    "yesno",
                    Pending::ConvNoCalib(replace_ok),
                );
            } else if self.conv_calib_dir().is_none() {
                self.msg(
                    "warning",
                    &format!(
                        "HESSIANS NEEDED: the recipe \"{recipe}\" is fitted by GPTQ against calibration Hessians, and no calibration data is set.\n\n\
                         rad-convert cannot run this recipe without them. Set Calibration data to a folder of gram.r*.bin files, \
                         or press Create Hessians… to record them (dense Qwen3.5/3.8 only). A round-to-nearest recipe needs none."
                    ),
                );
            } else {
                self.ask(
                    &format!(
                        "HESSIANS MISSING: the recipe \"{recipe}\" is fitted by GPTQ against calibration Hessians, but the Calibration data folder doesn't hold them:\n{status}\n\n\
                         Converting anyway quantises every weight without a Hessian UNCALIBRATED (a worse model than the recipe's numbers).\n\nConvert anyway?"
                    ),
                    "yesno",
                    Pending::ConvNoCalib(replace_ok),
                );
            }
            return;
        }
        let (cmd, env) = match self.conv_command() {
            Ok(x) => x,
            Err(e) => {
                self.msg("info", &e);
                return;
            }
        };
        let target = if self.conv.plan_only { None } else { self.conv_target() };
        if let Some(t) = &target {
            if t.exists() && !replace_ok {
                self.ask(&format!("{}\nalready exists. Replace it?", t.display()), "yesno", Pending::ConvReplace);
                return;
            }
            if let Some(d) = t.parent() {
                let _ = std::fs::create_dir_all(d);
            }
        }
        self.conv.out_existed = target.as_ref().map(|t| t.exists()).unwrap_or(false);
        self.conv.hint.clear();
        self.conv.fix_sets = None;
        self.push_event(json!({"type": "logClear", "side": "convert"}));
        self.conv.lines.clear();
        let hip = env.iter().find(|e| e.0 == "HIP_VISIBLE_DEVICES").map(|e| format!("HIP_VISIBLE_DEVICES={} ", e.1)).unwrap_or_default();
        let ph = plan_calib_placeholder().to_string_lossy().into_owned();
        let calib = env.iter().find(|e| e.0 == "CALIB").map(|e| format!("CALIB={} ", shlex_quote(&e.1))).unwrap_or_default();
        if self.conv.plan_only && env.iter().any(|e| e.0 == "CALIB" && e.1 == ph) {
            self.conv_log("[LLM Duo] HESSIANS NEEDED: this recipe needs calibration Hessians and none are set. As you confirmed, this plan check runs with CALIB = an empty stand-in folder; a real convert needs the Hessians");
        }
        let cwd = self.conv_cwd();
        let cd = cwd.as_ref().map(|d| format!("cd {} && ", shlex_quote(&d.to_string_lossy()))).unwrap_or_default();
        self.conv_log(&format!("$ {cd}{calib}{hip}{}", shlex_join(&cmd)));
        while self.conv.rx.try_recv().is_ok() {}
        match proc::spawn_in(&cmd, &env, false, cwd.as_deref(), self.conv.tx.clone()) {
            Ok(pr) => self.conv.proc = Some(pr),
            Err(e) => {
                self.conv_log(&format!("Failed to start rad-convert: {e}\n(set the Radiance binary in Settings: rad-convert is looked for beside it)"));
                return;
            }
        }
        self.conv.calib_confirmed = false;
        self.conv.out_path = target;
        self.conv.t0 = Some(Instant::now());
        self.conv.status = if self.conv.plan_only { "Planning…".into() } else { "Converting…".into() };
    }

    fn conv_log(&mut self, line: &str) {
        if self.conv.lines.len() >= 2000 {
            self.conv.lines.pop_front();
        }
        self.conv.lines.push_back(line.to_string());
        self.push_event(json!({"type": "log", "side": "convert", "text": line}));
    }

    /// Confirmed: start without calibration data.
    pub fn conv_start_uncalibrated(&mut self, replace_ok: bool) {
        self.conv.calib_confirmed = true;
        self.conv_start(replace_ok);
    }

    pub fn conv_stop(&mut self) {
        let Some(pr) = &self.conv.proc else { return };
        self.conv.user_stopped = true;
        proc::term_group(pr.pid);
        self.conv.status = "Stopping…".into();
    }

    pub fn conv_close(&mut self) {
        if self.conv.proc.is_some() || self.conv.cal.busy() {
            self.ask(
                "A conversion is running.\n\nYes: stop it\nNo: keep it running in the background (reopen with Convert… in the top bar)",
                "yesnocancel",
                Pending::ConvCloseRunning,
            );
            return;
        }
        self.conv.visible = false;
    }

    pub fn conv_poll(&mut self) {
        for _ in 0..500 {
            let Ok(msg) = self.conv.rx.try_recv() else { break };
            match msg {
                ProcMsg::Line(l) => {
                    let el = self.conv.t0.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                    let short: String = l.trim().chars().take(110).collect();
                    self.conv.status = format!("{:.1} min  •  {short}", el / 60.0);
                    self.conv_log(&l);
                }
                ProcMsg::Exit(id, rc) => self.conv_on_exit(id, rc),
            }
        }
    }

    fn conv_on_exit(&mut self, id: u64, rc: i32) {
        if self.conv.proc.as_ref().map(|p| p.id) != Some(id) {
            return;
        }
        self.conv.proc = None;
        let stopped = std::mem::take(&mut self.conv.user_stopped);
        let mins = self.conv.t0.map(|t| t.elapsed().as_secs_f64() / 60.0).unwrap_or(0.0);
        // a bf16 checkpoint run without the keys that select the quantised plugin
        let lines: Vec<String> = self.conv.lines.iter().cloned().collect();
        if !stopped && bf16_plugin_mismatch(&lines) {
            let rk = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe).map(|r| recipe_kind(&r.text)).unwrap_or_default();
            let ck = checkpoint_kind(&PathBuf::from(&self.conv.src));
            let fix = known_sets(ck.family(), &rk).map(|(v, _)| sets_text(&v));
            self.conv.hint = "The checkpoint is bf16 but the recipe writes quantised weights, so Radiance picked its bf16 plugin - \
                              add the quantization_config --set values (Metadata)."
                .to_string()
                + match &fix {
                    Some(_) => " The known values for this model family are ready to apply.",
                    None => " LLM Duo doesn't know the values for this family: see the recipe's header or Radiance's docs (GUIDE §3.2).",
                };
            self.conv.fix_sets = fix;
        }
        if rc == 0 {
            if self.conv.plan_only {
                self.conv.status = format!("Plan done in {mins:.1} min - nothing was written (see the log)");
                return;
            }
            let name = self.conv.out_path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            self.conv.status = format!("Done in {mins:.1} min - '{name}' is now in the Radiance model lists");
            self.rescan_models();
            self.set_top_status(format!("Convert: {name} is ready"));
            self.conv_update_info();
            return;
        }
        self.conv.status = if stopped { "Stopped".into() } else { format!("Failed (exit code {rc}) - see the log") };
        self.set_top_status(format!("Convert: {}", if stopped { "stopped" } else { "failed" }));
        if let Some(t) = self.conv.out_path.clone() {
            if t.exists() {
                self.ask(&format!("Delete the unfinished output?\n{}", t.display()), "yesno", Pending::ConvDeleteUnfinished(t));
            }
        }
    }

    pub fn conv_call(&mut self, action: &str, v: &str) {
        match action {
            "show" => self.conv_show(),
            "refresh" => self.conv_refresh(),
            "setSrc" => self.conv_set_src(v),
            "setRecipe" => self.conv_set_recipe(v),
            "setOut" => {
                self.conv.out = v.to_string();
                self.conv_update_info();
            }
            "setSets" => {
                self.conv.sets = v.to_string();
                if v.trim().is_empty() {
                    self.conv.sets_auto.clear();
                    self.conv.set_note.clear();
                }
                if self.conv.sets != self.conv.sets_auto {
                    self.conv.set_note.clear();
                }
            }
            "applyFix" => {
                if let Some(f) = self.conv.fix_sets.take() {
                    self.conv.sets = f.clone();
                    self.conv.sets_auto = f;
                    self.conv.set_note = "Metadata set to the known values for this model family - run Check plan again".into();
                    self.conv.hint.clear();
                }
            }
            "setGpu" => self.conv.gpu = v.to_string(),
            "setDraft" => {
                self.conv.draft = v.to_string();
                self.conv_default_out();
                self.conv_update_info();
            }
            "setMaxCtx" => self.conv.max_ctx = v.to_string(),
            "setExtra" => self.conv.extra = v.to_string(),
            "setEnv" => {
                self.conv.env = v.to_string();
                self.conv_update_info();
            }
            "setPlanOnly" => {
                self.conv.plan_only = v == "true";
                self.conv_update_info();
            }
            "start" => self.conv_start(false),
            "setCalib" => {
                self.conv.calib = v.to_string();
                self.conv_update_info();
            }
            "calShow" => self.cal_show(),
            "calHide" => self.conv.cal.shown = false,
            "calSetSrc" => self.cal_set_src(v),
            "calSetCorpus" => self.conv.cal.corpus = v.to_string(),
            "calSetOut" => self.conv.cal.out = v.to_string(),
            "calSetGpu" => self.conv.cal.gpu = v.to_string(),
            "calSetMaxLen" => self.conv.cal.max_len = v.to_string(),
            "calSetKeepRaw" => self.conv.cal.keep_raw = v == "true",
            "calStart" => self.cal_start(),
            "calStop" => self.cal_stop(),
            "stop" => self.conv_stop(),
            "hide" => self.conv.visible = false,
            "close" => self.conv_close(),
            _ => {}
        }
    }
}
