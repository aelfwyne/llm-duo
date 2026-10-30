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
use super::radiance::{bf16_plugin_mismatch, checkpoint_kind, known_sets, list_recipes, recipe_kind, recipe_sets, sets_text, tool_path, Recipe};
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

fn dir_size(d: &std::path::Path) -> f64 {
    let mut t = 0.0;
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
        json!({
            "visible": c.visible, "sources": c.sources, "recipes": names, "src": c.src, "recipe": c.recipe,
            "out": c.out, "sets": c.sets, "draft": c.draft, "maxCtx": c.max_ctx, "extra": c.extra, "env": c.env,
            "planOnly": c.plan_only, "info": c.info, "about": c.about, "status": c.status, "running": c.proc.is_some(),
            "outRoot": self.conv_out_root().display().to_string(),
            "setNote": c.set_note, "hint": c.hint, "fixSets": c.fix_sets.clone().unwrap_or_default(),
            "gpus": self.conv_gpu_choices(), "gpu": c.gpu,
            "recipesDir": gs(self.settings.g(), "radiance_recipes_dir", ""),
        })
    }

    pub fn conv_show(&mut self) {
        self.conv.visible = true;
        if self.conv.proc.is_none() {
            self.conv_refresh();
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
        self.conv_update_info();
    }

    /// What the Metadata (--set) box should hold for this checkpoint and recipe, and why:
    /// the recipe header's own keys first; else, for a bf16 checkpoint whose trunk the recipe
    /// quantises, the keys that select the family's quantised serving plugin.
    fn conv_suggest_sets(&self) -> (Option<Vec<String>>, String) {
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
            tag += "-mtp";
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
                if src.is_dir() && !src.join("config.json").is_file() {
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
        if self.conv.env.contains("CALIB") || self.conv.recipes.iter().any(|r| r.name == self.conv.recipe && r.about.contains("$CALIB")) {
            parts.push("this recipe reads calibration data: set CALIB=<dir> in Environment".into());
        }
        self.conv.info = parts.join("   •   ");
        self.conv.about = self.conv.recipes.iter().find(|r| r.name == self.conv.recipe).map(|r| r.about.clone()).unwrap_or_default();
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
        let mut cmd = vec![tool.to_string_lossy().into_owned(), c.src.clone(), "-o".into(), out.to_string_lossy().into_owned()];
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
            cmd.extend(["--draft-model".into(), expanduser(c.draft.trim()).to_string_lossy().into_owned()]);
        }
        if !c.max_ctx.trim().is_empty() {
            let n: i64 = c.max_ctx.trim().replace(['_', ','], "").parse().map_err(|_| "Max context must be a number.")?;
            cmd.extend(["--max-ctx".into(), n.to_string()]);
        }
        if c.plan_only {
            cmd.extend(["--plan-only".into(), "-v".into()]);
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
        for part in shlex_split(&c.env)? {
            match part.split_once('=') {
                Some((k, v)) if !k.is_empty() => set(k.into(), v.into()),
                _ => return Err(format!("'{part}' in Environment is not KEY=VALUE")),
            }
        }
        Ok((cmd, env))
    }

    pub fn conv_start(&mut self, replace_ok: bool) {
        if self.conv.proc.is_some() {
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
        self.conv_log(&format!("$ {hip}{}", shlex_join(&cmd)));
        while self.conv.rx.try_recv().is_ok() {}
        match proc::spawn(&cmd, &env, false, self.conv.tx.clone()) {
            Ok(pr) => self.conv.proc = Some(pr),
            Err(e) => {
                self.conv_log(&format!("Failed to start rad-convert: {e}\n(set the Radiance binary in Settings: rad-convert is looked for beside it)"));
                return;
            }
        }
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

    pub fn conv_stop(&mut self) {
        let Some(pr) = &self.conv.proc else { return };
        self.conv.user_stopped = true;
        proc::term_group(pr.pid);
        self.conv.status = "Stopping…".into();
    }

    pub fn conv_close(&mut self) {
        if self.conv.proc.is_some() {
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
            "stop" => self.conv_stop(),
            "hide" => self.conv.visible = false,
            "close" => self.conv_close(),
            _ => {}
        }
    }
}
