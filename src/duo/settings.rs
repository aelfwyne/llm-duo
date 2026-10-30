//! Persistent settings: global options, per-(GPU, model) options, calibration and stats.
//! File format is identical to the Python app's `llm_duo_settings.json`.

use serde_json::{Map, Value};
use std::path::PathBuf;

use super::consts::{default_global, default_model, KV_TYPES};
use super::util::{dirs, Opts};

pub struct Settings {
    pub data: Map<String, Value>,
    pub path: PathBuf,
}

fn obj(v: Option<&Value>) -> Map<String, Value> {
    v.and_then(|x| x.as_object()).cloned().unwrap_or_default()
}

impl Settings {
    pub fn settings_file() -> PathBuf {
        dirs().config.join("llm_duo_settings.json")
    }

    pub fn load() -> Self {
        let path = Self::settings_file();
        let mut s = Settings { data: Map::new(), path: path.clone() };
        s.data.insert("global".into(), Value::Object(default_global()));
        for k in ["models", "calib", "stats"] {
            s.data.insert(k.into(), Value::Object(Map::new()));
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
            let old = dir.join("llama_duo_settings.json"); // from before the rename
            if !path.exists() && old.exists() {
                let _ = std::fs::copy(&old, &path);
            }
        }
        if path.exists() {
            match std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
                Some(Value::Object(loaded)) => {
                    for k in ["global", "models", "calib", "stats"] {
                        let src = obj(loaded.get(k));
                        let dst = s.data.get_mut(k).unwrap().as_object_mut().unwrap();
                        for (kk, vv) in src {
                            dst.insert(kk, vv);
                        }
                    }
                }
                _ => {
                    let _ = std::fs::rename(&path, path.with_extension("json.bad"));
                }
            }
        }
        s
    }

    pub fn g(&self) -> &Opts {
        self.data["global"].as_object().unwrap()
    }
    pub fn g_mut(&mut self) -> &mut Opts {
        self.data.get_mut("global").unwrap().as_object_mut().unwrap()
    }

    fn sect(&self, k: &str) -> &Opts {
        self.data[k].as_object().unwrap()
    }
    fn sect_mut(&mut self, k: &str) -> &mut Opts {
        self.data.get_mut(k).unwrap().as_object_mut().unwrap()
    }

    pub fn model_key(dev: &str, rel: &str) -> String {
        format!("{dev}|{rel}")
    }

    pub fn has_model(&self, dev: &str, rel: &str) -> bool {
        self.sect("models").contains_key(&Self::model_key(dev, rel))
    }

    /// Saved options merged over the defaults. `defaults` = vLLM defaults (no llama.cpp migrations).
    pub fn get_model(&self, dev: &str, rel: &str, defaults: Option<&Opts>) -> Opts {
        let mut saved = obj(self.sect("models").get(&Self::model_key(dev, rel)));
        if let Some(d) = defaults {
            let mut s = d.clone();
            for (k, v) in saved {
                s.insert(k, v);
            }
            return s;
        }
        let mut s = default_model();
        // migrate old separate K/V settings to the paired setting
        let old_k = saved.remove("ctk");
        saved.remove("ctv");
        if !saved.contains_key("kv") {
            if let Some(k) = old_k {
                if super::util::truthy(Some(&k)) {
                    saved.insert("kv".into(), k);
                }
            }
        }
        for (k, v) in saved {
            s.insert(k, v);
        }
        let kv = s.get("kv").and_then(|v| v.as_str()).unwrap_or("");
        if !KV_TYPES.contains(&kv) {
            s.insert("kv".into(), Value::from("f16"));
        }
        s
    }

    pub fn set_model(&mut self, dev: &str, rel: &str, values: Opts) {
        self.sect_mut("models").insert(Self::model_key(dev, rel), Value::Object(values));
        self.save();
    }

    pub fn get_stats(&self, dev: &str, rel: &str) -> Option<Opts> {
        self.sect("stats").get(&Self::model_key(dev, rel)).and_then(|v| v.as_object()).cloned()
    }
    /// Remove one configuration's running totals.
    pub fn del_stats(&mut self, dev: &str, rel: &str) -> bool {
        let k = Self::model_key(dev, rel);
        let had = self.sect_mut("stats").remove(&k).is_some();
        if had {
            self.save();
        }
        had
    }
    pub fn set_stats(&mut self, dev: &str, rel: &str, v: Opts) {
        self.sect_mut("stats").insert(Self::model_key(dev, rel), Value::Object(v));
        self.save();
    }

    pub fn get_calib(&self, dev: &str, rel: &str) -> Option<Opts> {
        self.sect("calib").get(&Self::model_key(dev, rel)).and_then(|v| v.as_object()).cloned()
    }
    pub fn set_calib(&mut self, dev: &str, rel: &str, v: Opts) {
        self.sect_mut("calib").insert(Self::model_key(dev, rel), Value::Object(v));
        self.save();
    }
    pub fn raw_section(&self, name: &str) -> Opts {
        self.sect(name).clone()
    }
    pub fn calib_all(&self) -> Vec<(String, Value)> {
        self.sect("calib").iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }
    pub fn calib_set_sig(&mut self, key: &str, sig: &str) {
        if let Some(Value::Object(o)) = self.sect_mut("calib").get_mut(key) {
            o.insert("sig".into(), Value::from(sig));
        }
    }

    pub fn save(&self) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = self.path.with_extension("tmp");
        if let Ok(text) = serde_json::to_string_pretty(&Value::Object(self.data.clone())) {
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }
}
