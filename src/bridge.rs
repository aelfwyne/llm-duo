//! The Qt side of LLM Duo: one QML singleton (`Backend`) that forwards everything to the
//! plain-Rust application controller in `crate::duo`.
//!
//! The UI state travels as one JSON document (`uiJson`), re-sent whenever it changes; one-off
//! things (log lines, dialogs, questions) arrive through the `backendEvent` signal. QML calls
//! `poll()` every 100 ms and `tick()` every 1.5 s, like the Tk version's `after()` loops.

#[cxx_qt::bridge]
pub mod qobject {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;
    }

    extern "RustQt" {
        #[qobject]
        #[qml_element]
        #[qml_singleton]
        #[qproperty(QString, ui_json, cxx_name = "uiJson", READ, NOTIFY = ui_changed)]
        type Backend = super::BackendRust;

        #[qsignal]
        #[cxx_name = "uiChanged"]
        fn ui_changed(self: Pin<&mut Backend>);

        /// One-off events as JSON: log lines, dialogs, questions, quit, ...
        #[qsignal]
        #[cxx_name = "backendEvent"]
        fn backend_event(self: Pin<&mut Backend>, json: QString);

        #[qinvokable]
        fn poll(self: Pin<&mut Backend>);

        #[qinvokable]
        fn tick(self: Pin<&mut Backend>);

        /// `target`: "app", "left", "right" or "quant"; `arg`: JSON. Returns JSON.
        #[qinvokable]
        fn call(self: Pin<&mut Backend>, target: &QString, action: &QString, arg: &QString) -> QString;

        /// Tooltip texts (JSON object), fetched once by the UI.
        #[qinvokable]
        #[cxx_name = "helpJson"]
        fn help_json(self: &Backend) -> QString;
    }
}

use core::pin::Pin;
use cxx_qt::CxxQtType;
use cxx_qt_lib::QString;
use serde_json::Value;

use crate::duo::app::App;

pub struct BackendRust {
    ui_json: QString,
    app: App,
    last_ui: String,
}

impl Default for BackendRust {
    fn default() -> Self {
        Self { ui_json: QString::from("{}"), app: App::new(), last_ui: String::new() }
    }
}

extern "C" {
    fn llmduo_set_clipboard(data: *const u8, len: usize);
}

fn set_clipboard(text: &str) {
    unsafe { llmduo_set_clipboard(text.as_ptr(), text.len()) }
}

impl qobject::Backend {
    /// Publish the UI state (if it changed) and queued events. Called at the end of every
    /// entry point, after all state changes, so QML handlers never re-enter a running call.
    fn flush(mut self: Pin<&mut Self>) {
        let (ui, events) = {
            let mut r = self.as_mut().rust_mut();
            let ui = r.app.ui().to_string();
            let events = r.app.take_events();
            (ui, events)
        };
        if ui != self.rust().last_ui {
            {
                let mut r = self.as_mut().rust_mut();
                r.ui_json = QString::from(ui.as_str());
                r.last_ui = ui;
            }
            self.as_mut().ui_changed();
        }
        for e in events {
            if e.get("type").and_then(|t| t.as_str()) == Some("clipboard") {
                set_clipboard(e.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                continue;
            }
            self.as_mut().backend_event(QString::from(e.to_string().as_str()));
        }
    }

    pub fn poll(mut self: Pin<&mut Self>) {
        self.as_mut().rust_mut().app.poll();
        self.flush();
    }

    pub fn tick(mut self: Pin<&mut Self>) {
        self.as_mut().rust_mut().app.tick();
        self.flush();
    }

    pub fn call(mut self: Pin<&mut Self>, target: &QString, action: &QString, arg: &QString) -> QString {
        let arg: Value = serde_json::from_str(&arg.to_string()).unwrap_or(Value::Null);
        let out = self.as_mut().rust_mut().app.call(&target.to_string(), &action.to_string(), &arg);
        self.flush();
        QString::from(out.to_string().as_str())
    }

    pub fn help_json(&self) -> QString {
        QString::from(serde_json::to_string(crate::duo::consts::help()).unwrap_or_default().as_str())
    }
}
