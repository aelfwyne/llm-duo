//! Everything except the Qt user interface. Plain Rust, testable without Qt.
#![allow(dead_code)] // some fields mirror the Python data structures for completeness

pub mod app;
pub mod bench;
pub mod calib;
pub mod chat_tmpl;
pub mod consts;
pub mod dash;
pub mod devices;
pub mod estimate;
pub mod fleet;
pub mod fleet_app;
pub mod gguf;
pub mod http;
pub mod kobold;
pub mod mtp;
pub mod pane;
pub mod pane_k;
pub mod pane_run;
pub mod pane_r;
pub mod pane_v;
pub mod prefix;
pub mod proc;
pub mod quantize;
pub mod rad_convert;
pub mod radiance;
pub mod settings;
pub mod telemetry;
pub mod util;
pub mod vllm;
