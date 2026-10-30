//! LLM Duo - run one llama-server, KoboldCpp or vLLM server per GPU from a two-pane
//! KDE Plasma (Qt 6 / Kirigami) GUI. Rust port of the Python/Tk original.
//!
//! The application logic lives in `duo` (plain Rust); `bridge` exposes it to QML through
//! cxx-qt; `cpp/launcher.cpp` creates the QApplication so the Plasma style and platform
//! integration apply.

#![recursion_limit = "512"]

mod bridge;
mod duo;

use std::ffi::{c_char, c_int, CString};

extern "C" {
    fn llmduo_run_gui(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

fn main() {
    // Register the QML module (Backend singleton + QML files) with Qt.
    cxx_qt::init_qml_module!("io.github.llmduo");

    let args: Vec<CString> = std::env::args().map(|a| CString::new(a).unwrap_or_default()).collect();
    // QApplication keeps pointers to argv for its whole life: leak them deliberately.
    let mut argv: Vec<*mut c_char> = args.into_iter().map(|a| a.into_raw()).collect();
    argv.push(std::ptr::null_mut());
    let argc = (argv.len() - 1) as c_int;
    let argv = Box::leak(argv.into_boxed_slice()).as_mut_ptr();
    let code = unsafe { llmduo_run_gui(argc, argv) };
    std::process::exit(code);
}
