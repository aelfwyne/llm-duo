// Generates and compiles the C++ side of the cxx-qt bridge, bundles the QML files into the
// binary as the QML module `io.github.llmduo`, and links Qt (found through qmake6/QMAKE).
use cxx_qt_build::{CxxQtBuilder, QmlModule};

fn main() {
    CxxQtBuilder::new_qml_module(QmlModule::new("io.github.llmduo").qml_files([
        "qml/Main.qml",
        "qml/PaneView.qml",
        "qml/SettingsWindow.qml",
        "qml/QuantizeWindow.qml",
        "qml/RadConvertWindow.qml",
        "qml/ReportWindow.qml",
        "qml/PathField.qml",
        "qml/BackendBox.qml",
        "qml/HelpTip.qml",
        "qml/OptLabel.qml",
        "qml/OptSpin.qml",
        "qml/OptCombo.qml",
        "qml/OptCheck.qml",
        "qml/OptText.qml",
        "qml/DashPalette.qml",
        "qml/ChartCard.qml",
        "qml/LineChart.qml",
        "qml/BarChart.qml",
        "qml/ScatterChart.qml",
        "qml/StatTile.qml",
        "qml/Meter.qml",
        "qml/LiveDashboard.qml",
        "qml/HistoryDashboard.qml",
    ]))
    // Qt modules used besides Core (Gui/Qml come with cxx-qt-lib's qt_full feature)
    .qt_module("Quick")
    .qt_module("QuickControls2")
    .qt_module("Widgets")
    .files(["src/bridge.rs"])
    // QApplication + QML engine + clipboard helper
    .cpp_file("cpp/launcher.cpp")
    .build();

    // build stamp shown in the window title, so it is obvious which build is running
    println!("cargo::rerun-if-env-changed=LLM_DUO_BUILD");
    let stamp = std::env::var("LLM_DUO_BUILD").unwrap_or_else(|_| "dev".into());
    println!("cargo::rustc-env=LLM_DUO_BUILD={stamp}");
    println!("cargo::rerun-if-changed=assets/help.json");
    println!("cargo::rerun-if-changed=assets/fp8_dynamic_quant.py");
}
