// LLM Duo main window: toolbar + two GPU panes (the right one collapses in Dual GPU mode).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

Kirigami.ApplicationWindow {
    id: root

    readonly property var ui: JSON.parse(Backend.uiJson)
    readonly property var app: ui.app || ({})
    readonly property var panes: ui.panes || [({}), ({})]
    property var help: ({})

    title: (app.title || "LLM Duo") + (app.build ? "  (build " + app.build + ")" : "")
    width: Kirigami.Units.gridUnit * 78
    height: Kirigami.Units.gridUnit * 52
    minimumWidth: Kirigami.Units.gridUnit * 40
    minimumHeight: Kirigami.Units.gridUnit * 30

    pageStack.globalToolBar.style: Kirigami.ApplicationHeaderStyle.None
    pageStack.initialPage: mainPage

    Component.onCompleted: {
        help = JSON.parse(Backend.helpJson())
        Qt.callLater(function () { Backend.call("app", "startup", "{}") })
    }

    // The Tk version polled its queues every 100 ms and refreshed GPU stats every 1.5 s.
    Timer { interval: 100; running: true; repeat: true; onTriggered: Backend.poll() }
    Timer { interval: 1500; running: true; repeat: true; onTriggered: Backend.tick() }

    onClosing: function (close) {
        close.accepted = false
        Qt.callLater(function () { Backend.call("app", "requestClose", "{}") })
    }

    // ------------------------------------------------------------- events
    Connections {
        target: Backend
        // Handled later, never inside the Rust call that emitted it. (Qt.callLater merges
        // repeated calls of one function, so events are queued and drained together.)
        function onBackendEvent(json) {
            root.eventQueue.push(json)
            Qt.callLater(root.drainEvents)
        }
    }

    property var eventQueue: []
    function drainEvents() {
        const q = eventQueue
        eventQueue = []
        for (let i = 0; i < q.length; i++) handleEvent(q[i])
    }

    function handleEvent(json) {
        const e = JSON.parse(json)
        switch (e.type) {
        case "log":
            if (e.side === "quant") quantWindow.appendLog(e.text)
            else if (e.side === "convert") convertWindow.appendLog(e.text)
            else (e.side === "left" ? leftPane : rightPane).appendLog(e.text)
            break
        case "logClear":
            if (e.side === "quant") quantWindow.clearLog()
            else if (e.side === "convert") convertWindow.clearLog()
            else (e.side === "left" ? leftPane : rightPane).clearLog()
            break
        case "msg":
            enqueueDialog({ text: e.text, buttons: "ok", kind: e.kind })
            break
        case "question":
            enqueueDialog({ text: e.text, buttons: e.buttons, id: e.id })
            break
        case "report":
            reportComponent.createObject(root, { title: e.title, text: e.text, logPath: e.logPath || "", logDir: e.logDir || "" })
            break
        case "openSettings":
            openSettings()
            break
        case "settingsDevices":
            settingsWindow.setDevices(e.devices, e.leftDevice, e.rightDevice)
            break
        case "quit":
            Qt.quit()
            break
        }
    }

    // ------------------------------------------------- message / question dialogs
    property var dialogQueue: []
    function enqueueDialog(d) {
        dialogQueue.push(d)
        if (!promptDialog.visible) showNextDialog()
    }
    function showNextDialog() {
        if (dialogQueue.length === 0) return
        const d = dialogQueue.shift()
        promptDialog.qid = d.id !== undefined ? d.id : -1
        promptDialog.buttons = d.buttons
        promptDialog.kind = d.kind || "question"
        promptDialog.text = d.text
        promptDialog.answered = false
        promptDialog.open()
    }
    function answer(a) {
        if (promptDialog.qid >= 0 && !promptDialog.answered) {
            promptDialog.answered = true
            Backend.call("app", "answer", JSON.stringify({ id: promptDialog.qid, answer: a }))
        }
        promptDialog.close()
    }

    Kirigami.Dialog {
        id: promptDialog
        property int qid: -1
        property string buttons: "ok"
        property string kind: "info"
        property string text: ""
        property bool answered: false

        title: kind === "error" ? "LLM Duo - error" : kind === "warning" ? "LLM Duo - warning" : "LLM Duo"
        padding: Kirigami.Units.largeSpacing
        preferredWidth: Kirigami.Units.gridUnit * 36
        showCloseButton: false
        closePolicy: QQC2.Popup.NoAutoClose

        customFooterActions: [
            Kirigami.Action {
                text: "OK"
                icon.name: "dialog-ok"
                visible: promptDialog.buttons === "ok"
                onTriggered: root.answer("ok")
            },
            Kirigami.Action {
                text: "Yes"
                icon.name: "dialog-ok-apply"
                visible: promptDialog.buttons !== "ok"
                onTriggered: root.answer("yes")
            },
            Kirigami.Action {
                text: "No"
                icon.name: "dialog-close"
                visible: promptDialog.buttons !== "ok"
                onTriggered: root.answer("no")
            },
            Kirigami.Action {
                text: "Cancel"
                icon.name: "dialog-cancel"
                visible: promptDialog.buttons === "yesnocancel"
                onTriggered: root.answer("cancel")
            }
        ]

        RowLayout {
            spacing: Kirigami.Units.largeSpacing
            Kirigami.Icon {
                Layout.alignment: Qt.AlignTop
                implicitWidth: Kirigami.Units.iconSizes.large
                implicitHeight: Kirigami.Units.iconSizes.large
                source: promptDialog.kind === "error" ? "dialog-error"
                      : promptDialog.kind === "warning" ? "dialog-warning"
                      : promptDialog.kind === "question" ? "dialog-question" : "dialog-information"
            }
            QQC2.Label {
                Layout.fillWidth: true
                Layout.maximumWidth: Kirigami.Units.gridUnit * 40
                text: promptDialog.text
                textFormat: Text.PlainText
                wrapMode: Text.Wrap
                // tables (Fit's MoE offer) line up in a fixed-width font
                font.family: promptDialog.text.indexOf("   ") >= 0 ? "monospace" : Qt.application.font.family
            }
        }
        onClosed: {
            // closed without a button (Esc): treat like Cancel / No
            if (qid >= 0 && !answered) {
                answered = true
                Backend.call("app", "answer", JSON.stringify({ id: qid, answer: buttons === "yesnocancel" ? "cancel" : "no" }))
            }
            Qt.callLater(root.showNextDialog)
        }
    }

    Component {
        id: reportComponent
        ReportWindow {}
    }

    // ------------------------------------------------------------ windows
    SettingsWindow {
        id: settingsWindow
        help: root.help
    }
    function openSettings() {
        settingsWindow.load(JSON.parse(Backend.call("app", "settingsData", "{}")))
        settingsWindow.show()
        settingsWindow.raise()
        settingsWindow.requestActivate()
    }

    LiveDashboard { id: liveDash }
    HistoryDashboard { id: historyDash }
    function openWindow(w) {
        w.show()
        w.raise()
        w.requestActivate()
    }

    QuantizeWindow {
        id: quantWindow
        help: root.help
        q: root.ui.quant || ({})
    }
    RadConvertWindow {
        id: convertWindow
        help: root.help
        c: root.ui.convert || ({})
    }

    // --------------------------------------------------------------- page
    Kirigami.Page {
        id: mainPage
        padding: 0
        title: "LLM Duo"

        header: QQC2.ToolBar {
            RowLayout {
                anchors.fill: parent
                QQC2.ToolButton {
                    text: "Settings"
                    icon.name: "configure"
                    onClicked: root.openSettings()
                }
                QQC2.ToolButton {
                    text: "Rescan models"
                    icon.name: "view-refresh"
                    onClicked: Backend.call("app", "rescan", "{}")
                }
                QQC2.ToolButton {
                    id: dashBtn
                    text: "Dashboard"
                    icon.name: "office-chart-line"
                    onClicked: root.openWindow(liveDash)
                    HelpTip { text: "Live performance: request speeds, GPU memory, load, temperature and power, with analysis"; shown: dashBtn.hovered }
                }
                QQC2.ToolButton {
                    id: histBtn
                    text: "History"
                    icon.name: "view-history"
                    onClicked: root.openWindow(historyDash)
                    HelpTip { text: "Every configuration you have run: speeds, requests, aborts, crashes"; shown: histBtn.hovered }
                }
                QQC2.ToolButton {
                    id: quantBtn
                    visible: !!root.app.quantVisible
                    text: root.app.quantLabel || "Quantize…"
                    icon.name: "archive-extract"
                    onClicked: Backend.call("quant", "show", "{}")
                    HelpTip { text: "Make a permanent FP8 copy of a 16-bit vLLM model (runs in the background)"; shown: quantBtn.hovered }
                }
                QQC2.ToolButton {
                    id: convBtn
                    visible: !!root.app.convVisible
                    text: root.app.convLabel || "Convert…"
                    icon.name: "run-build"
                    onClicked: Backend.call("convert", "show", "{}")
                    HelpTip { text: "Convert a Hugging Face checkpoint into a Radiance .rad container with one of Radiance's recipes (runs in the background)"; shown: convBtn.hovered }
                }
                QQC2.CheckBox {
                    id: dualBox
                    text: "Dual GPU mode"
                    checked: !!root.app.dualOn
                    enabled: root.app.dualEnabled !== false
                    onToggled: {
                        Backend.call("app", "setDual", JSON.stringify({ on: checked }))
                        dualBox.checked = Qt.binding(function () { return !!root.app.dualOn })
                    }
                    HelpTip { text: root.help["dual_toggle"] || ""; shown: dualBox.hovered }
                }
                QQC2.Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: Kirigami.Units.largeSpacing
                    text: root.app.topStatus || ""
                    color: Kirigami.Theme.disabledTextColor
                    elide: Text.ElideRight
                }
                QQC2.Button {
                    visible: !!root.app.why
                    text: "Why?"
                    onClicked: Backend.call("app", "showWhy", "{}")
                }
                QQC2.Button {
                    visible: (root.app.alerts || 0) > 0
                    text: (root.app.alerts || 0) > 1 ? "View error (" + root.app.alerts + ")" : "View error"
                    icon.name: "dialog-error"
                    palette.buttonText: Kirigami.Theme.negativeTextColor
                    onClicked: Backend.call("app", "showAlerts", "{}")
                }
            }
        }

        QQC2.SplitView {
            anchors.fill: parent
            orientation: Qt.Horizontal

            PaneView {
                id: leftPane
                side: "left"
                pd: root.panes[0] || ({})
                help: root.help
                lists: root.ui.lists || ({})
                QQC2.SplitView.fillWidth: true
                QQC2.SplitView.minimumWidth: Kirigami.Units.gridUnit * 30
            }
            PaneView {
                id: rightPane
                side: "right"
                // Dual GPU mode: the right pane collapses; the left pane shows both cards' monitors
                visible: !root.app.dualOn
                pd: root.panes[1] || ({})
                help: root.help
                lists: root.ui.lists || ({})
                QQC2.SplitView.preferredWidth: root.width / 2
                QQC2.SplitView.minimumWidth: Kirigami.Units.gridUnit * 30
            }
        }
    }
}
