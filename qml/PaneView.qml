// One GPU pane: header + VRAM monitor, model, load options, estimate, buttons, status, log.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.Pane {
    id: pv
    property string side: "left"
    property var pd: ({})          // this pane's part of Backend.uiJson
    property var help: ({})
    property var lists: ({})

    padding: Kirigami.Units.largeSpacing

    function setVar(key, value) {
        Backend.call(side, "setVar", JSON.stringify({ key: key, value: value }))
    }
    // A number typed into a box only counts once the box loses focus, and clicking a button
    // doesn't always take the focus. Every action first commits what the boxes show, so Load /
    // Fit / Copy command use exactly the values on screen.
    signal flushEdits()
    function act(action, arg) {
        flushEdits()
        Backend.call(side, action, JSON.stringify(arg || {}))
    }
    function h(key) { return help[key] || "" }
    function tc(hex) {
        switch (hex) {
        case "#2e9e44": return Kirigami.Theme.positiveTextColor
        case "#d08a00": return Kirigami.Theme.neutralTextColor
        case "#d0342c": case "#c62828": return Kirigami.Theme.negativeTextColor
        case "#2a7fd4": return Kirigami.Theme.linkColor
        case "#666": case "#888888": return Kirigami.Theme.disabledTextColor
        default: return Kirigami.Theme.textColor
        }
    }

    // ---- log (append-only, trimmed like the Tk version: >2500 lines -> keep 2000) ----
    function appendLog(text) {
        const flick = logFlick
        const atEnd = flick.contentY + flick.height >= flick.contentHeight - 4
        logArea.append(text)
        if (logArea.lineCount > 2500) {
            const t = logArea.text
            let idx = 0
            for (let i = 0; i < logArea.lineCount - 2000 && idx >= 0; i++)
                idx = t.indexOf("\n", idx) + 1
            if (idx > 0) logArea.remove(0, idx)
        }
        if (atEnd) Qt.callLater(function () { flick.contentY = Math.max(0, flick.contentHeight - flick.height) })
    }
    function clearLog() { logArea.clear() }

    readonly property bool running: !!pd.running
    // four option pairs per row when the pane is wide enough, else two
    readonly property bool wide: width >= Kirigami.Units.gridUnit * 34
    function st(k) { return pd.status ? (pd.status[k] || "—") : "—" }
    readonly property var vars: pd.vars || ({})

    QQC2.ScrollView {
        id: paneScroll
        anchors.fill: parent
        contentWidth: availableWidth
        QQC2.ScrollBar.horizontal.policy: QQC2.ScrollBar.AlwaysOff

    ColumnLayout {
        width: paneScroll.availableWidth
        // fill the pane when there is room (the log takes the rest); scroll when there isn't
        height: Math.max(implicitHeight, paneScroll.availableHeight)
        spacing: Kirigami.Units.smallSpacing

        // ------------------------------------------------------------ header
        RowLayout {
            Layout.fillWidth: true
            Kirigami.Heading {
                level: 3
                text: pd.gpuName || ""
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            QQC2.Label {
                id: badge
                text: pd.badge || ""
                color: pv.tc(pd.badgeColor)
                font.bold: true
                HoverHandler { id: badgeHover }
                HelpTip {
                    text: "This pane's backend and build.\nBackend: ⚙ Settings → Backends. Build: in the load options."
                    shown: badgeHover.hovered
                }
            }
        }
        // VRAM / busy monitor (both cards side by side in Dual GPU mode)
        RowLayout {
            Layout.fillWidth: true
            spacing: Kirigami.Units.largeSpacing
            RowLayout {
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                QQC2.Label { text: pd.gpuStat || "" }
                QQC2.ProgressBar { from: 0; to: 100; value: pd.vramPct || 0; Layout.fillWidth: true; Layout.minimumWidth: Kirigami.Units.gridUnit * 3 }
            }
            RowLayout {
                visible: !!pd.showMon2
                Layout.fillWidth: true
                Layout.preferredWidth: 1
                QQC2.Label { text: pd.gpuStat2 || "" }
                QQC2.ProgressBar { from: 0; to: 100; value: pd.vramPct2 || 0; Layout.fillWidth: true; Layout.minimumWidth: Kirigami.Units.gridUnit * 3 }
            }
        }

        // ------------------------------------------------------------- model
        RowLayout {
            Layout.fillWidth: true
            QQC2.Label { text: "Model" }
            QQC2.ComboBox {
                id: modelBox
                Layout.fillWidth: true
                enabled: !pv.running
                property var items: []
                property string sig: ""
                property var src: pd.models || []
                function sync() {
                    const s = JSON.stringify(src)
                    if (s !== sig) {
                        sig = s
                        items = src.slice()
                        currentIndex = Qt.binding(function () { return modelBox.items.indexOf(pd.model || "") })
                    }
                }
                onSrcChanged: sync()
                Component.onCompleted: sync()
                model: items
                currentIndex: items.indexOf(pd.model || "")
                displayText: currentIndex >= 0 ? currentText : ""
                onActivated: function (i) { pv.act("selectModel", { model: items[i] }) }
            }
            QQC2.ToolButton {
                icon.name: "view-refresh"
                onClicked: Backend.call("app", "rescan", "{}")
                HelpTip { text: "Rescan the models folder"; shown: parent.hovered }
            }
            QQC2.ToolButton {
                icon.name: "documentinfo"
                onClicked: pv.act("showMetadata")
                HelpTip {
                    text: pd.isVllm ? "Show this model's config.json and what the estimate reads from it"
                                    : "Show this model's GGUF metadata (layers, heads, sliding window...)"
                    shown: parent.hovered
                }
            }
        }
        QQC2.Label {
            id: modelInfo
            Layout.fillWidth: true
            text: pd.modelInfo || ""
            color: Kirigami.Theme.disabledTextColor
            elide: Text.ElideRight
            HoverHandler { id: infoHover }
            HelpTip { text: modelInfo.truncated ? modelInfo.text : ""; shown: infoHover.hovered }
        }

        // ------------------------------------------------------ load options
        QQC2.GroupBox {
            Layout.fillWidth: true
            title: pd.isRad ? "Load options (Radiance)" : pd.isVllm ? "Load options (vLLM)" : pd.isKobold ? "Load options (KoboldCpp)" : "Load options (llama.cpp)"
            enabled: !pv.running
            Loader {
                anchors.left: parent.left
                anchors.right: parent.right
                sourceComponent: pd.isRad ? radOptions : pd.isVllm ? vllmOptions : llamaOptions
            }
        }

        // ----------------------------------------------------- dual section
        QQC2.GroupBox {
            Layout.fillWidth: true
            title: "Dual GPU"
            visible: !!pd.showDual
            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                spacing: Kirigami.Units.smallSpacing
                GridLayout {
                    id: dualGrid
                    Layout.fillWidth: true
                    columns: pv.wide ? 8 : 4
                    columnSpacing: Kirigami.Units.smallSpacing
                    OptLabel { visible: !pd.isRad; text: "Strategy"; help: pv.h(pd.isVllm ? "v_dual_strategy" : "dual_strategy") }
                    OptCombo {
                        visible: !pd.isRad
                        pane: pv; key: "dual_strategy"; values: pd.dualStrategies || []
                        enabled: !pv.running
                        help: pv.h(pd.isVllm ? "v_dual_strategy" : "dual_strategy")
                    }
                    OptLabel { visible: !pd.isRad; text: "Split mode"; help: pv.h(pd.isVllm ? "v_dual_mode" : "dual_mode") }
                    OptCombo {
                        visible: !pd.isRad
                        pane: pv; key: "dual_mode"; values: pd.dualModes || []
                        enabled: !!pd.dualManualEnabled
                        help: pv.h(pd.isVllm ? "v_dual_mode" : "dual_mode")
                    }
                    OptLabel { visible: !!pd.isRad; text: "Tensor parallel wire"; help: pv.h("r_tp_wire") }
                    OptCombo {
                        visible: !!pd.isRad
                        pane: pv; key: "r_tp_wire"; values: pv.lists.rTpWire || []
                        enabled: !pv.running
                        help: pv.h("r_tp_wire")
                    }
                    OptLabel { visible: !pd.isRad; text: "Ratio"; help: pv.h(pd.isVllm ? "v_dual_ts" : "dual_ts") }
                    OptText {
                        visible: !pd.isRad
                        pane: pv; key: "dual_ts"
                        enabled: !!pd.dualManualEnabled
                        help: pv.h(pd.isVllm ? "v_dual_ts" : "dual_ts")
                        Layout.fillWidth: true
                        Layout.minimumWidth: Kirigami.Units.gridUnit * 4
                        Layout.preferredWidth: Kirigami.Units.gridUnit * 5
                    }
                    OptLabel { visible: !pd.isRad; text: "First layers on"; help: pv.h("dual_first") }
                    OptCombo {
                        visible: !pd.isRad
                        pane: pv; key: "dual_first"; values: pd.dualFirstValues || []; prefixMatch: true
                        enabled: !!pd.dualManualEnabled
                        help: pv.h("dual_first")
                    }
                    OptLabel { visible: !!pd.isLlama; text: "MTP drafter on"; help: pv.h("mtp_dev") }
                    OptCombo {
                        visible: !!pd.isLlama
                        pane: pv; key: "mtp_dev"; values: ["(auto)"].concat(pd.dualFirstValues || []); prefixMatch: true
                        enabled: !pv.running && !!pd.mtpEnabled
                        help: pv.h("mtp_dev")
                    }
                }
                OptCheck {
                    visible: !pd.isVllm && !pd.isRad
                    pane: pv; key: "dual_moe_spread"
                    text: "Spread CPU MoE layers across both cards"
                    enabled: !pv.running
                    help: pv.h("dual_moe_spread")
                }
                QQC2.Label {
                    Layout.fillWidth: true
                    text: pd.dualInfo || ""
                    color: Kirigami.Theme.disabledTextColor
                    font: Kirigami.Theme.smallFont
                    wrapMode: Text.Wrap
                }
            }
        }

        // ---------------------------------------------------------- estimate
        QQC2.Label {
            id: est
            Layout.fillWidth: true
            text: pd.estText || ""
            color: pv.tc(pd.estColor)
            wrapMode: Text.Wrap
            HoverHandler { id: estHover }
            HelpTip { text: pv.h(pd.isRad ? "r_estimate" : pd.isVllm ? "v_estimate" : "estimate"); shown: estHover.hovered }
        }
        QQC2.ProgressBar {
            Layout.fillWidth: true
            from: 0; to: 100
            value: pd.estPct || 0
        }

        // ----------------------------------------------------------- buttons
        RowLayout {
            Layout.fillWidth: true
            QQC2.Button {
                text: "Load"
                icon.name: "media-playback-start"
                enabled: !pv.running
                onClicked: pv.act("load")
            }
            QQC2.Button {
                text: "Stop"
                icon.name: "media-playback-stop"
                enabled: pv.running
                onClicked: pv.act("stop")
            }
            QQC2.Button {
                id: webBtn
                text: pd.webLabel || "Open web UI"
                icon.name: "internet-web-browser"
                enabled: !!pd.webEnabled
                onClicked: pv.act("openWeb")
                HelpTip { text: pv.h(pd.isRad ? "r_web" : pd.isVllm ? "v_web" : "web"); shown: webBtn.hovered }
            }
            Item { Layout.fillWidth: true }
            QQC2.Button {
                id: copyBtn
                text: "Copy command"
                icon.name: "edit-copy"
                onClicked: pv.act("copyCmd")
                HelpTip { text: pv.h("copy"); shown: copyBtn.hovered }
            }
        }

        // ------------------------------------------------------------ status
        QQC2.GroupBox {
            Layout.fillWidth: true
            title: "Status"
            GridLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                columns: 4
                columnSpacing: Kirigami.Units.largeSpacing
                rowSpacing: 1

                QQC2.Label { text: "State" }
                QQC2.Label {
                    text: pd.state || ""
                    color: pv.tc(pd.stateColor)
                    font.bold: true
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                }
                QQC2.Label { text: "Uptime" }
                QQC2.Label { text: pv.st("uptime"); Layout.fillWidth: true; Layout.preferredWidth: 1 }

                QQC2.Label { text: "Endpoint" }
                QQC2.Label { text: pv.st("url"); Layout.columnSpan: 3; Layout.fillWidth: true; elide: Text.ElideRight }

                Repeater {
                    model: [["speed", "Last request"], ["avg", "Averages"], ["buffers", "GPU memory"],
                            ["cache", "Prompt cache"], ["life", "This setup"]]
                    delegate: QQC2.Label {
                        required property var modelData
                        text: modelData[1]
                        Layout.alignment: Qt.AlignTop
                        Layout.row: 2 + ["speed", "avg", "buffers", "cache", "life"].indexOf(modelData[0])
                        Layout.column: 0
                    }
                }
                Repeater {
                    model: ["speed", "avg", "buffers", "cache", "life"]
                    delegate: QQC2.Label {
                        required property string modelData
                        required property int index
                        text: pv.st(modelData)
                        Layout.row: 2 + index
                        Layout.column: 1
                        Layout.columnSpan: 3
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                    }
                }
            }
        }

        // --------------------------------------------------------------- log
        RowLayout {
            Layout.fillWidth: true
            QQC2.Label { text: "Server output"; font.bold: true; Layout.fillWidth: true }
            QQC2.ToolButton {
                id: copyLogBtn
                text: "Copy log"
                icon.name: "edit-copy"
                display: QQC2.AbstractButton.TextBesideIcon
                onClicked: {
                    pv.act("copyLog")
                    copyLogBtn.text = "Copied ✓"
                    copyReset.restart()
                }
                Timer { id: copyReset; interval: 1500; onTriggered: copyLogBtn.text = "Copy log" }
                HelpTip { text: "Copy the launch command, estimate and full server output to the clipboard"; shown: copyLogBtn.hovered }
            }
            QQC2.ToolButton {
                id: saveLogBtn
                text: "Save log"
                icon.name: "document-save"
                display: QQC2.AbstractButton.TextBesideIcon
                onClicked: pv.act("saveLog")
                HelpTip { text: "Save the same text to a file in the logs folder"; shown: saveLogBtn.hovered }
            }
        }
        // The log box takes whatever height is left in the window (at least a few lines) and
        // scrolls inside itself. It sits in a plain Item with a fixed size request, so its text
        // can never make the pane taller, whatever the widget style does with scroll views.
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: Kirigami.Units.gridUnit * 4
            Layout.preferredHeight: Kirigami.Units.gridUnit * 4
            implicitHeight: Kirigami.Units.gridUnit * 4

            QQC2.ScrollView {
                id: logView
                anchors.fill: parent
                clip: true
                QQC2.ScrollBar.horizontal.policy: QQC2.ScrollBar.AlwaysOff

                Flickable {
                    id: logFlick
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    contentWidth: width
                    contentHeight: logArea.implicitHeight

                    QQC2.TextArea.flickable: QQC2.TextArea {
                        id: logArea
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        textFormat: TextEdit.PlainText
                        font.family: "monospace"
                        font.pointSize: Math.max(7, Qt.application.font.pointSize - 2)
                    }
                }
            }
        }
    }
    }

    // ================================================== llama.cpp / KoboldCpp
    Component {
        id: llamaOptions
        GridLayout {
            columns: pv.wide ? 8 : 4
            columnSpacing: Kirigami.Units.smallSpacing
            rowSpacing: Kirigami.Units.smallSpacing

            OptLabel { text: "Context"; help: pv.h("ctx") }
            RowLayout {
                spacing: 2
                Layout.fillWidth: true
                OptSpin { pane: pv; key: "ctx"; from: 512; to: 1048576; stepSize: 1024; help: pv.h("ctx") }
                QQC2.Button {
                    id: fitBtn
                    text: "Fit"
                    Layout.minimumWidth: implicitWidth   // never squeezed away by the spin box
                    Layout.fillWidth: false
                    onClicked: pv.act("fit")
                    HelpTip { text: pv.h("fit"); shown: fitBtn.hovered }
                }
            }
            OptLabel { text: "GPU layers"; help: pv.h("ngl") }
            OptSpin { pane: pv; key: "ngl"; from: -1; to: pd.nglMax || 999; help: pv.h("ngl") }
            OptLabel { text: "Flash attn"; help: pv.h("fa") }
            OptCombo { pane: pv; key: "fa"; values: pv.lists.fa || []; help: pv.h("fa") }
            OptLabel { text: "KV cache"; help: pv.h("kv") }
            OptCombo { pane: pv; key: "kv"; values: pv.lists.kvTypes || []; help: pv.h("kv") }

            OptLabel { text: "CPU MoE"; help: pv.h("n_cpu_moe") }
            OptSpin { pane: pv; key: "n_cpu_moe"; from: 0; to: 999; help: pv.h("n_cpu_moe") }
            OptLabel { text: "Parallel"; help: pv.h("parallel") }
            OptSpin { pane: pv; key: "parallel"; from: 1; to: 64; enabled: !pd.isKobold; help: pv.h("parallel") }
            OptLabel { text: "Threads"; help: pv.h("threads") }
            OptSpin { pane: pv; key: "threads"; from: 0; to: 256; help: pv.h("threads") }
            OptLabel { text: "Port"; help: pv.h("port") }
            OptSpin { pane: pv; key: "port"; from: 1024; to: 65535; help: pv.h("port") }

            OptLabel { text: "Batch"; help: pv.h("batch") }
            OptSpin { pane: pv; key: "batch"; from: 32; to: 65536; stepSize: 256; help: pv.h("batch") }
            OptLabel { text: "UBatch"; help: pv.h("ubatch") }
            OptSpin { pane: pv; key: "ubatch"; from: 32; to: 65536; stepSize: 128; help: pv.h("ubatch") }
            OptLabel { text: "Build"; help: pv.h("kbuild") }
            OptCombo {
                pane: pv; key: "kbuild"; values: pv.lists.kbuilds || []
                help: pv.h("kbuild") + (pd.kNote ? "\n\nBinary: " + pd.kNote : "")
            }
            RowLayout {
                Layout.columnSpan: 2
                spacing: 0
                OptCheck { pane: pv; key: "jinja"; text: "--jinja"; help: pv.h("jinja") }
                OptCheck { pane: pv; key: "relax_tpl"; text: "Relaxed tpl"; help: pv.h("relax_tpl") }
                OptCheck {
                    visible: !!pd.isKobold
                    pane: pv; key: "kquiet"; text: "Quiet log"; help: pv.h("kquiet")
                }
            }

            OptLabel { text: "mmproj"; help: pv.h("mmproj") }
            OptCombo {
                pane: pv; key: "mmproj"; values: pd.mmprojValues || []; help: pv.h("mmproj")
                Layout.columnSpan: pv.wide ? 7 : 3
                Layout.preferredWidth: Kirigami.Units.gridUnit * 12
            }

            OptLabel { text: "MTP drafter"; help: pv.h("mtp") }
            OptCombo {
                pane: pv; key: "mtp"; values: pd.mtpValues || []; help: pv.h("mtp")
                enabled: !!pd.mtpEnabled
                Layout.columnSpan: pv.wide ? 5 : 3
                Layout.preferredWidth: Kirigami.Units.gridUnit * 12
            }
            OptLabel { text: "Draft max"; help: pv.h("mtp_n") }
            OptSpin { pane: pv; key: "mtp_n"; from: 1; to: 16; help: pv.h("mtp_n") }
            Item { visible: !pv.wide; Layout.columnSpan: 2; Layout.preferredHeight: 1 }

            QQC2.Label {
                Layout.columnSpan: pv.wide ? 8 : 4
                Layout.fillWidth: true
                visible: text.length > 0
                text: [pd.mtpNote || "", pd.kNoteColor === "#d0342c" ? pd.kNote : ""].filter(function (x) { return x }).join("\n")
                color: pd.kNoteColor === "#d0342c" ? Kirigami.Theme.negativeTextColor : Kirigami.Theme.disabledTextColor
                font: Kirigami.Theme.smallFont
                wrapMode: Text.Wrap
            }

            OptLabel { text: "Extra args"; help: pv.h("extra") }
            OptText {
                pane: pv; key: "extra"; help: pv.h("extra")
                Layout.columnSpan: pv.wide ? 7 : 3; Layout.fillWidth: true
            }
        }
    }

    // ================================================================= vLLM
    Component {
        id: vllmOptions
        GridLayout {
            columns: pv.wide ? 8 : 4
            columnSpacing: Kirigami.Units.smallSpacing
            rowSpacing: Kirigami.Units.smallSpacing

            OptLabel { text: "Context"; help: pv.h("v_ctx") }
            RowLayout {
                spacing: 2
                Layout.fillWidth: true
                OptSpin { pane: pv; key: "ctx"; from: 256; to: 1048576; stepSize: 1024; help: pv.h("v_ctx") }
                QQC2.Button {
                    id: vfitBtn
                    text: "Fit"
                    Layout.minimumWidth: implicitWidth   // never squeezed away by the spin box
                    Layout.fillWidth: false
                    onClicked: pv.act("fit")
                    HelpTip { text: pv.h("v_fit"); shown: vfitBtn.hovered }
                }
            }
            OptLabel { text: "GPU memory"; help: pv.h("gpu_util") }
            OptSpin { pane: pv; key: "gpu_util"; from: 30; to: 99; scale: 100; decimals: 2; help: pv.h("gpu_util") }
            OptLabel { text: "KV cache"; help: pv.h("kv_dtype") }
            OptCombo { pane: pv; key: "kv_dtype"; values: pv.lists.vKvTypes || []; help: pv.h("kv_dtype") }
            OptLabel { text: "Dtype"; help: pv.h("dtype") }
            OptCombo { pane: pv; key: "dtype"; values: pv.lists.vDtypes || []; help: pv.h("dtype") }

            OptLabel { text: "Max seqs"; help: pv.h("v_parallel") }
            OptSpin { pane: pv; key: "parallel"; from: 1; to: 256; help: pv.h("v_parallel") }
            OptLabel { text: "Batch tokens"; help: pv.h("max_batched") }
            OptSpin { pane: pv; key: "max_batched"; from: 0; to: 65536; stepSize: 512; help: pv.h("max_batched") }
            OptLabel { text: "CPU offload"; help: pv.h("cpu_offload") }
            OptSpin { pane: pv; key: "cpu_offload"; from: 0; to: 5120; scale: 10; decimals: 1; stepSize: 10; help: pv.h("cpu_offload") }
            OptLabel { text: "Port"; help: pv.h("port") }
            OptSpin { pane: pv; key: "port"; from: 1024; to: 65535; help: pv.h("port") }

            OptLabel { text: "Quantize"; help: pv.h("quant") }
            OptCombo {
                pane: pv; key: "quant"; values: pv.lists.vQuant || []; help: pv.h("quant")
                enabled: !!pd.vQuantEnabled
            }
            OptLabel { text: "Reasoning"; help: pv.h("reasoning") }
            QQC2.ComboBox {
                id: reasonBox
                Layout.columnSpan: pv.wide ? 5 : 1
                Layout.fillWidth: true
                Layout.minimumWidth: Kirigami.Units.gridUnit * 4
                Layout.preferredWidth: Kirigami.Units.gridUnit * 6
                editable: true
                model: pv.lists.vReasoning || []
                currentIndex: (pv.lists.vReasoning || []).indexOf(pv.vars.reasoning || "")
                Binding {
                    target: reasonBox
                    property: "editText"
                    value: pv.vars.reasoning || ""
                    when: !reasonBox.activeFocus
                }
                onActivated: function (i) { pv.setVar("reasoning", reasonBox.textAt(i)) }
                onAccepted: pv.setVar("reasoning", reasonBox.editText)
                onActiveFocusChanged: if (!activeFocus) pv.setVar("reasoning", reasonBox.editText)
                Connections {
                    target: pv
                    function onFlushEdits() { if (reasonBox.editText !== (pv.vars.reasoning || "")) pv.setVar("reasoning", reasonBox.editText) }
                }
                HelpTip { text: pv.h("reasoning"); shown: reasonBox.hovered && !reasonBox.popup.visible }
            }

            OptLabel { text: "MTP"; help: pv.h("v_mtp") }
            OptCombo {
                pane: pv; key: "mtp"; values: pd.vMtpValues || []; help: pv.h("v_mtp")
                enabled: !!pd.vMtpEnabled
            }
            OptLabel { text: "Spec tokens"; help: pv.h("v_mtp_n") }
            OptSpin { pane: pv; key: "mtp_n"; from: 1; to: 8; help: pv.h("v_mtp_n") }
            RowLayout {
                Layout.columnSpan: 4
                spacing: Kirigami.Units.smallSpacing
                OptCheck { pane: pv; key: "eager"; text: "Eager"; help: pv.h("eager") }
                OptCheck {
                    pane: pv; key: "text_only"; text: "Text only"; help: pv.h("text_only")
                    enabled: !!pd.vTextEnabled
                }
                OptCheck { pane: pv; key: "relax_tpl"; text: "Relaxed tpl"; help: pv.h("v_relax_tpl") }
            }

            QQC2.Label {
                Layout.columnSpan: pv.wide ? 8 : 4
                Layout.fillWidth: true
                visible: text.length > 0
                text: pd.vMtpNote || ""
                color: Kirigami.Theme.disabledTextColor
                font: Kirigami.Theme.smallFont
                wrapMode: Text.Wrap
            }

            OptLabel { text: "Extra args"; help: pv.h("v_extra") }
            OptText {
                pane: pv; key: "extra"; help: pv.h("v_extra")
                Layout.columnSpan: pv.wide ? 7 : 3; Layout.fillWidth: true
            }
        }
    }
    // ============================================================= Radiance
    Component {
        id: radOptions
        GridLayout {
            id: rg
            // longer labels than llama.cpp's: four pairs a row only when there is room
            readonly property bool w8: pv.width >= Kirigami.Units.gridUnit * 46
            columns: w8 ? 8 : 4
            columnSpacing: Kirigami.Units.smallSpacing
            rowSpacing: Kirigami.Units.smallSpacing

            OptLabel { text: "Context"; help: pv.h("r_ctx") }
            RowLayout {
                spacing: 2
                Layout.fillWidth: true
                OptSpin { pane: pv; key: "ctx"; from: 0; to: 4194304; stepSize: 1024; help: pv.h("r_ctx") }
                QQC2.Button {
                    id: rfitBtn
                    text: pd.probing ? "Cancel" : "Fit"
                    Layout.minimumWidth: implicitWidth   // never squeezed away by the spin box
                    Layout.fillWidth: false
                    onClicked: pv.act("fit")
                    HelpTip { text: pv.h("r_fit"); shown: rfitBtn.hovered }
                }
            }
            OptLabel { text: "KV cache"; help: pv.h("r_kv") }
            OptCombo { pane: pv; key: "r_kv"; values: pv.lists.rKv || []; help: pv.h("r_kv") }
            OptLabel { text: "Max seqs"; help: pv.h("r_parallel") }
            OptSpin { pane: pv; key: "parallel"; from: 1; to: 256; help: pv.h("r_parallel") }
            OptLabel { text: "Batch tokens"; help: pv.h("r_batched") }
            OptSpin { pane: pv; key: "r_batched"; from: 64; to: 65536; stepSize: 512; help: pv.h("r_batched") }

            OptLabel { text: "Spec tokens"; help: pv.h("r_spec") }
            OptCombo { pane: pv; key: "r_spec"; values: pv.lists.rSpec || []; help: pv.h("r_spec") }
            OptLabel { text: "Draft window"; help: pv.h("r_draft_window") }
            OptCombo { pane: pv; key: "r_draft_window"; values: pv.lists.rDraftWindow || []; help: pv.h("r_draft_window") }
            OptLabel { text: "Headroom"; help: pv.h("r_headroom") }
            OptSpin { pane: pv; key: "r_headroom"; from: 0; to: 16384; stepSize: 32; help: pv.h("r_headroom") }
            OptLabel { text: "Port"; help: pv.h("port") }
            OptSpin { pane: pv; key: "port"; from: 1024; to: 65535; help: pv.h("port") }

            OptLabel { text: "Vision tower"; help: pv.h("r_vision") }
            OptCombo { pane: pv; key: "r_vision"; values: pv.lists.rVision || []; help: pv.h("r_vision") }
            OptLabel { text: "Embedding"; help: pv.h("r_embed") }
            OptCombo { pane: pv; key: "r_embed"; values: pv.lists.rEmbed || []; help: pv.h("r_embed") }
            OptLabel { text: "Placement"; help: pv.h("r_placement") }
            OptCombo { pane: pv; key: "r_placement"; values: pv.lists.rPlacement || []; help: pv.h("r_placement") }
            OptLabel { text: "Host pool"; help: pv.h("r_host_pool") }
            OptSpin { pane: pv; key: "r_host_pool"; from: 0; to: 1048576; stepSize: 1024; help: pv.h("r_host_pool") }

            OptLabel { text: "Prefix cache"; help: pv.h("r_prefix_mib") }
            OptSpin { pane: pv; key: "r_prefix_mib"; from: 0; to: 1048576; stepSize: 1024; help: pv.h("r_prefix_mib") }
            OptLabel { text: "Prefill span"; help: pv.h("r_prefill_span") }
            OptSpin { pane: pv; key: "r_prefill_span"; from: 1; to: 16; help: pv.h("r_prefill_span") }
            OptLabel { text: "Reasoning"; help: pv.h("r_reasoning") }
            QQC2.ComboBox {
                id: rReasonBox
                Layout.columnSpan: 3
                Layout.fillWidth: true
                Layout.minimumWidth: Kirigami.Units.gridUnit * 4
                Layout.preferredWidth: Kirigami.Units.gridUnit * 6
                editable: true
                model: pv.lists.rReasoning || []
                currentIndex: (pv.lists.rReasoning || []).indexOf(pv.vars.r_reasoning || "")
                Binding {
                    target: rReasonBox
                    property: "editText"
                    value: pv.vars.r_reasoning || ""
                    when: !rReasonBox.activeFocus
                }
                onActivated: function (i) { pv.setVar("r_reasoning", rReasonBox.textAt(i)) }
                onAccepted: pv.setVar("r_reasoning", rReasonBox.editText)
                onActiveFocusChanged: if (!activeFocus) pv.setVar("r_reasoning", rReasonBox.editText)
                Connections {
                    target: pv
                    function onFlushEdits() { if (rReasonBox.editText !== (pv.vars.r_reasoning || "")) pv.setVar("r_reasoning", rReasonBox.editText) }
                }
                HelpTip { text: pv.h("r_reasoning"); shown: rReasonBox.hovered && !rReasonBox.popup.visible }
            }

            OptLabel { text: "Extra args"; help: pv.h("r_extra") }
            OptText {
                pane: pv; key: "extra"; help: pv.h("r_extra")
                Layout.columnSpan: rg.w8 ? 7 : 3; Layout.fillWidth: true
            }
        }
    }
}
