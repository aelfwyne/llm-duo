// Radiance convert tool window: rad-convert a checkpoint folder into a .rad container
// (the conversion keeps running when the window is hidden).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Dialogs
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo: convert for Radiance"
    width: Kirigami.Units.gridUnit * 52
    height: Kirigami.Units.gridUnit * 40

    property var help: ({})
    property var c: ({})
    function h(k) { return help[k] || "" }
    function call(action, value) {
        Backend.call("convert", action, JSON.stringify({ value: value === undefined ? "" : value }))
    }
    function appendLog(text) {
        logArea.append(text)
        if (logArea.lineCount > 4000) {
            const t = logArea.text
            let idx = 0
            for (let i = 0; i < logArea.lineCount - 3000 && idx >= 0; i++)
                idx = t.indexOf("\n", idx) + 1
            if (idx > 0) logArea.remove(0, idx)
        }
        logArea.cursorPosition = logArea.length
    }
    function clearLog() { logArea.clear() }
    function toPath(url) { return decodeURIComponent(String(url).replace(/^file:\/\//, "")) }

    visible: !!c.visible
    onClosing: function (close) {
        close.accepted = false
        Qt.callLater(function () { win.call("close") })
    }

    FolderDialog {
        id: srcDialog
        onAccepted: win.call("setSrc", win.toPath(selectedFolder))
    }
    FolderDialog {
        id: draftDialog
        onAccepted: { draftField.text = win.toPath(selectedFolder); win.call("setDraft", draftField.text) }
    }

    // a text field that sends its value on every edit, and follows the backend otherwise
    component BoundField: QQC2.TextField {
        id: bf
        property string action
        property string value: ""
        property string help: ""
        Layout.fillWidth: true
        Binding { target: bf; property: "text"; value: bf.value; when: !bf.activeFocus }
        onTextEdited: win.call(action, text)
        HelpTip { text: bf.help; shown: bf.hovered }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing

        QQC2.Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            color: Kirigami.Theme.disabledTextColor
            text: "Writes a Radiance .rad container from a Hugging Face checkpoint folder (safetensors + config.json). "
                + "A recipe decides each weight's quantisation; recipes are read from " + (win.c.recipesDir || "the recipes folder")
                + ". A finetune uses its base model's recipe. Runs in the background - you can keep using LLM Duo or close this window."
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            OptLabel { text: "Checkpoint"; help: win.h("c_source") }
            RowLayout {
                Layout.fillWidth: true
                QQC2.ComboBox {
                    id: srcBox
                    Layout.fillWidth: true
                    enabled: !win.c.running
                    editable: true
                    property var items: []
                    property string sig: ""
                    property var src: win.c.sources || []
                    function sync() {
                        const s = JSON.stringify(src)
                        if (s !== sig) {
                            sig = s
                            items = src.slice()
                            currentIndex = Qt.binding(function () { return srcBox.items.indexOf(win.c.src || "") })
                        }
                    }
                    onSrcChanged: sync()
                    Component.onCompleted: sync()
                    model: items
                    currentIndex: items.indexOf(win.c.src || "")
                    Binding { target: srcBox; property: "editText"; value: win.c.src || ""; when: !srcBox.activeFocus }
                    onActivated: function (i) { win.call("setSrc", items[i]) }
                    onAccepted: win.call("setSrc", editText)
                    onActiveFocusChanged: if (!activeFocus && editText !== (win.c.src || "")) win.call("setSrc", editText)
                    HelpTip { text: win.h("c_source"); shown: srcBox.hovered && !srcBox.popup.visible }
                }
                QQC2.Button {
                    text: "Browse…"
                    icon.name: "document-open-folder"
                    enabled: !win.c.running
                    onClicked: srcDialog.open()
                }
            }
            OptLabel { text: "Recipe"; help: win.h("c_recipe") }
            RowLayout {
                Layout.fillWidth: true
                QQC2.ComboBox {
                    id: recipeBox
                    Layout.fillWidth: true
                    enabled: !win.c.running
                    property var items: []
                    property string sig: ""
                    property var src: win.c.recipes || []
                    function sync() {
                        const s = JSON.stringify(src)
                        if (s !== sig) {
                            sig = s
                            items = src.slice()
                            currentIndex = Qt.binding(function () { return recipeBox.items.indexOf(win.c.recipe || "") })
                        }
                    }
                    onSrcChanged: sync()
                    Component.onCompleted: sync()
                    model: items
                    currentIndex: items.indexOf(win.c.recipe || "")
                    onActivated: function (i) { win.call("setRecipe", items[i]) }
                    HelpTip { text: win.h("c_recipe"); shown: recipeBox.hovered && !recipeBox.popup.visible }
                }
                QQC2.Button {
                    id: reloadBtn
                    text: "Reload"
                    icon.name: "view-refresh"
                    enabled: !win.c.running
                    onClicked: win.call("refresh")
                    HelpTip { text: "Read the recipes folder and the model folders again"; shown: reloadBtn.hovered }
                }
            }
            Item { visible: !!win.c.about; Layout.preferredWidth: 1 }
            QQC2.Label {
                visible: !!win.c.about
                Layout.fillWidth: true
                text: win.c.about || ""
                wrapMode: Text.Wrap
                font: Kirigami.Theme.smallFont
                color: Kirigami.Theme.disabledTextColor
            }
            OptLabel { text: "Output"; help: win.h("c_output") }
            BoundField { action: "setOut"; value: win.c.out || ""; help: win.h("c_output"); enabled: !win.c.running }
            OptLabel { text: "Metadata (--set)"; help: win.h("c_sets") }
            BoundField { action: "setSets"; value: win.c.sets || ""; help: win.h("c_sets"); enabled: !win.c.running
                placeholderText: "KEY=VALUE … (a recipe that needs some fills them in)" }
            Item { visible: !!win.c.setNote; Layout.preferredWidth: 1 }
            QQC2.Label {
                visible: !!win.c.setNote
                Layout.fillWidth: true
                text: "ⓘ " + (win.c.setNote || "")
                wrapMode: Text.Wrap
                font: Kirigami.Theme.smallFont
                color: Kirigami.Theme.neutralTextColor
            }
            OptLabel { text: "Draft model"; help: win.h("c_draft") }
            RowLayout {
                Layout.fillWidth: true
                BoundField { id: draftField; action: "setDraft"; value: win.c.draft || ""; help: win.h("c_draft"); enabled: !win.c.running
                    placeholderText: "optional: a drafter that ships as its own repository (DSpark, DFlash2)" }
                QQC2.Button { text: "Browse…"; icon.name: "document-open-folder"; enabled: !win.c.running; onClicked: draftDialog.open() }
            }
            OptLabel { text: "Max context"; help: win.h("c_maxctx") }
            BoundField { action: "setMaxCtx"; value: win.c.maxCtx || ""; help: win.h("c_maxctx"); enabled: !win.c.running
                placeholderText: "optional (default: the checkpoint's training context)" }
            OptLabel { text: "Environment"; help: win.h("c_env") }
            BoundField { action: "setEnv"; value: win.c.env || ""; help: win.h("c_env"); enabled: !win.c.running
                placeholderText: "optional: CALIB=/path/to/grams … (for recipes that read $CALIB)" }
            OptLabel { text: "Extra args"; help: win.h("c_extra") }
            BoundField { action: "setExtra"; value: win.c.extra || ""; help: win.h("c_extra"); enabled: !win.c.running
                placeholderText: "e.g. --imatrix FILE  --quant 'PATTERN=QUANTISER:k=v'  --reuse OLD.rad" }
            OptLabel { text: "GPU"; help: win.h("c_gpu") }
            QQC2.ComboBox {
                id: gpuBox
                Layout.fillWidth: true
                enabled: !win.c.running
                property var items: []
                property string sig: ""
                property var src: win.c.gpus || []
                function sync() {
                    const s = JSON.stringify(src)
                    if (s !== sig) {
                        sig = s
                        items = src.slice()
                        currentIndex = Qt.binding(function () { return gpuBox.items.indexOf(win.c.gpu || "(auto)") })
                    }
                }
                onSrcChanged: sync()
                Component.onCompleted: sync()
                model: items
                currentIndex: items.indexOf(win.c.gpu || "(auto)")
                onActivated: function (i) { win.call("setGpu", items[i]) }
                HelpTip { text: win.h("c_gpu"); shown: gpuBox.hovered && !gpuBox.popup.visible }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            visible: !!win.c.hint
            implicitHeight: hintRow.implicitHeight + Kirigami.Units.largeSpacing * 2
            radius: Kirigami.Units.cornerRadius
            color: Qt.rgba(Kirigami.Theme.neutralTextColor.r, Kirigami.Theme.neutralTextColor.g, Kirigami.Theme.neutralTextColor.b, 0.15)
            border.color: Kirigami.Theme.neutralTextColor
            RowLayout {
                id: hintRow
                anchors.fill: parent
                anchors.margins: Kirigami.Units.largeSpacing
                Kirigami.Icon { source: "dialog-warning"; Layout.preferredWidth: Kirigami.Units.iconSizes.medium; Layout.preferredHeight: Layout.preferredWidth }
                QQC2.Label {
                    Layout.fillWidth: true
                    text: win.c.hint || ""
                    wrapMode: Text.Wrap
                }
                QQC2.Button {
                    visible: !!win.c.fixSets
                    text: "Apply the known values"
                    icon.name: "dialog-ok-apply"
                    onClicked: win.call("applyFix")
                    QQC2.ToolTip.visible: hovered
                    QQC2.ToolTip.text: win.c.fixSets || ""
                }
            }
        }
        QQC2.Label {
            Layout.fillWidth: true
            text: win.c.info || ""
            color: Kirigami.Theme.disabledTextColor
            wrapMode: Text.Wrap
        }
        RowLayout {
            Layout.fillWidth: true
            QQC2.Button { text: win.c.planOnly ? "Check plan" : "Start"; icon.name: "media-playback-start"; enabled: !win.c.running; onClicked: win.call("start") }
            QQC2.Button { text: "Stop"; icon.name: "media-playback-stop"; enabled: !!win.c.running; onClicked: win.call("stop") }
            QQC2.CheckBox {
                id: planBox
                text: "Plan only (write nothing)"
                enabled: !win.c.running
                checked: !!win.c.planOnly
                onToggled: win.call("setPlanOnly", checked ? "true" : "false")
                HelpTip { text: win.h("c_plan"); shown: planBox.hovered }
            }
            Item { Layout.fillWidth: true }
            QQC2.Button { text: "Hide"; onClicked: win.call("hide") }
        }
        QQC2.ProgressBar {
            Layout.fillWidth: true
            indeterminate: !!win.c.running
            from: 0; to: 1
            value: 0
        }
        QQC2.Label { Layout.fillWidth: true; text: win.c.status || "Idle"; elide: Text.ElideRight }
        QQC2.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            QQC2.TextArea {
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
