// FP8 quantize tool window (the conversion keeps running when the window is hidden).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo: quantize to FP8"
    width: Kirigami.Units.gridUnit * 46
    height: Kirigami.Units.gridUnit * 32

    property var help: ({})
    property var q: ({})
    function h(k) { return help[k] || "" }
    function call(action, value) {
        Backend.call("quant", action, JSON.stringify({ value: value === undefined ? "" : value }))
    }
    function appendLog(text) {
        logArea.append(text)
        logArea.cursorPosition = logArea.length
    }
    function clearLog() { logArea.clear() }

    visible: !!q.visible
    onClosing: function (close) {
        close.accepted = false
        Qt.callLater(function () { win.call("close") })
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing

        QQC2.Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            color: Kirigami.Theme.disabledTextColor
            text: "Makes a permanent FP8 copy of a 16-bit model: Linear weights become 8-bit floats with one "
                + "scale per output row, activations are quantized per token at run time (W8A8 'FP8 dynamic'). "
                + "About half the size, loads faster than 'Quantize at load', and needs no calibration data. "
                + "Runs on the CPU in the background - you can keep using LLM Duo or close this window."
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            OptLabel { text: "Source model"; help: win.h("q_source") }
            QQC2.ComboBox {
                id: srcBox
                Layout.fillWidth: true
                enabled: !win.q.running
                property var items: []
                property string sig: ""
                property var src: win.q.sources || []
                function sync() {
                    const s = JSON.stringify(src)
                    if (s !== sig) {
                        sig = s
                        items = src.slice()
                        currentIndex = Qt.binding(function () { return srcBox.items.indexOf(win.q.src || "") })
                    }
                }
                onSrcChanged: sync()
                Component.onCompleted: sync()
                model: items
                currentIndex: items.indexOf(win.q.src || "")
                onActivated: function (i) { win.call("setSrc", items[i]) }
                HelpTip { text: win.h("q_source"); shown: srcBox.hovered && !srcBox.popup.visible }
            }
            OptLabel { text: "Output folder"; help: win.h("q_output") }
            QQC2.TextField {
                id: outField
                Layout.fillWidth: true
                text: win.q.out || ""
                onTextEdited: win.call("setOut", text)
                HelpTip { text: win.h("q_output"); shown: outField.hovered }
            }
            OptLabel { text: "Also keep 16-bit"; help: win.h("q_ignore") }
            QQC2.TextField {
                id: ignField
                Layout.fillWidth: true
                text: win.q.ignore || ""
                onTextEdited: win.call("setIgnore", text)
                HelpTip { text: win.h("q_ignore"); shown: ignField.hovered }
            }
        }
        QQC2.Label {
            Layout.fillWidth: true
            text: win.q.info || ""
            color: Kirigami.Theme.disabledTextColor
            wrapMode: Text.Wrap
        }
        RowLayout {
            Layout.fillWidth: true
            QQC2.Button { text: "Start"; icon.name: "media-playback-start"; enabled: !win.q.running; onClicked: win.call("start") }
            QQC2.Button { text: "Stop"; icon.name: "media-playback-stop"; enabled: !!win.q.running; onClicked: win.call("stop") }
            Item { Layout.fillWidth: true }
            QQC2.Button { text: "Hide"; onClicked: win.call("hide") }
        }
        QQC2.ProgressBar {
            Layout.fillWidth: true
            from: 0; to: 1000
            value: win.q.progress || 0
        }
        QQC2.Label { Layout.fillWidth: true; text: win.q.status || "Idle"; elide: Text.ElideRight }
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
