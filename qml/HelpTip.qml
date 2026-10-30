// Hover tooltip that keeps the help texts' column alignment (monospace), like the Tk version.
import QtQuick
import QtQuick.Controls as QQC2

QQC2.ToolTip {
    id: tip
    property bool shown: false
    visible: shown && text.length > 0
    delay: 450
    timeout: -1
    contentItem: QQC2.Label {
        text: tip.text
        textFormat: Text.PlainText
        font.family: "monospace"
        font.pointSize: Math.max(7, Qt.application.font.pointSize - 1)
    }
}
