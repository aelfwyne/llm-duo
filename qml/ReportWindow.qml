// Read-only, copyable text window: error reports, GGUF metadata, "why Fit chose...".
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    property string text: ""
    property string logPath: ""
    property string logDir: ""

    width: Kirigami.Units.gridUnit * 56
    height: Kirigami.Units.gridUnit * 34
    visible: true
    onClosing: win.destroy(100)

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing

        QQC2.Label {
            Layout.fillWidth: true
            text: win.logPath.length > 0 ? "Log saved to: " + win.logPath : "Details:"
            elide: Text.ElideMiddle
        }
        QQC2.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            QQC2.TextArea {
                id: area
                text: win.text
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.NoWrap
                textFormat: TextEdit.PlainText
                font.family: "monospace"
                font.pointSize: Math.max(7, Qt.application.font.pointSize - 1)
                Component.onCompleted: cursorPosition = length   // show the end, like the Tk dialog
            }
        }
        RowLayout {
            Layout.fillWidth: true
            QQC2.Button {
                id: copyBtn
                text: "Copy all"
                icon.name: "edit-copy"
                onClicked: {
                    const pos = area.cursorPosition
                    area.selectAll()
                    area.copy()
                    area.deselect()
                    area.cursorPosition = pos
                    copyBtn.text = "Copied ✓"
                }
            }
            QQC2.Button {
                visible: win.logPath.length > 0
                text: "Open logs folder"
                icon.name: "folder-open"
                onClicked: Backend.call("app", "openLogs", "{}")
            }
            Item { Layout.fillWidth: true }
            QQC2.Button {
                text: "Close"
                icon.name: "window-close"
                onClicked: win.close()
            }
        }
    }
}
