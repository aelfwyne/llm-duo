// Settings row: text field + optional Browse… button (file or folder chooser).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Dialogs
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

RowLayout {
    id: row
    property alias text: field.text
    property string help: ""
    property string browse: ""      // "", "file" or "folder"
    property bool secret: false     // an API key / password: shown as dots unless revealed

    function toPath(url) {
        return decodeURIComponent(String(url).replace(/^file:\/\//, ""))
    }
    function startFolder() {
        const t = field.text.trim()
        if (t.length === 0) return ""
        const dir = row.browse === "folder" ? t : t.substring(0, t.lastIndexOf("/"))
        return dir.length > 0 ? "file://" + dir : ""
    }

    QQC2.TextField {
        id: field
        Layout.fillWidth: true
        Layout.minimumWidth: Kirigami.Units.gridUnit * 24
        echoMode: row.secret && !reveal.checked ? TextInput.Password : TextInput.Normal
        HelpTip { text: row.help; shown: field.hovered }
    }
    QQC2.ToolButton {
        id: reveal
        visible: row.secret
        checkable: true
        icon.name: checked ? "password-show-off" : "password-show-on"
        text: checked ? "Hide" : "Show"
        display: QQC2.AbstractButton.IconOnly
        QQC2.ToolTip.visible: hovered
        QQC2.ToolTip.text: checked ? "Hide" : "Show"
    }
    QQC2.Button {
        visible: row.browse.length > 0
        text: "Browse…"
        icon.name: "document-open-folder"
        onClicked: {
            const f = row.startFolder()
            if (row.browse === "folder") {
                if (f) folderDialog.currentFolder = f
                folderDialog.open()
            } else {
                if (f) fileDialog.currentFolder = f
                fileDialog.open()
            }
        }
    }
    FolderDialog {
        id: folderDialog
        onAccepted: field.text = row.toPath(selectedFolder)
    }
    FileDialog {
        id: fileDialog
        fileMode: FileDialog.OpenFile
        onAccepted: field.text = row.toPath(selectedFile)
    }
}
