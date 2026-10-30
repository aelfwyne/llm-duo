// Backend chooser for the Settings window.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

QQC2.ComboBox {
    id: bb
    property var labels: []
    property string experimental: ""
    property string help: ""
    model: labels
    Layout.preferredWidth: Kirigami.Units.gridUnit * 14
    HelpTip { text: bb.help; shown: bb.hovered && !bb.popup.visible }
}
