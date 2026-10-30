// A form label that shows the option's help text on hover.
import QtQuick
import QtQuick.Controls as QQC2

QQC2.Label {
    id: lbl
    property string help: ""
    HoverHandler { id: hov }
    HelpTip { text: lbl.help; shown: hov.hovered }
}
