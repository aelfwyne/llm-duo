// Check box bound to one boolean pane option.
import QtQuick
import QtQuick.Controls as QQC2

QQC2.CheckBox {
    id: cb
    property var pane
    property string key
    property string help: ""
    checked: pane && pane.pd && pane.pd.vars ? !!pane.pd.vars[key] : false
    onToggled: {
        pane.setVar(key, checked)
        cb.checked = Qt.binding(function () { return pane && pane.pd && pane.pd.vars ? !!pane.pd.vars[key] : false })
    }
    HelpTip { text: cb.help; shown: cb.hovered }
}
