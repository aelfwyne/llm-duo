// Text field bound to one pane option (Extra args, split ratio).
import QtQuick
import QtQuick.Controls as QQC2

QQC2.TextField {
    id: tf
    property var pane
    property string key
    property string help: ""
    text: pane && pane.pd && pane.pd.vars ? String(pane.pd.vars[key] ?? "") : ""
    onTextEdited: pane.setVar(key, text)
    HelpTip { text: tf.help; shown: tf.hovered }
}
