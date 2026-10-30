// Combo box bound to one pane option.
//
// The list is only replaced when its contents really change: the UI state is re-sent every
// few seconds and a fresh array would otherwise reset the box and close an open popup.
// `prefixMatch` matches "ROCm1" against "ROCm1: AMD Radeon ..." (Dual GPU "First layers on").
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

QQC2.ComboBox {
    id: box
    property var pane
    property string key
    property string help: ""
    property var values: []
    property bool prefixMatch: false
    property var items: []
    property string sig: ""

    function sync() {
        const s = JSON.stringify(values || [])
        if (s !== sig) {
            sig = s
            items = (values || []).slice()
            // re-apply the selection after the model was replaced
            currentIndex = Qt.binding(function () { return indexFor(raw()) })
        }
    }
    function raw() { return pane && pane.pd && pane.pd.vars ? String(pane.pd.vars[key] ?? "") : "" }
    function indexFor(v) {
        if (prefixMatch) {
            const id = v.split(":")[0]
            for (let i = 0; i < items.length; i++)
                if (String(items[i]).split(":")[0] === id && id !== "") return i
            return -1
        }
        return items.indexOf(v)
    }

    Layout.preferredWidth: Kirigami.Units.gridUnit * 6
    Layout.minimumWidth: Kirigami.Units.gridUnit * 4
    Layout.fillWidth: true
    onValuesChanged: sync()
    Component.onCompleted: sync()
    model: items
    currentIndex: indexFor(raw())
    displayText: currentIndex >= 0 ? currentText : raw()
    onActivated: function (i) { pane.setVar(key, items[i]) }

    HelpTip { text: box.help; shown: box.hovered && !box.popup.visible }
}
