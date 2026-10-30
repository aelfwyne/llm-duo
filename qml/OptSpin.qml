// Spin box bound to one pane option. `scale` > 1 shows decimals (value is stored as text).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

QQC2.SpinBox {
    id: spin
    property var pane            // PaneView
    property string key
    property string help: ""
    property int scale: 1
    property int decimals: 0

    function raw() { return pane && pane.pd && pane.pd.vars ? pane.pd.vars[key] : "" }
    function parsed() {
        const n = Number(String(raw()).trim())
        return isNaN(n) || String(raw()).trim() === "" ? 0 : Math.round(n * scale)
    }

    Layout.preferredWidth: Kirigami.Units.gridUnit * 6
    Layout.minimumWidth: Kirigami.Units.gridUnit * 4
    Layout.fillWidth: true
    editable: true
    value: parsed()
    textFromValue: function (v, locale) {
        return scale === 1 ? String(v) : (v / scale).toFixed(decimals)
    }
    valueFromText: function (text, locale) {
        const n = Number(String(text).trim())
        return isNaN(n) ? spin.value : Math.round(n * scale)
    }
    validator: RegularExpressionValidator { regularExpression: /-?[0-9]*[.]?[0-9]*/ }
    onValueModified: pane.setVar(key, scale === 1 ? String(value) : (value / scale).toFixed(decimals))

    // the pane is about to act (Load, Fit, …): a number typed but not yet confirmed counts now
    Connections {
        target: spin.pane
        function onFlushEdits() {
            const t = String(spin.contentItem.text).trim()
            if (t === "" || t === spin.textFromValue(spin.value, spin.locale)) return
            const v = spin.valueFromText(t, spin.locale)
            if (isNaN(v)) return
            const c = Math.max(spin.from, Math.min(spin.to, v))
            spin.pane.setVar(spin.key, spin.scale === 1 ? String(c) : (c / spin.scale).toFixed(spin.decimals))
            spin.value = c
            spin.value = Qt.binding(function () { return spin.parsed() })   // keep following the backend (Fit, …)
            spin.contentItem.text = Qt.binding(function () { return spin.displayText })
        }
    }

    HelpTip { text: spin.help; shown: spin.hovered }
}
