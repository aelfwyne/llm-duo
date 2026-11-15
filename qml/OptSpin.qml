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

    // Typing into the box breaks contentItem.text's binding, so after an edit the box would keep
    // showing the typed number even when the backend changes the value (switching models restores
    // that model's saved Context, Fit, …). Whenever the backend value changes, show it again.
    //
    // The KDE desktop style commits every keystroke (value = typed number, clamped to from..to,
    // then valueModified), so the backend value also changes while the user types. Re-showing it
    // then would rewrite the text under the cursor ("8192" ← backspace → "819" → "819" again with
    // the cursor moved, "2" → "512" because of the clamp). While the box is being edited, only a
    // value that differs from what the typed text means is an outside change worth showing.
    readonly property int backendValue: parsed()
    function clampedTyped() {
        const t = String(spin.contentItem.text).trim()
        if (t === "") return NaN
        const v = spin.valueFromText(t, spin.locale)
        return isNaN(v) ? NaN : Math.max(spin.from, Math.min(spin.to, v))
    }
    function showBackend() {
        spin.value = Qt.binding(function () { return spin.parsed() })
        spin.contentItem.text = Qt.binding(function () { return spin.displayText })
    }
    onBackendValueChanged: {
        if (spin.contentItem.activeFocus && clampedTyped() === backendValue) return
        showBackend()
    }
    // leaving the box: show the value that counts (a typed "2" in Context became 512)
    Connections {
        target: spin.contentItem
        function onActiveFocusChanged() {
            if (!spin.contentItem.activeFocus) spin.showBackend()
        }
    }

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
            spin.showBackend()   // keep following the backend (Fit, …)
        }
    }

    HelpTip { text: spin.help; shown: spin.hovered }
}
