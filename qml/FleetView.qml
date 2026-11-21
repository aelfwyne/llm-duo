// Fleet mode: the pinned Radiance configurations, served together by one radiance process
// (Radiance 2.x multi-model serving). Left: the pins and the fleet's settings and groups.
// Right: Load / Stop / Apply, each model's state from /admin/fleet, and the log.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.Pane {
    id: fv
    property var fd: ({})          // the fleet part of Backend.uiJson
    property var help: ({})
    padding: Kirigami.Units.largeSpacing

    function h(k) { return help[k] || "" }
    function call(action, arg) { Backend.call("fleet", action, JSON.stringify(arg || {})) }
    function setPin(name, key, value) { call("setPin", { name: name, key: key, value: value }) }
    function setServer(key, value) { call("setServer", { key: key, value: value }) }
    function appendLog(text) {
        logArea.append(text)
        logArea.cursorPosition = logArea.length
    }
    function clearLog() { logArea.clear() }

    // The lists are rebuilt only when their names change: the UI state is re-sent every second
    // while the fleet runs, and a fresh array would recreate every editor (and drop the focus).
    property var pinNames: []
    property string pinSig: ""
    property var rowIds: []
    property string rowSig: ""
    function sync() {
        const n = (fd.pins || []).map(function (p) { return p.name })
        const s = JSON.stringify(n)
        if (s !== pinSig) { pinSig = s; pinNames = n }
        const r = (fd.rows || []).map(function (x) { return x.id })
        const rs = JSON.stringify(r)
        if (rs !== rowSig) { rowSig = rs; rowIds = r }
    }
    onFdChanged: sync()
    Component.onCompleted: sync()
    function pinOf(name) {
        const ps = fd.pins || []
        for (let i = 0; i < ps.length; i++) if (ps[i].name === name) return ps[i]
        return ({})
    }
    function rowOf(id) {
        const rs = fd.rows || []
        for (let i = 0; i < rs.length; i++) if (rs[i].id === id) return rs[i]
        return ({})
    }
    function tc(c) {
        return c === "neg" ? Kirigami.Theme.negativeTextColor : c === "pos" ? Kirigami.Theme.positiveTextColor
             : c === "neu" ? Kirigami.Theme.neutralTextColor : Kirigami.Theme.textColor
    }
    function stateColor(s) {
        if (s === "resident") return "pos"
        if (s === "failed") return "neg"
        if (s === "waiting" || s === "loading" || s === "closing") return "neu"
        return ""
    }

    // ------------------------------------------------------------ editors
    component PinText: QQC2.TextField {
        id: pt
        property var pin: ({})
        property string key
        property string tip: ""
        Layout.fillWidth: true
        Layout.minimumWidth: Kirigami.Units.gridUnit * 4
        text: pin.fleet ? String(pin.fleet[key] ?? "") : ""
        onTextEdited: fv.setPin(pin.name, key, text)
        HelpTip { text: pt.tip; shown: pt.hovered }
    }
    component PinCombo: QQC2.ComboBox {
        id: pc
        property var pin: ({})
        property string key
        property var values: []
        property string tip: ""
        Layout.fillWidth: true
        model: values
        currentIndex: Math.max(0, values.indexOf(pin.fleet ? String(pin.fleet[key] || values[0]) : values[0]))
        onActivated: function (i) { fv.setPin(pin.name, key, values[i]) }
        HelpTip { text: pc.tip; shown: pc.hovered && !pc.popup.visible }
    }
    component ServerText: QQC2.TextField {
        id: st
        property string key
        property bool number: false
        property string tip: ""
        Layout.fillWidth: true
        text: fv.fd.server ? String(fv.fd.server[key] ?? "") : ""
        onTextEdited: fv.setServer(key, number ? (Number(text) || 0) : text)
        HelpTip { text: st.tip; shown: st.hovered }
    }

    QQC2.SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        // =========================================================== left
        QQC2.ScrollView {
            id: leftScroll
            QQC2.SplitView.preferredWidth: fv.width * 0.48
            QQC2.SplitView.minimumWidth: Kirigami.Units.gridUnit * 26
            contentWidth: availableWidth

            ColumnLayout {
                width: leftScroll.availableWidth
                spacing: Kirigami.Units.smallSpacing

                Kirigami.Heading { level: 3; text: "Pinned models" }
                QQC2.Label {
                    Layout.fillWidth: true
                    visible: fv.pinNames.length === 0
                    wrapMode: Text.Wrap
                    color: Kirigami.Theme.disabledTextColor
                    text: "Nothing pinned yet. Switch Fleet mode off, set a Radiance model up on a pane (Fit, KV cache, prefix cache …) and press 'Pin to fleet'. A pin made in Dual GPU mode runs tensor parallel over both cards (tp 2); one made on a single pane runs on one card (tp 1), and Radiance picks which card when it loads."
                }

                Repeater {
                    model: fv.pinNames
                    delegate: QQC2.Frame {
                        id: pf
                        required property string modelData
                        readonly property var pin: fv.pinOf(modelData)
                        Layout.fillWidth: true
                        opacity: pin.enabled ? 1.0 : 0.6

                        ColumnLayout {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            spacing: 2
                            RowLayout {
                                Layout.fillWidth: true
                                QQC2.CheckBox {
                                    id: enBox
                                    checked: !!pf.pin.enabled
                                    onToggled: {
                                        fv.setPin(pf.pin.name, "enabled", checked)
                                        enBox.checked = Qt.binding(function () { return !!pf.pin.enabled })
                                    }
                                    HelpTip { text: "Serve this model in the fleet (unticked: kept, but left out of the file)"; shown: enBox.hovered }
                                }
                                QQC2.TextField {
                                    id: nameField
                                    Layout.preferredWidth: Kirigami.Units.gridUnit * 12
                                    font.bold: true
                                    Binding on text { value: pf.pin.name || ""; when: !nameField.activeFocus }
                                    onEditingFinished: if (text !== pf.pin.name) fv.setPin(pf.pin.name, "name", text)
                                    HelpTip { text: fv.h("fleet_name"); shown: nameField.hovered }
                                }
                                QQC2.Label {
                                    text: "tp " + (pf.pin.tp || 1)
                                    font.bold: true
                                    color: (pf.pin.tp || 1) > 1 ? Kirigami.Theme.neutralTextColor : Kirigami.Theme.textColor
                                    HoverHandler { id: tpHover }
                                    HelpTip {
                                        text: (pf.pin.tp || 1) > 1 ? "Tensor parallel over both cards: pinned in Dual GPU mode. It takes both cards whenever it is loaded."
                                                                   : "One card: pinned on a single pane. Radiance picks the card when it loads it."
                                        shown: tpHover.hovered
                                    }
                                }
                                QQC2.Label {
                                    Layout.fillWidth: true
                                    text: pf.pin.model || ""
                                    elide: Text.ElideMiddle
                                    color: pf.pin.missing ? Kirigami.Theme.negativeTextColor : Kirigami.Theme.disabledTextColor
                                }
                                QQC2.Button {
                                    id: repinBtn
                                    text: pf.pin.differs ? "Re-pin (pane differs)" : "Re-pin"
                                    icon.name: "view-refresh"
                                    palette.buttonText: pf.pin.differs ? Kirigami.Theme.neutralTextColor : Kirigami.Theme.textColor
                                    onClicked: fv.call("repin", { name: pf.pin.name })
                                    HelpTip { text: fv.h("fleet_repin"); shown: repinBtn.hovered }
                                }
                                QQC2.Button {
                                    icon.name: "edit-delete"
                                    QQC2.ToolTip.text: "Unpin (remove from the fleet)"
                                    QQC2.ToolTip.visible: hovered
                                    onClicked: fv.call("unpin", { name: pf.pin.name })
                                }
                            }
                            QQC2.Label {
                                Layout.fillWidth: true
                                text: (pf.pin.missing ? "⚠ model file missing · " : "") + (pf.pin.info || "") + (pf.pin.pinnedAt ? "   (pinned " + pf.pin.pinnedAt.replace("T", " ") + ")" : "")
                                color: pf.pin.missing ? Kirigami.Theme.negativeTextColor : Kirigami.Theme.disabledTextColor
                                wrapMode: Text.Wrap
                                font.pointSize: Math.max(7, Qt.application.font.pointSize - 1)
                            }
                            GridLayout {
                                Layout.fillWidth: true
                                columns: 4
                                columnSpacing: Kirigami.Units.smallSpacing
                                rowSpacing: 2
                                QQC2.Label { text: "Aliases" }
                                PinText { pin: pf.pin; key: "aliases"; placeholderText: "e.g. default"; tip: fv.h("fleet_aliases") }
                                QQC2.Label { text: "Priority" }
                                PinText { pin: pf.pin; key: "priority"; placeholderText: "(none)"; tip: fv.h("fleet_priority") }
                                QQC2.Label { text: "Idle TTL" }
                                PinText { pin: pf.pin; key: "ttl"; placeholderText: "(never), e.g. 30m"; tip: fv.h("fleet_ttl") }
                                QQC2.Label { text: "Keep in VRAM" }
                                QQC2.CheckBox {
                                    id: pinBox
                                    checked: !!(pf.pin.fleet && pf.pin.fleet.pin)
                                    onToggled: {
                                        fv.setPin(pf.pin.name, "pin", checked)
                                        pinBox.checked = Qt.binding(function () { return !!(pf.pin.fleet && pf.pin.fleet.pin) })
                                    }
                                    HelpTip { text: fv.h("fleet_pinvram"); shown: pinBox.hovered }
                                }
                                QQC2.Label { text: "Park to RAM" }
                                PinCombo { pin: pf.pin; key: "park_ram"; values: ["allowed", "required", "never"]; tip: fv.h("fleet_park_ram") }
                                QQC2.Label { text: "Restore" }
                                PinCombo { pin: pf.pin; key: "park_restore"; values: ["auto", "full", "live"]; tip: fv.h("fleet_park_restore") }
                                QQC2.Label { text: "KV min tokens" }
                                PinText { pin: pf.pin; key: "kv_min_tokens"; placeholderText: "(default)"; tip: fv.h("fleet_kv_min") }
                                QQC2.Label { text: "KV max tokens" }
                                PinText { pin: pf.pin; key: "kv_max_tokens"; placeholderText: "(default)"; tip: fv.h("fleet_kv_max") }
                                QQC2.Label { text: "Experts max MiB" }
                                PinText { pin: pf.pin; key: "experts_max_mib"; placeholderText: "(default)"; tip: fv.h("fleet_experts") }
                            }
                            QQC2.Label {
                                Layout.fillWidth: true
                                visible: !!pf.pin.aliasError
                                text: "⚠ " + (pf.pin.aliasError || "").split("\n").join("\n⚠ ")
                                color: Kirigami.Theme.negativeTextColor
                                wrapMode: Text.Wrap
                            }
                        }
                    }
                }

                // ------------------------------------------------ fleet settings
                QQC2.GroupBox {
                    Layout.fillWidth: true
                    title: "Fleet settings"
                    GridLayout {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        columns: 4
                        columnSpacing: Kirigami.Units.smallSpacing
                        QQC2.Label { text: "Port" }
                        ServerText { key: "port"; number: true; tip: fv.h("fleet_port"); validator: IntValidator { bottom: 1; top: 65535 } }
                        QQC2.Label { text: "Host memory MiB" }
                        ServerText { key: "host_mib"; number: true; placeholderText: "0 = 75% of RAM"; tip: fv.h("fleet_host_mib"); validator: IntValidator { bottom: 0 } }
                        QQC2.Label { text: "Disk tier" }
                        ServerText { key: "disk_tier"; placeholderText: "(none)"; tip: fv.h("fleet_disk_tier") }
                        QQC2.Label { text: "Load timeout" }
                        ServerText { key: "load_timeout"; placeholderText: "120s"; tip: fv.h("fleet_load_timeout") }
                        QQC2.Label { text: "Warm-up" }
                        QQC2.CheckBox {
                            id: warmBox
                            checked: !!(fv.fd.server && fv.fd.server.warm_up)
                            onToggled: {
                                fv.setServer("warm_up", checked)
                                warmBox.checked = Qt.binding(function () { return !!(fv.fd.server && fv.fd.server.warm_up) })
                            }
                            HelpTip { text: fv.h("fleet_warm_up"); shown: warmBox.hovered }
                        }
                    }
                }

                // ------------------------------------------------------- groups
                QQC2.GroupBox {
                    Layout.fillWidth: true
                    title: "Groups"
                    ColumnLayout {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        Repeater {
                            model: fv.fd.groups || []
                            delegate: RowLayout {
                                required property var modelData
                                required property int index
                                Layout.fillWidth: true
                                QQC2.Label { text: modelData.kind; font.bold: true }
                                QQC2.Label { text: modelData.members; Layout.fillWidth: true; elide: Text.ElideRight }
                                QQC2.Button { icon.name: "list-remove"; onClicked: fv.call("delGroup", { index: index }) }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            QQC2.ComboBox { id: gKind; model: ["exclusive", "together"] }
                            QQC2.TextField { id: gMembers; Layout.fillWidth: true; placeholderText: "pin names, comma separated" }
                            QQC2.Button {
                                id: gAdd
                                text: "Add"
                                icon.name: "list-add"
                                onClicked: { fv.call("addGroup", { kind: gKind.currentText, members: gMembers.text }); gMembers.text = "" }
                                HelpTip { text: fv.h("fleet_groups"); shown: gAdd.hovered }
                            }
                        }
                    }
                }
                QQC2.Label {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    color: Kirigami.Theme.disabledTextColor
                    font.pointSize: Math.max(7, Qt.application.font.pointSize - 1)
                    text: "Written to " + (fv.fd.yamlPath || "") + ". The fleet uses both cards (cards: all); every pane must be stopped."
                }
            }
        }

        // ========================================================== right
        ColumnLayout {
            QQC2.SplitView.fillWidth: true
            QQC2.SplitView.minimumWidth: Kirigami.Units.gridUnit * 26
            spacing: Kirigami.Units.smallSpacing

            RowLayout {
                Layout.fillWidth: true
                QQC2.Button { text: "Load"; icon.name: "media-playback-start"; enabled: !fv.fd.running; onClicked: fv.call("load") }
                QQC2.Button { text: "Stop"; icon.name: "media-playback-stop"; enabled: !!fv.fd.running; onClicked: fv.call("stop") }
                QQC2.Button {
                    id: applyBtn
                    text: fv.fd.changed ? "Apply changes" : "Apply"
                    icon.name: "dialog-ok-apply"
                    enabled: !!fv.fd.running && !!fv.fd.ready
                    highlighted: !!fv.fd.changed
                    onClicked: fv.call("apply")
                    HelpTip { text: fv.h("fleet_apply"); shown: applyBtn.hovered }
                }
                QQC2.Button {
                    id: checkBtn
                    text: "Check"
                    icon.name: "tools-check-spelling"
                    onClicked: fv.call("check")
                    HelpTip { text: fv.h("fleet_check"); shown: checkBtn.hovered }
                }
                Item { Layout.fillWidth: true }
                QQC2.Button { text: "Open dashboard"; icon.name: "internet-web-browser"; enabled: !!fv.fd.ready; onClicked: fv.call("openDash") }
                QQC2.Button { text: "Copy YAML"; icon.name: "edit-copy"; onClicked: fv.call("copyYaml") }
            }
            QQC2.Label {
                Layout.fillWidth: true
                text: (fv.fd.status || "Stopped") + (fv.fd.endpoint ? "   •   " + fv.fd.endpoint + "  (requests pick a model by its \"model\" field)" : "")
                font.bold: true
                elide: Text.ElideRight
            }
            QQC2.Label {
                Layout.fillWidth: true
                visible: !!fv.fd.summary
                text: fv.fd.summary || ""
                color: Kirigami.Theme.disabledTextColor
                elide: Text.ElideRight
            }

            // ------------------------------------------------- model states
            QQC2.Frame {
                Layout.fillWidth: true
                visible: fv.rowIds.length > 0
                ColumnLayout {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    spacing: 1
                    Repeater {
                        model: fv.rowIds
                        delegate: RowLayout {
                            id: rd
                            required property string modelData
                            readonly property var row: fv.rowOf(modelData)
                            Layout.fillWidth: true
                            QQC2.Label { text: rd.row.id || ""; font.bold: true; Layout.preferredWidth: Kirigami.Units.gridUnit * 9; elide: Text.ElideRight }
                            QQC2.Label {
                                text: rd.row.state || ""
                                color: fv.tc(fv.stateColor(rd.row.raw))
                                Layout.preferredWidth: Kirigami.Units.gridUnit * 9
                                elide: Text.ElideRight
                            }
                            QQC2.Label {
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                                color: Kirigami.Theme.disabledTextColor
                                text: (rd.row.cards ? "card " + rd.row.cards + "  " : "") + (rd.row.vram || "")
                                      + ((rd.row.busy || rd.row.queued) ? "   " + rd.row.busy + " running, " + rd.row.queued + " queued" : "")
                                      + (rd.row.pinned ? "   kept in VRAM" : "")
                            }
                            QQC2.ToolButton { text: "Load"; onClicked: fv.call("model", { name: rd.row.id, action: "load" }) }
                            QQC2.ToolButton { text: "Park"; onClicked: fv.call("model", { name: rd.row.id, action: "park" }) }
                            QQC2.ToolButton { text: "Unload"; onClicked: fv.call("model", { name: rd.row.id, action: "unload" }) }
                            QQC2.ToolButton {
                                text: rd.row.pinned ? "Unpin" : "Keep"
                                QQC2.ToolTip.text: rd.row.pinned ? "Let Radiance park it again (until the next reload)" : "Keep it in VRAM until the next reload (the 'Keep in VRAM' box makes it permanent)"
                                QQC2.ToolTip.visible: hovered
                                onClicked: fv.call("model", { name: rd.row.id, action: rd.row.pinned ? "unpin" : "pin" })
                            }
                        }
                    }
                }
            }

            // ------------------------------------------------------------ log
            RowLayout {
                Layout.fillWidth: true
                QQC2.Label { text: "Log"; font.bold: true }
                Item { Layout.fillWidth: true }
                QQC2.ToolButton { text: "Copy log"; icon.name: "edit-copy"; onClicked: fv.call("copyLog") }
            }
            QQC2.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumHeight: Kirigami.Units.gridUnit * 8
                QQC2.TextArea {
                    id: logArea
                    readOnly: true
                    selectByMouse: true
                    wrapMode: TextEdit.Wrap
                    textFormat: TextEdit.PlainText
                    font.family: "monospace"
                    font.pointSize: Math.max(7, Qt.application.font.pointSize - 2)
                }
            }
        }
    }
}
