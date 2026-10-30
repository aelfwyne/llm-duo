// History of every configuration (GPU/backend + model + load options): runs, requests,
// aborts, crashes, speeds. Click a row for its request-by-request trend.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo – performance history"
    width: Kirigami.Units.gridUnit * 84
    height: Kirigami.Units.gridUnit * 56
    color: th.plane
    visible: false

    readonly property DashPalette th: DashPalette {}
    property var d: ({})
    property var detail: null
    property var sel: null           // selected row
    property string filter: ""
    property string backendFilter: "All"
    property bool includeEarlier: true
    property string sortKey: "genTps"
    property bool sortDesc: true
    property bool showAllBars: false
    property int minReqs: 0          // charts: only setups with MORE than this many requests
    readonly property string reqsNote: minReqs > 0 ? "  •  setups with more than " + minReqs + " request" + (minReqs === 1 ? "" : "s") : ""

    readonly property var backends: ["llama.cpp ROCm", "llama.cpp Vulkan", "KoboldCpp ROCm", "KoboldCpp Vulkan", "vLLM", "Radiance"]
    function bkey(r) { return r.build ? r.backend + " " + r.build : r.backend }
    function bcolor(r) { return th.s(Math.max(0, backends.indexOf(bkey(r)))) }

    readonly property var rows: {
        let v = (d.rows || []).filter(r => (includeEarlier || r.status === "current")
                                            && (backendFilter === "All" || bkey(r) === backendFilter)
                                            && (filter === "" || (r.model + " " + r.hardware + " " + bkey(r)).toLowerCase().indexOf(filter.toLowerCase()) >= 0))
        const k = sortKey, desc = sortDesc
        v.sort(function (a, b) {
            let x = a[k], y = b[k]
            if (x === null || x === undefined) return 1
            if (y === null || y === undefined) return -1
            if (typeof x === "string") return desc ? y.localeCompare(x) : x.localeCompare(y)
            return desc ? y - x : x - y
        })
        return v
    }
    readonly property var chartRows: rows.filter(r => (r.requests || 0) > minReqs)
    readonly property var speedRows: chartRows.filter(r => r.genTps).slice().sort((a, b) => b.genTps - a.genTps)
    readonly property var promptRows: chartRows.filter(r => r.promptTps).slice().sort((a, b) => b.promptTps - a.promptTps)

    property var menuRow: null
    property string deleteMsg: ""
    function deleteRow(r) {
        const res = JSON.parse(Backend.call("dash", "delete", JSON.stringify({ key: r.key, model: r.model, status: r.status,
                                                                              since: r.since || "", retired: r.retired || "" })))
        if (res && res.error) {
            deleteMsg = res.error
        } else {
            deleteMsg = "Deleted " + label(r) + " (" + r.hardware + ", " + bkey(r) + ")" + (res.events ? " and " + res.events + " recorded event(s)" : "")
            if (sel && sel.id === r.id) { sel = null; detail = null }
        }
        refresh()
    }

    QQC2.Menu {
        id: rowMenu
        QQC2.MenuItem {
            text: "Show details"
            icon.name: "view-list-details"
            onTriggered: if (win.menuRow) win.select(win.menuRow)
        }
        QQC2.MenuSeparator {}
        QQC2.MenuItem {
            text: "Delete this configuration…"
            icon.name: "edit-delete"
            onTriggered: win.pendingDelete = win.menuRow
        }
    }
    // Confirmation inside this window (a dialog window could open behind it): a bar at the
    // bottom, so deleting several configurations is right-click, Delete, Delete.
    property var pendingDelete: null
    footer: QQC2.ToolBar {
        visible: win.pendingDelete !== null
        RowLayout {
            anchors.fill: parent
            spacing: Kirigami.Units.largeSpacing
            Kirigami.Icon { source: "edit-delete"; Layout.preferredWidth: Kirigami.Units.iconSizes.smallMedium; Layout.preferredHeight: Layout.preferredWidth }
            QQC2.Label {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                readonly property var r: win.pendingDelete
                text: r ? "Delete " + win.label(r) + "  (" + r.hardware + ", " + win.bkey(r) + ", "
                          + (r.status === "earlier" ? "earlier settings" : "current settings") + ")?  "
                          + (r.runs || 0) + " run(s), " + (r.requests || 0) + " request(s) since " + win.day(r.since)
                          + " are removed for good. The model and its load options are not touched." : ""
            }
            QQC2.Button {
                id: delBtn
                text: "Delete"
                icon.name: "edit-delete"
                onClicked: { const r = win.pendingDelete; win.pendingDelete = null; if (r) win.deleteRow(r) }
            }
            QQC2.Button {
                text: "Cancel"
                onClicked: win.pendingDelete = null
            }
        }
    }
    Shortcut { sequence: "Escape"; enabled: win.pendingDelete !== null; onActivated: win.pendingDelete = null }
    Shortcut {
        sequences: ["Return", "Enter"]
        enabled: win.pendingDelete !== null
        onActivated: { const r = win.pendingDelete; win.pendingDelete = null; if (r) win.deleteRow(r) }
    }

    function refresh() {
        const arg = sel ? { key: sel.key, model: sel.model, id: sel.id } : {}
        d = JSON.parse(Backend.call("dash", "history", JSON.stringify(arg)))
        detail = d.detail || null
        if (sel) {
            const again = (d.rows || []).find(r => r.id === sel.id)
            if (again) sel = again
        }
    }
    function select(r) {
        sel = r
        refresh()
    }
    onVisibleChanged: if (visible) refresh()

    function label(r) { return r.model.replace(/\.gguf$/i, "") }
    function hw(r) { return r.hardware + "  •  " + bkey(r) }
    function day(s) { return s ? String(s).substring(0, 10) : "—" }

    header: QQC2.ToolBar {
        RowLayout {
            anchors.fill: parent
            spacing: Kirigami.Units.largeSpacing
            Kirigami.SearchField {
                Layout.preferredWidth: Kirigami.Units.gridUnit * 16
                placeholderText: "Filter models / GPUs…"
                onTextChanged: win.filter = text
            }
            QQC2.ComboBox {
                model: ["All"].concat(win.backends)
                onActivated: win.backendFilter = currentText
            }
            QQC2.CheckBox {
                text: "Include earlier settings"
                checked: win.includeEarlier
                onToggled: win.includeEarlier = checked
                QQC2.ToolTip.visible: hovered
                QQC2.ToolTip.text: "Totals of a configuration before its load options were changed (recorded from now on)"
            }
            QQC2.Label { text: "Charts: more than"; color: th.muted }
            QQC2.SpinBox {
                id: minReqsBox
                Layout.preferredWidth: Kirigami.Units.gridUnit * 5
                from: 0; to: 999
                editable: true
                value: win.minReqs
                onValueModified: win.minReqs = value
                QQC2.ToolTip.visible: hovered
                QQC2.ToolTip.text: "Only setups with more than this many completed requests appear in the speed charts (0 = all). The table below still lists every setup."
            }
            QQC2.Label { text: "requests"; color: th.muted }
            Item { Layout.fillWidth: true }
            QQC2.Label { text: "Sort"; color: th.muted }
            QQC2.ComboBox {
                id: sortBox
                readonly property var keys: [["genTps", "Generation speed"], ["promptTps", "Prompt speed"], ["requests", "Requests"],
                                             ["runs", "Runs"], ["last", "Last used"], ["model", "Model"], ["mtp", "Draft acceptance"],
                                             ["crashes", "Crashes"]]
                model: keys.map(k => k[1])
                onActivated: function (i) { win.sortKey = keys[i][0]; win.sortDesc = keys[i][0] !== "model" }
            }
            QQC2.ToolButton { icon.name: "view-refresh"; text: "Refresh"; onClicked: win.refresh() }
        }
    }

    QQC2.ScrollView {
        id: scroll
        anchors.fill: parent
        contentWidth: availableWidth
        QQC2.ScrollBar.horizontal.policy: QQC2.ScrollBar.AlwaysOff

        ColumnLayout {
            width: scroll.availableWidth
            spacing: Kirigami.Units.largeSpacing
            Item { implicitHeight: 1 }

            // ---------------------------------------------------------- tiles
            RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: Kirigami.Units.largeSpacing
                Layout.rightMargin: Kirigami.Units.largeSpacing
                spacing: Kirigami.Units.largeSpacing
                readonly property var t: d.totals || ({})
                StatTile { th: win.th; hero: true; label: "Tokens generated"; value: th.compact(parent.t.genTokens); sub: th.compact(parent.t.promptTokens) + " prompt tokens processed" }
                StatTile { th: win.th; label: "Configurations"; value: th.fmt(parent.t.configs); sub: th.fmt(parent.t.models) + " models" }
                StatTile { th: win.th; label: "Runs"; value: th.fmt(parent.t.runs); sub: th.fmt(parent.t.loadFailed) + " failed to load" }
                StatTile { th: win.th; label: "Requests"; value: th.compact(parent.t.requests); sub: th.fmt(parent.t.aborts) + " aborted" }
                StatTile { th: win.th; label: "Crashes"; value: th.fmt(parent.t.crashes); level: parent.t.crashes > 0 ? "serious" : ""; sub: "while running or loading" }
                StatTile { th: win.th; label: "RAM spills"; value: th.fmt(parent.t.spills); level: parent.t.spills > 0 ? "warning" : ""; sub: "model pushed out of VRAM" }
            }

            Kirigami.InlineMessage {
                Layout.fillWidth: true
                Layout.leftMargin: Kirigami.Units.largeSpacing
                Layout.rightMargin: Kirigami.Units.largeSpacing
                visible: true
                type: Kirigami.MessageType.Information
                text: "Totals per configuration come from your settings file (they restart when a configuration's load options change). "
                      + "Aborted requests, failed loads, per-request trends and the totals of replaced settings are recorded "
                      + (d.historySince ? "since " + new Date(d.historySince * 1000).toLocaleDateString(Qt.locale(), Locale.ShortFormat) : "from now on") + "."
            }

            // --------------------------------------------------------- charts
            GridLayout {
                Layout.fillWidth: true
                Layout.leftMargin: Kirigami.Units.largeSpacing
                Layout.rightMargin: Kirigami.Units.largeSpacing
                columns: win.width > Kirigami.Units.gridUnit * 64 ? 2 : 1
                columnSpacing: Kirigami.Units.largeSpacing
                rowSpacing: Kirigami.Units.largeSpacing

                Repeater {
                    model: [
                        { title: "Generation speed", sub: "average tokens/s per configuration • whisker: slowest to fastest request" + win.reqsNote, key: "genTps", list: win.speedRows, d: 1 },
                        { title: "Prompt speed", sub: "average tokens/s for prompts of 1K+ new tokens" + win.reqsNote, key: "promptTps", list: win.promptRows, d: 0 }
                    ]
                    ChartCard {
                        id: bars
                        th: win.th
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: 1
                        readonly property var spec: modelData
                        readonly property var list: win.showAllBars ? spec.list : spec.list.slice(0, 15)
                        readonly property real maxV: {
                            let m = 1
                            for (const r of spec.list) m = Math.max(m, r[spec.key], spec.key === "genTps" && r.genMax ? r.genMax : 0)
                            return m
                        }
                        title: spec.title
                        subtitle: spec.sub
                        legend: win.backends.map((b, i) => ({ name: b, color: th.s(i) })).filter(l => (d.rows || []).some(r => win.bkey(r) === l.name))
                        Repeater {
                            model: bars.list
                            Item {
                                Layout.fillWidth: true
                                implicitHeight: Kirigami.Units.gridUnit * 1.5
                                readonly property var r: modelData
                                readonly property real v: r[bars.spec.key]
                                readonly property real labelW: Math.min(width * 0.45, Kirigami.Units.gridUnit * 22)
                                readonly property real barArea: width - labelW - Kirigami.Units.gridUnit * 3.5
                                function bx(val) { return labelW + val / bars.maxV * barArea }
                                HoverHandler { id: hh }
                                TapHandler { onTapped: win.select(r) }
                                Rectangle {
                                    anchors.fill: parent
                                    color: win.sel && win.sel.id === r.id ? Qt.rgba(th.text.r, th.text.g, th.text.b, 0.07)
                                         : hh.hovered ? Qt.rgba(th.text.r, th.text.g, th.text.b, 0.04) : "transparent"
                                    radius: 4
                                }
                                Column {
                                    x: 4
                                    width: labelW - 10
                                    anchors.verticalCenter: parent.verticalCenter
                                    QQC2.Label { width: parent.width; text: win.label(r); elide: Text.ElideMiddle; color: th.text; font.pointSize: win.th.small }
                                    QQC2.Label { width: parent.width; text: r.hardware + (r.status === "earlier" ? "  •  earlier settings" : ""); elide: Text.ElideRight; color: th.muted; font.pointSize: win.th.small * 0.92 }
                                }
                                // min-max whisker
                                Rectangle {
                                    // (min/max skip very short replies, so the average can sit outside them)
                                    readonly property real lo: Math.min(r.genMin || v, v)
                                    readonly property real hi: Math.max(r.genMax || v, v)
                                    visible: bars.spec.key === "genTps" && r.genMin !== null && r.genMax !== null
                                    x: bx(lo); width: Math.max(1, bx(hi) - bx(lo)); height: 1
                                    anchors.verticalCenter: parent.verticalCenter
                                    color: th.text2
                                }
                                Rectangle {
                                    x: labelW
                                    width: Math.max(2, bx(v) - labelW)
                                    height: Math.min(14, parent.height - 8)
                                    anchors.verticalCenter: parent.verticalCenter
                                    radius: 4
                                    color: win.bcolor(r)
                                    // square end on the baseline
                                    Rectangle { width: 4; height: parent.height; color: parent.color; visible: parent.width > 4 }
                                }
                                QQC2.Label {
                                    x: Math.max(bx(v), bars.spec.key === "genTps" && r.genMax ? bx(Math.max(r.genMax, v)) : 0) + 6
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: th.fmt(v, bars.spec.d)
                                    color: th.text
                                    font.pointSize: win.th.small
                                }
                                QQC2.ToolTip.visible: hh.hovered
                                QQC2.ToolTip.delay: 300
                                QQC2.ToolTip.text: win.label(r) + "\n" + win.hw(r) + "\n" + th.fmt(r.genTps, 1) + " t/s generation"
                                                   + (r.genMin !== null ? " (" + th.fmt(r.genMin, 1) + "–" + th.fmt(r.genMax, 1) + ")" : "")
                                                   + "\n" + th.fmt(r.promptTps, 0) + " t/s prompt  •  " + th.fmt(r.requests) + " requests"
                            }
                        }
                        QQC2.Button {
                            visible: bars.spec.list.length > 15
                            flat: true
                            text: win.showAllBars ? "Show top 15" : "Show all " + bars.spec.list.length
                            onClicked: win.showAllBars = !win.showAllBars
                        }
                        QQC2.Label { visible: bars.spec.list.length === 0; text: win.minReqs > 0 && win.rows.length > 0 ? "No setup with more than " + win.minReqs + " request" + (win.minReqs === 1 ? "" : "s") + " matches - lower the number to see more" : "No completed requests yet"; color: th.muted }
                    }
                }
            }

            // ---------------------------------------------------------- detail
            ChartCard {
                id: detailCard
                th: win.th
                visible: !!win.sel
                Layout.leftMargin: Kirigami.Units.largeSpacing
                Layout.rightMargin: Kirigami.Units.largeSpacing
                readonly property var r: win.sel || ({})
                // only this row's settings: from when they started (and, for earlier settings, until replaced)
                readonly property var allReqs: (win.detail && win.detail.requests) || []
                readonly property var reqs: allReqs.filter(x => (!r.since || (x.when || "") >= r.since)
                                                             && (!r.retired || (x.when || "") <= r.retired))
                readonly property var ok: reqs.filter(x => x.kind === "request" && x.gTps)
                title: r.model ? win.label(r) : ""
                subtitle: r.model ? win.hw(r) + "  •  " + (r.status === "earlier" ? "earlier settings (replaced " + win.day(r.retired) + ")" : "current settings") : ""
                swatch: r.model ? win.bcolor(r) : "transparent"
                Flow {
                    Layout.fillWidth: true
                    spacing: Kirigami.Units.smallSpacing
                    Repeater {
                        model: Object.keys(detailCard.r.setup || {}).filter(k => { const v = detailCard.r.setup[k]; return v !== "" && v !== null })
                        Rectangle {
                            radius: 6
                            color: Qt.rgba(th.text.r, th.text.g, th.text.b, 0.05)
                            border.color: th.border
                            width: chip.implicitWidth + 14; height: chip.implicitHeight + 6
                            QQC2.Label {
                                id: chip
                                anchors.centerIn: parent
                                text: modelData + " " + String(detailCard.r.setup[modelData])
                                color: th.text2
                                font.pointSize: win.th.small
                            }
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: Kirigami.Units.largeSpacing
                    Repeater {
                        model: [
                            ["Runs", th.fmt(detailCard.r.runs), th.fmt(detailCard.r.loadFailed) + " failed  •  " + th.fmt(detailCard.r.loadAborted) + " cancelled"],
                            ["Requests", th.fmt(detailCard.r.requests), th.fmt(detailCard.r.aborts) + " aborted"],
                            ["Generation", th.fmt(detailCard.r.genTps, 1) + " t/s", detailCard.r.genMin !== null ? th.fmt(detailCard.r.genMin, 1) + "–" + th.fmt(detailCard.r.genMax, 1) : ""],
                            ["Prompt", th.fmt(detailCard.r.promptTps, 0) + " t/s", "prompts ≥ 1K tokens"],
                            ["Draft acceptance", detailCard.r.mtp === null ? "off" : th.fmt(detailCard.r.mtp) + " %", ""],
                            ["Cache reuse", detailCard.r.cache === null ? "—" : th.fmt(detailCard.r.cache) + " %", ""],
                            ["Crashes / spills", th.fmt(detailCard.r.crashes) + " / " + th.fmt(detailCard.r.spills), ""],
                            ["Time loaded", detailCard.r.runSecs ? th.dur(detailCard.r.runSecs) : "—", "since " + win.day(detailCard.r.since)]
                        ]
                        ColumnLayout {
                            spacing: 0
                            Layout.fillWidth: true
                            Layout.minimumWidth: Kirigami.Units.gridUnit * 6
                            Layout.preferredWidth: Kirigami.Units.gridUnit * 8
                            QQC2.Label { text: modelData[0]; color: th.muted; font.pointSize: win.th.small }
                            QQC2.Label { text: modelData[1]; color: th.text; font.weight: Font.DemiBold; font.pointSize: win.th.base * 1.25 }
                            QQC2.Label { text: modelData[2]; color: th.muted; font.pointSize: win.th.small; visible: text.length > 0 }
                        }
                    }
                }
                QQC2.Label {
                    text: "Generation speed per request, these settings" + (detailCard.allReqs.length > detailCard.reqs.length
                          ? "  (" + (detailCard.allReqs.length - detailCard.reqs.length) + " more from this model's other settings not shown)" : "")
                    color: th.text2
                    Layout.topMargin: Kirigami.Units.smallSpacing
                }
                BarChart {
                    th: win.th
                    color: win.sel ? win.bcolor(win.sel) : th.s(0)
                    values: detailCard.reqs.map(x => x.kind === "abort" ? null : x.gTps)
                    tips: detailCard.reqs.map(x => (x.when || "").replace("T", " ") + (x.kind === "abort" ? "  •  aborted"
                          : "\n" + th.fmt(x.gTok) + " tokens @ " + th.fmt(x.gTps, 1) + " t/s" + (x.pTok ? "\nprompt " + th.fmt(x.pTok) + " @ " + th.fmt(x.pTps, 0) + " t/s" : "")
                            + (x.nEnd ? "\ncontext " + th.fmt(x.nEnd) : "")))
                    refs: win.sel && win.sel.genTps ? [{ v: win.sel.genTps, label: "average " + th.fmt(win.sel.genTps, 1) }] : []
                    emptyText: "No per-request records for this configuration yet (they are recorded from this version on)"
                    xLabelFirst: detailCard.reqs.length ? win.day(detailCard.reqs[0].when) : ""
                    xLabelLast: detailCard.reqs.length ? win.day(detailCard.reqs[detailCard.reqs.length - 1].when) : ""
                }
            }

            // ----------------------------------------------------------- table
            ChartCard {
                id: tableCard
                th: win.th
                Layout.leftMargin: Kirigami.Units.largeSpacing
                Layout.rightMargin: Kirigami.Units.largeSpacing
                title: "All configurations"
                subtitle: rows.length + " shown • click a row for details • right-click to delete • click a heading to sort"
                readonly property var cols: [
                    ["Model", 0, "model", r => win.label(r)],
                    ["Hardware", 11, "hardware", r => r.hardware],
                    ["Backend", 8, "backend", r => win.bkey(r)],
                    ["Runs", 3.5, "runs", r => th.fmt(r.runs)],
                    ["Failed", 3.5, "loadFailed", r => r.loadFailed ? th.fmt(r.loadFailed) : "·"],
                    ["Requests", 4.5, "requests", r => th.fmt(r.requests)],
                    ["Aborted", 4, "aborts", r => r.aborts ? th.fmt(r.aborts) : "·"],
                    ["Crashes", 4, "crashes", r => r.crashes ? th.fmt(r.crashes) : "·"],
                    ["Gen t/s", 4.5, "genTps", r => th.fmt(r.genTps, 1)],
                    ["Range", 6, "genMin", r => r.genMin !== null ? th.fmt(r.genMin, 0) + "–" + th.fmt(r.genMax, 0) : "—"],
                    ["Prompt t/s", 5, "promptTps", r => th.fmt(r.promptTps, 0)],
                    ["Draft", 3.5, "mtp", r => r.mtp === null ? "off" : th.fmt(r.mtp) + "%"],
                    ["Cache", 3.5, "cache", r => r.cache === null ? "—" : th.fmt(r.cache) + "%"],
                    ["Context", 4.5, "ctx", r => r.setup && r.setup.ctx ? th.compact(r.setup.ctx) : "—"],
                    ["Last used", 6, "last", r => win.day(r.last)]
                ]
                readonly property real fixedW: {
                    let w = 0
                    for (const c of cols) w += c[1]
                    return w * Kirigami.Units.gridUnit
                }
                function colW(c, total) { return c[1] === 0 ? Math.max(Kirigami.Units.gridUnit * 10, total - fixedW) : Kirigami.Units.gridUnit * c[1] }

                Row {
                    id: head
                    Layout.fillWidth: true
                    Repeater {
                        model: tableCard.cols
                        QQC2.Label {
                            width: tableCard.colW(modelData, head.width)
                            text: modelData[0] + (win.sortKey === modelData[2] ? (win.sortDesc ? " ▾" : " ▴") : "")
                            color: win.sortKey === modelData[2] ? th.text : th.muted
                            font.pointSize: win.th.small
                            horizontalAlignment: index < 3 ? Text.AlignLeft : Text.AlignRight
                            rightPadding: Kirigami.Units.smallSpacing * 2
                            leftPadding: index === 0 ? 10 : 0
                            elide: Text.ElideRight
                            TapHandler {
                                onTapped: {
                                    if (win.sortKey === modelData[2]) win.sortDesc = !win.sortDesc
                                    else { win.sortKey = modelData[2]; win.sortDesc = index >= 3 }
                                }
                            }
                        }
                    }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: th.grid }
                Repeater {
                    model: win.rows
                    Rectangle {
                        id: rowRect
                        Layout.fillWidth: true
                        implicitHeight: rr.implicitHeight + 8
                        readonly property var r: modelData
                        color: win.pendingDelete && win.pendingDelete.id === r.id ? Qt.rgba(0.82, 0.23, 0.23, 0.22)
                             : win.sel && win.sel.id === r.id ? Qt.rgba(th.text.r, th.text.g, th.text.b, 0.08)
                             : rh.hovered ? Qt.rgba(th.text.r, th.text.g, th.text.b, 0.05)
                             : index % 2 ? "transparent" : Qt.rgba(th.text.r, th.text.g, th.text.b, 0.025)
                        radius: 4
                        HoverHandler { id: rh }
                        TapHandler { onTapped: win.select(rowRect.r) }
                        TapHandler {
                            acceptedButtons: Qt.RightButton
                            onTapped: function (ev) { win.menuRow = rowRect.r; rowMenu.popup() }
                        }
                        TapHandler {
                            acceptedDevices: PointerDevice.TouchScreen
                            onLongPressed: { win.menuRow = rowRect.r; rowMenu.popup() }
                        }
                        Rectangle { width: 3; height: parent.height - 8; anchors.verticalCenter: parent.verticalCenter; x: 2; radius: 1.5; color: win.bcolor(rowRect.r) }
                        Row {
                            id: rr
                            anchors.verticalCenter: parent.verticalCenter
                            Repeater {
                                model: tableCard.cols
                                QQC2.Label {
                                    width: tableCard.colW(modelData, rowRect.width)
                                    text: modelData[3](rowRect.r)
                                    color: rowRect.r.status === "earlier" ? th.text2 : th.text
                                    font.italic: index === 0 && rowRect.r.status === "earlier"
                                    horizontalAlignment: index < 3 ? Text.AlignLeft : Text.AlignRight
                                    rightPadding: Kirigami.Units.smallSpacing * 2
                                    leftPadding: index === 0 ? 10 : 0
                                    elide: index === 0 ? Text.ElideMiddle : Text.ElideRight
                                }
                            }
                        }
                    }
                }
                QQC2.Label { visible: win.rows.length === 0; text: "Nothing matches the filter."; color: th.muted }
                QQC2.Label {
                    visible: win.deleteMsg !== ""
                    Layout.fillWidth: true
                    text: win.deleteMsg
                    color: th.muted
                    wrapMode: Text.Wrap
                }
            }
            Item { implicitHeight: Kirigami.Units.largeSpacing }
        }
    }
}
