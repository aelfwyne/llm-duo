// Live performance dashboard: both servers and every card on one screen (llama.cpp request
// figures; GPU graphs for any backend). Sized to fit a 1440-pixel-high screen.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo – live performance"
    width: Kirigami.Units.gridUnit * 92
    height: Kirigami.Units.gridUnit * 64
    color: th.plane
    visible: false

    property int minutes: 10
    property var d: ({})
    readonly property DashPalette th: DashPalette {}

    readonly property var servers: d.servers || []
    readonly property var gpus: d.gpus || []
    readonly property var reqs: d.requests || []
    readonly property var done: reqs.filter(r => !r.aborted)
    readonly property real chartH: Kirigami.Units.gridUnit * 6.5
    readonly property bool wide: width > Kirigami.Units.gridUnit * 84
    readonly property real gap: Kirigami.Units.largeSpacing

    // ---- findings stay until dismissed (status lines like "draft acceptance fine" don't)
    property var notes: ({})          // id -> {ins, first, last, active, dismissed}
    property var noteList: []
    property var statusList: []
    function updateNotes(list) {
        const now = Date.now()
        const seen = {}
        const n = notes
        const status = []
        for (const ins of list) {
            if (ins.level === "good") { status.push(ins); continue }
            seen[ins.id] = true
            const e = n[ins.id]
            if (e) { e.ins = ins; e.last = now; e.active = true }
            else n[ins.id] = { ins: ins, first: now, last: now, active: true, dismissed: false }
        }
        for (const id in n) {
            if (!seen[id]) {
                if (n[id].dismissed) delete n[id]      // gone after dismissal: may come back later
                else n[id].active = false
            }
        }
        notes = n
        refreshNoteList()
        statusList = status
    }
    function refreshNoteList() {
        const rank = { critical: 0, serious: 1, warning: 2, info: 3 }
        noteList = Object.keys(notes).map(k => notes[k]).filter(e => !e.dismissed)
            .sort((a, b) => (a.active === b.active ? 0 : a.active ? -1 : 1) || (rank[a.ins.level] - rank[b.ins.level]) || (a.first - b.first))
    }
    function dismiss(id) {
        if (notes[id]) notes[id].dismissed = true
        refreshNoteList()
    }
    function dismissAll() {
        for (const id in notes) notes[id].dismissed = true
        refreshNoteList()
    }

    function refresh() {
        d = JSON.parse(Backend.call("dash", "live", JSON.stringify({ minutes: minutes })))
        updateNotes(d.insights || [])
    }
    onMinutesChanged: refresh()
    onVisibleChanged: if (visible) refresh()
    Timer { interval: 1500; repeat: true; running: win.visible; onTriggered: win.refresh() }

    // one x-array for all GPUs (they are sampled together); shorter series are padded at the front
    function gpuSeries(key) {
        let n = 0
        for (const g of gpus) n = Math.max(n, (g.series[key] || []).length)
        return gpus.map(function (g) {
            const v = (g.series[key] || []).slice()
            while (v.length < n) v.unshift(null)
            return { name: g.short, color: th.s(g.slot || 0), values: v }
        })
    }
    function gpuXs() {
        let best = []
        for (const g of gpus) if ((g.series.t || []).length > best.length) best = g.series.t
        return best
    }
    function anyValue(key) {
        for (const g of gpus) for (const v of (g.series[key] || [])) if (v !== null) return true
        return false
    }
    function gpuLegend() { return gpus.map(g => ({ name: g.short, color: th.s(g.slot || 0) })) }
    function srvLegend() { return servers.map(s => ({ name: s.who, color: th.s(s.slot) })) }
    function refsOf(f, label) {
        const out = [], seen = {}
        for (const g of gpus) {
            const v = f(g)
            if (v === null || v === undefined || seen[v]) continue
            seen[v] = true
            out.push({ v: v, label: label(g, v) })
        }
        return out
    }
    function clock(t) { return new Date(t * 1000).toLocaleTimeString(Qt.locale(), "HH:mm:ss") }
    function reqTip(r) {
        let s = clock(r.t) + "  •  " + r.label + (r.aborted ? "  •  aborted" : "")
        if (r.pTok || r.total) s += "\nPrompt: " + th.fmt(r.pTok) + " new tokens" + (r.total ? " of " + th.fmt(r.total) : "") + (r.pTps ? "  @ " + th.fmt(r.pTps, 1) + " t/s" : "")
        if (r.gTok) s += "\nOutput: " + th.fmt(r.gTok) + " tokens  @ " + th.fmt(r.gTps, 1) + " t/s"
        if (r.mtp !== null && r.mtp !== undefined && r.gen) s += "\nDraft acceptance: " + th.fmt(r.mtp) + "%"
        if (r.cache !== null && r.cache !== undefined) s += "\nCache reuse: " + th.fmt(r.cache) + "%"
        if (r.nEnd) s += "\nContext at end: " + th.fmt(r.nEnd) + (r.ctx ? " / " + th.fmt(r.ctx) : "")
        s += "\nDuration: " + th.dur(r.dur)
        return s
    }
    function stateLevel(s) {
        switch (s) {
        case "Ready": case "Processing": return "good"
        case "Loading": return "info"
        case "In system RAM": case "Crashed": case "Exited": return "critical"
        default: return ""
        }
    }
    function avgOf(list) { return list.length ? list.reduce((a, b) => a + b, 0) / list.length : 0 }

    header: QQC2.ToolBar {
        RowLayout {
            anchors.fill: parent
            spacing: win.gap
            Kirigami.Heading { level: 4; text: "Live performance"; Layout.leftMargin: Kirigami.Units.smallSpacing }
            QQC2.Label {
                text: servers.length > 1 ? "both servers and all cards" : (servers.length && servers[0].dual ? "Dual GPU mode" : "")
                color: th.muted
                Layout.fillWidth: true
            }
            QQC2.Label { text: "Range"; color: th.muted }
            Repeater {
                model: [5, 10, 30]
                QQC2.Button {
                    text: modelData + " min"
                    checkable: true
                    checked: win.minutes === modelData
                    flat: !checked
                    onClicked: win.minutes = modelData
                }
            }
        }
    }

    QQC2.ScrollView {
        id: scroll
        anchors.fill: parent
        contentWidth: availableWidth
        QQC2.ScrollBar.horizontal.policy: QQC2.ScrollBar.AlwaysOff

        ColumnLayout {
            width: scroll.availableWidth
            height: Math.max(implicitHeight, scroll.availableHeight)
            spacing: win.gap

            // ------------------------------------------------- one card per server
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                Layout.topMargin: win.gap
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                spacing: win.gap
                Repeater {
                    model: servers.length
                    ChartCard {
                        id: sc
                        th: win.th
                        Layout.preferredWidth: 1
                        Layout.alignment: Qt.AlignTop
                        readonly property var s: win.servers[index] || ({})
                        readonly property var ss: s.session || ({})
                        readonly property var last: s.last || null
                        readonly property var a: s.active || null
                        readonly property var hist: s.history || null
                        readonly property real ctxFill: last && last.nEnd && last.ctx ? 100 * last.nEnd / last.ctx : NaN
                        swatch: th.s(s.slot || 0)
                        title: (s.label || "") + (s.model ? "  —  " + s.model : "")
                        subtitle: [s.backend + (s.build ? " " + s.build : ""),
                                   s.ctx ? "ctx " + th.fmt(s.ctx) : "",
                                   s.kvPool ? "KV pool " + th.compact(s.kvPool) : "",
                                   s.kv ? "KV " + s.kv : "",
                                   s.mtp && s.mtp !== "(off)" ? (s.backend === "Radiance" ? "spec " + (s.mtpN || s.mtp) : "MTP " + s.mtp + " ×" + s.mtpN) : "",
                                   s.loadSecs ? "loaded in " + th.dur(s.loadSecs) : ""].filter(x => x).join("  •  ")

                        QQC2.Label {
                            visible: s.supported === false
                            text: "Request figures are read from llama.cpp's log or Radiance's /metrics; for " + s.backend + " only the GPU graphs are live for now."
                            color: th.muted
                            font.pointSize: th.small
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: win.gap * 1.5
                            Repeater {
                                id: stats
                                model: 7
                                readonly property var rows: [
                                    ["Generation", sc.last && sc.last.gTps ? th.fmt(sc.last.gTps, 1) : "—", "t/s",
                                     "avg " + th.fmt(sc.ss.genTps, 1) + (sc.hist && sc.hist.genTps ? " • setup " + th.fmt(sc.hist.genTps, 1) : ""), ""],
                                    ["Prompt", sc.last && sc.last.pTps && sc.last.pTok >= 256 ? th.fmt(sc.last.pTps, 0) : "—", "t/s",
                                     "avg " + th.fmt(sc.ss.promptTps, 0), ""],
                                    ["Draft", th.fmt(sc.ss.mtp, 0), sc.ss.mtp !== null && sc.ss.mtp !== undefined ? "%" : "",
                                     sc.last && sc.last.mtp !== null && sc.last.mtp !== undefined ? "last " + th.fmt(sc.last.mtp) + "%" : "", sc.ss.mtp < 35 ? "serious" : ""],
                                    ["Cache reuse", th.fmt(sc.ss.cache, 0), sc.ss.cache !== null && sc.ss.cache !== undefined ? "%" : "",
                                     th.compact(sc.ss.cachedTokens) + " tokens", sc.ss.cache !== null && sc.ss.cache !== undefined && sc.ss.cache < 30 && sc.ss.requests >= 2 ? "warning" : ""],
                                    ["Context", isNaN(sc.ctxFill) ? "—" : th.fmt(sc.ctxFill, 0), isNaN(sc.ctxFill) ? "" : "%",
                                     sc.last && sc.last.nEnd ? th.compact(sc.last.nEnd) + " / " + th.compact(sc.last.ctx || sc.s.ctx) : "of " + th.compact(sc.s.ctx), sc.ctxFill > 85 ? "warning" : ""],
                                    ["Requests", th.fmt(sc.ss.requests), "", th.fmt(sc.ss.aborted) + " aborted • busy " + th.fmt(sc.ss.busyPct, 0) + "%", ""],
                                    ["Output", th.compact(sc.ss.genTokens), "tok", th.compact(sc.ss.promptTokens) + " prompt", ""]
                                ]
                                ColumnLayout {
                                    readonly property var r: stats.rows[index] || ["", "", "", "", ""]
                                    spacing: 0
                                    Layout.fillWidth: true
                                    Layout.preferredWidth: 1
                                    RowLayout {
                                        spacing: 4
                                        QQC2.Label { text: r[0]; color: th.text2; font.pointSize: th.small }
                                        Rectangle {
                                            visible: r[4] !== ""
                                            width: 12; height: 12; radius: 6
                                            color: th.status(r[4])
                                            QQC2.Label { anchors.centerIn: parent; text: "!"; color: "#111"; font.bold: true; font.pointSize: th.small - 2 }
                                        }
                                    }
                                    RowLayout {
                                        spacing: 3
                                        QQC2.Label {
                                            text: r[1]
                                            color: th.text
                                            font.pointSize: Math.round(th.base * (index === 0 ? 1.9 : 1.45))
                                            font.weight: Font.DemiBold
                                        }
                                        QQC2.Label { text: r[2]; color: th.muted; font.pointSize: th.small; Layout.alignment: Qt.AlignBaseline }
                                    }
                                    QQC2.Label { text: r[3]; color: th.muted; font.pointSize: th.small; elide: Text.ElideRight; Layout.fillWidth: true }
                                }
                            }
                        }
                        // activity: what this server is doing right now
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: win.gap
                            Rectangle {
                                readonly property string lvl: win.stateLevel(sc.s.state)
                                radius: height / 2
                                color: lvl ? Qt.rgba(th.status(lvl).r, th.status(lvl).g, th.status(lvl).b, 0.18) : th.grid
                                border.color: lvl ? th.status(lvl) : th.border
                                implicitWidth: pill.implicitWidth + 16
                                implicitHeight: pill.implicitHeight + 4
                                RowLayout {
                                    id: pill
                                    anchors.centerIn: parent
                                    spacing: 5
                                    Rectangle { width: 7; height: 7; radius: 3.5; color: parent.parent.lvl ? th.status(parent.parent.lvl) : th.muted }
                                    QQC2.Label { text: (sc.s.state || "Stopped") + (sc.s.running && sc.s.uptime ? "  " + sc.s.uptime : ""); font.pointSize: th.small }
                                }
                            }
                            QQC2.Label {
                                text: sc.a ? (sc.a.genTokens
                                              ? "Generating  •  " + th.fmt(sc.a.genTokens) + " tokens  •  " + th.fmt(sc.a.genTps, 1) + " t/s"
                                              : sc.a.progress !== null && sc.a.progress < 100
                                              ? "Processing prompt" + (sc.a.promptTokens ? " (" + th.fmt(sc.a.promptTokens) + " tokens)" : "")
                                              : "Generating") + "  •  " + th.dur(sc.a.since)
                                           : (sc.s.running ? "Idle — waiting for a request" : "Not running")
                                color: th.text2
                                font.pointSize: th.small
                            }
                            Meter {
                                th: win.th
                                visible: !!sc.a && !sc.a.genTokens
                                Layout.fillWidth: true
                                label: ""
                                text: sc.a && sc.a.progress !== null ? th.fmt(sc.a.progress, 0) + "%" : ""
                                value: sc.a && sc.a.progress !== null ? sc.a.progress / 100 : 0
                                color: th.s(sc.s.slot || 0)
                            }
                            Item { visible: !sc.a || !!sc.a.genTokens; Layout.fillWidth: true }
                        }
                    }
                }
            }

            // ----------------------------------------------- requests (both servers)
            GridLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                columns: 3
                columnSpacing: win.gap
                rowSpacing: win.gap
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "Generation speed per request"
                    subtitle: "tokens/s • dashed: run average"
                    legend: win.srvLegend()
                    BarChart {
                        th: win.th
                        implicitHeight: win.chartH
                        values: reqs.map(r => r.aborted ? null : r.gTps)
                        colors: reqs.map(r => th.s(r.slot || 0))
                        tips: reqs.map(r => win.reqTip(r))
                        refs: servers.filter(s => s.session && s.session.genTps).map(s => ({ v: s.session.genTps, label: (servers.length > 1 ? s.who + " " : "avg ") + th.fmt(s.session.genTps, 1) }))
                        xLabelFirst: reqs.length ? win.clock(reqs[0].t) : ""
                        xLabelLast: reqs.length ? win.clock(reqs[reqs.length - 1].t) : ""
                    }
                }
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "Prompt speed per request"
                    subtitle: "tokens/s of newly processed prompt"
                    legend: win.srvLegend()
                    BarChart {
                        th: win.th
                        implicitHeight: win.chartH
                        decimals: 0
                        values: reqs.map(r => r.aborted || !r.pTok ? null : r.pTps)
                        colors: reqs.map(r => th.s(r.slot || 0))
                        tips: reqs.map(r => win.reqTip(r))
                        refs: servers.filter(s => s.session && s.session.promptTps).map(s => ({ v: s.session.promptTps, label: (servers.length > 1 ? s.who + " " : "avg ") + th.fmt(s.session.promptTps, 0) }))
                        xLabelFirst: reqs.length ? win.clock(reqs[0].t) : ""
                        xLabelLast: reqs.length ? win.clock(reqs[reqs.length - 1].t) : ""
                    }
                }
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "Generation speed vs context"
                    subtitle: "tokens/s against context length at the end of each request"
                    legend: win.srvLegend()
                    ScatterChart {
                        th: win.th
                        implicitHeight: win.chartH
                        xUnit: "tokens"
                        points: done.filter(r => r.nEnd && r.gTps).map(r => ({ x: r.nEnd, y: r.gTps, color: th.s(r.slot || 0), tip: win.reqTip(r) }))
                    }
                }
            }

            // ---------------------------------------------------------- GPU charts
            GridLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                columns: win.wide ? 4 : 2
                columnSpacing: win.gap
                rowSpacing: win.gap
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "GPU memory"
                    subtitle: "GiB • dashed: card size"
                    legend: win.gpuLegend()
                    LineChart {
                        th: win.th
                        implicitHeight: win.chartH
                        xMin: -win.minutes * 60
                        xs: win.gpuXs()
                        series: win.gpuSeries("vram")
                        unit: "GiB"; decimals: 1
                        refs: win.refsOf(g => g.totalGiB, (g, v) => "")
                        emptyText: "No VRAM readings"
                    }
                }
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "GPU load"
                    subtitle: "% busy"
                    legend: win.gpuLegend()
                    LineChart {
                        th: win.th
                        implicitHeight: win.chartH
                        xMin: -win.minutes * 60
                        xs: win.gpuXs()
                        series: win.gpuSeries("busy")
                        yMin: 0; yMax: 100
                        unit: "%"
                        emptyText: "No load readings"
                    }
                }
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "Hotspot temperature"
                    subtitle: "°C • dashed: limit"
                    legend: win.gpuLegend()
                    LineChart {
                        th: win.th
                        implicitHeight: win.chartH
                        xMin: -win.minutes * 60
                        xs: win.gpuXs()
                        series: win.anyValue("junction") ? win.gpuSeries("junction") : win.gpuSeries("edge")
                        yMin: null
                        yFloorMax: 60
                        unit: "°C"
                        refs: win.refsOf(g => g.limits.junction, (g, v) => "limit " + th.fmt(v) + " °C")
                        emptyText: "No temperature sensors"
                    }
                }
                ChartCard {
                    th: win.th
                    Layout.preferredWidth: 1
                    title: "Power"
                    subtitle: "watts • dashed: cap"
                    legend: win.gpuLegend()
                    LineChart {
                        th: win.th
                        implicitHeight: win.chartH
                        xMin: -win.minutes * 60
                        xs: win.gpuXs()
                        series: win.gpuSeries("power")
                        unit: "W"
                        refs: win.refsOf(g => g.limits.power, (g, v) => gpus.length > 1 ? "cap " + th.fmt(v) + " W" : "cap")
                        emptyText: "No power readings"
                    }
                }
            }

            // ------------------------------------------------- cards + analysis
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                spacing: win.gap
                // Models here are counts, not the arrays themselves: a new array every refresh would
                // rebuild the delegates (meters would regrow from zero).
                Repeater {
                    model: gpus.length
                    ChartCard {
                        id: gcard
                        th: win.th
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: 1
                        Layout.fillHeight: true
                        readonly property var g: win.gpus[index] || ({ cur: {}, limits: {} })
                        readonly property var c: g.cur || ({})
                        title: g.short || ""
                        subtitle: (g.id || "") + "  •  " + (g.usedBy ? "used by " + g.usedBy : "not in use")
                                  + (g.live ? "  •  ↑ peak in " + win.minutes + " min" : "  •  no live sensors")
                        swatch: th.s(g.slot || 0)
                        Meter {
                            th: win.th
                            label: "VRAM"
                            value: gcard.c.vramGiB && gcard.g.totalGiB ? gcard.c.vramGiB / gcard.g.totalGiB : 0
                            text: th.fmt(gcard.c.vramGiB, 1) + " / " + th.fmt(gcard.g.totalGiB, 1) + " GiB" + (gcard.c.gttGiB ? "  (+" + th.fmt(gcard.c.gttGiB, 1) + " GTT)" : "")
                            color: th.s(gcard.g.slot || 0)
                            warnAt: 0.97
                        }
                        Meter {
                            th: win.th
                            label: "Load"
                            value: (gcard.c.busy || 0) / 100
                            text: th.fmt(gcard.c.busy) + " %" + (gcard.g.busyAvg !== null && gcard.g.busyAvg !== undefined ? "  •  avg " + th.fmt(gcard.g.busyAvg) + " %" : "")
                            color: th.s(gcard.g.slot || 0)
                        }
                        Meter {
                            th: win.th
                            visible: gcard.c.power !== null && gcard.c.power !== undefined
                            // scale: the cap, or the shown peak rounded up to the next 50 W when it went past the cap
                            readonly property real cap: (gcard.g.limits || {}).power || 0
                            readonly property real pk: (gcard.g.peak || {}).power || 0
                            readonly property real range: Math.max(cap, Math.ceil(Math.max(pk, gcard.c.power || 0) / 50) * 50) || 1
                            label: "Power"
                            value: (gcard.c.power || 0) / range
                            marker: cap > 0 && range > cap ? cap / range : -1
                            warnAt: cap > 0 ? 0.98 * cap / range : 2
                            critAt: cap > 0 ? cap * 1.001 / range : 2
                            text: th.fmt(gcard.c.power) + (cap ? " / " + th.fmt(cap) : "") + " W" + (pk ? "  •  peak " + th.fmt(pk) + " W" : "")
                            color: th.s(gcard.g.slot || 0)
                        }
                        // fixed, equal columns so changing readings never shift their neighbours
                        TextMetrics { id: valW; font.weight: Font.DemiBold; text: "8888 MHz" }
                        TextMetrics { id: pkW; font.pointSize: win.th.small; text: "↑8888" }
                        GridLayout {
                            Layout.fillWidth: true
                            Layout.topMargin: Kirigami.Units.smallSpacing
                            columns: 3
                            columnSpacing: win.gap
                            rowSpacing: Kirigami.Units.smallSpacing
                            Repeater {
                                id: sensors
                                model: 6
                                readonly property var lim: gcard.g.limits || ({})
                                readonly property var pk: gcard.g.peak || ({})
                                readonly property var rows: [
                                    ["Edge", gcard.c.edge, " °C", lim.edge, pk.edge],
                                    ["Hotspot", gcard.c.junction, " °C", lim.junction, pk.junction],
                                    ["Memory", gcard.c.mem, " °C", lim.mem, pk.mem],
                                    ["Fan", gcard.c.fan, " rpm", null, pk.fan],
                                    ["Core clock", gcard.c.sclk, " MHz", null, pk.sclk],
                                    ["Memory clock", gcard.c.mclk, " MHz", null, pk.mclk]
                                ]
                                ColumnLayout {
                                    readonly property var sr: sensors.rows[index]
                                    readonly property bool has: sr[1] !== null && sr[1] !== undefined
                                    readonly property bool hot: has && sr[3] !== null && sr[3] !== undefined && sr[1] >= sr[3] - 10
                                    spacing: 0
                                    Layout.fillWidth: true
                                    Layout.preferredWidth: 1
                                    Layout.minimumWidth: valW.width + pkW.width + win.gap
                                    QQC2.Label { text: sr[0]; color: th.muted; font.pointSize: win.th.small }
                                    RowLayout {
                                        spacing: Kirigami.Units.smallSpacing
                                        QQC2.Label {
                                            Layout.preferredWidth: valW.width
                                            text: parent.parent.has ? th.fmt(sr[1]) + sr[2] : "—"
                                            color: th.text
                                            font.weight: Font.DemiBold
                                        }
                                        QQC2.Label {
                                            Layout.preferredWidth: pkW.width
                                            text: sr[4] !== null && sr[4] !== undefined ? "↑" + th.fmt(sr[4]) : ""
                                            color: th.muted
                                            font.pointSize: win.th.small
                                        }
                                        Rectangle {
                                            visible: parent.parent.hot
                                            width: 14; height: 14; radius: 7
                                            color: th.warning
                                            QQC2.Label { anchors.centerIn: parent; text: "!"; color: "#111"; font.bold: true; font.pointSize: win.th.small - 1 }
                                            QQC2.ToolTip.visible: hotHover.hovered
                                            QQC2.ToolTip.text: "Within 10 °C of the card's limit (" + th.fmt(sr[3]) + " °C)"
                                            HoverHandler { id: hotHover }
                                        }
                                        Item { Layout.fillWidth: true }
                                    }
                                }
                            }
                        }
                    }
                }

                ChartCard {
                    th: win.th
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: 1.3
                    Layout.fillHeight: true
                    title: "Analysis"
                    subtitle: noteList.length ? "findings stay until you dismiss them" : "from requests, card sensors and each setup's history"
                    Repeater {
                        model: noteList.length
                        RowLayout {
                            readonly property var e: win.noteList[index] || ({ ins: {} })
                            Layout.fillWidth: true
                            spacing: win.gap
                            opacity: e.active ? 1 : 0.6
                            Rectangle {
                                Layout.alignment: Qt.AlignTop
                                Layout.topMargin: 2
                                width: 20; height: 20; radius: 10
                                color: th.status(e.ins.level)
                                QQC2.Label { anchors.centerIn: parent; text: th.statusIcon(e.ins.level); color: "#111"; font.weight: Font.Bold; font.pointSize: th.small - 1 }
                            }
                            ColumnLayout {
                                spacing: 0
                                Layout.fillWidth: true
                                QQC2.Label {
                                    text: e.ins.title + "  ·  " + e.ins.scope
                                    font.weight: Font.DemiBold; color: th.text; Layout.fillWidth: true; wrapMode: Text.Wrap
                                }
                                QQC2.Label { text: e.ins.text; color: th.text2; Layout.fillWidth: true; wrapMode: Text.Wrap; font.pointSize: th.small }
                                QQC2.Label {
                                    text: e.active ? "since " + new Date(e.first).toLocaleTimeString(Qt.locale(), "HH:mm:ss")
                                                   : "cleared — last seen " + new Date(e.last).toLocaleTimeString(Qt.locale(), "HH:mm:ss")
                                    color: th.muted; font.pointSize: th.small
                                }
                            }
                            QQC2.ToolButton {
                                Layout.alignment: Qt.AlignTop
                                icon.name: "window-close"
                                text: "✕"
                                display: QQC2.AbstractButton.IconOnly
                                implicitWidth: Kirigami.Units.gridUnit * 1.4
                                implicitHeight: implicitWidth
                                onClicked: win.dismiss(e.ins.id)
                                QQC2.ToolTip.visible: hovered
                                QQC2.ToolTip.text: "Dismiss"
                            }
                        }
                    }
                    // when nothing needs attention: the live status lines (draft acceptance etc.)
                    Repeater {
                        model: noteList.length === 0 ? statusList.length : 0
                        RowLayout {
                            readonly property var ins: win.statusList[index] || ({})
                            Layout.fillWidth: true
                            spacing: win.gap
                            Rectangle {
                                Layout.alignment: Qt.AlignTop
                                Layout.topMargin: 2
                                width: 20; height: 20; radius: 10
                                color: th.good
                                QQC2.Label { anchors.centerIn: parent; text: "✓"; color: "#111"; font.weight: Font.Bold; font.pointSize: th.small - 1 }
                            }
                            ColumnLayout {
                                spacing: 0
                                Layout.fillWidth: true
                                QQC2.Label { text: ins.title + "  ·  " + ins.scope; font.weight: Font.DemiBold; color: th.text; Layout.fillWidth: true; wrapMode: Text.Wrap }
                                QQC2.Label { text: ins.text; color: th.text2; Layout.fillWidth: true; wrapMode: Text.Wrap; font.pointSize: th.small }
                            }
                        }
                    }
                    QQC2.Button {
                        visible: noteList.length > 1
                        flat: true
                        Layout.alignment: Qt.AlignRight
                        text: "Dismiss all"
                        onClicked: win.dismissAll()
                    }
                    QQC2.Label {
                        visible: noteList.length === 0 && statusList.length === 0
                        text: servers.some(s => s.running) ? (reqs.length ? "Nothing unusual." : "Waiting for the first completed request.") : "Load a model to see live analysis."
                        color: th.muted
                    }
                }
            }

            // ------------------------------------------------- one request log
            ChartCard {
                id: reqCard
                th: win.th
                fillBody: true
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumHeight: Kirigami.Units.gridUnit * 8
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                title: "Requests"
                subtitle: reqs.length ? "newest first • " + reqs.length + " this run" : "none yet"
                readonly property var cols: [
                    ["Time", 5.5, r => win.clock(r.t)],
                    ["Server", 0, r => r.label],
                    ["New prompt", 5.5, r => r.pTok ? th.fmt(r.pTok) : "—"],
                    ["Cached", 4.5, r => r.cache !== null && r.cache !== undefined ? th.fmt(r.cache) + " %" : "—"],
                    ["Prompt t/s", 5, r => r.pTps ? th.fmt(r.pTps, 0) : "—"],
                    ["Output", 4.5, r => r.gTok ? th.fmt(r.gTok) : "—"],
                    ["Gen t/s", 4.5, r => r.gTps ? th.fmt(r.gTps, 1) : "—"],
                    ["Draft", 4, r => r.gen ? th.fmt(r.mtp) + " %" : "—"],
                    ["Context", 8, r => r.nEnd ? th.fmt(r.nEnd) + (r.ctx ? " / " + th.compact(r.ctx) : "") : "—"],
                    ["Duration", 5, r => th.dur(r.dur)],
                    ["", 5, r => r.aborted ? "aborted" : ""]
                ]
                readonly property real fixedW: {
                    let w = 0
                    for (const c of cols) w += c[1]
                    return w * Kirigami.Units.gridUnit
                }
                function colW(c, total) { return c[1] === 0 ? Math.max(Kirigami.Units.gridUnit * 12, total - fixedW) : Kirigami.Units.gridUnit * c[1] }
                Row {
                    id: reqHead
                    Layout.fillWidth: true
                    Repeater {
                        model: reqCard.cols
                        QQC2.Label {
                            width: reqCard.colW(modelData, reqHead.width)
                            text: modelData[0]
                            color: th.muted
                            font.pointSize: th.small
                            horizontalAlignment: index < 2 ? Text.AlignLeft : Text.AlignRight
                            rightPadding: win.gap
                            leftPadding: index === 1 ? 14 : 0
                        }
                    }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: th.grid }
                ListView {
                    id: table
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: Kirigami.Units.gridUnit * 4
                    implicitHeight: Math.min(contentHeight, Kirigami.Units.gridUnit * 10)
                    clip: true
                    model: reqs.length
                    QQC2.ScrollBar.vertical: QQC2.ScrollBar {}
                    delegate: Rectangle {
                        width: table.width
                        height: rowL.implicitHeight + 4
                        color: index % 2 ? "transparent" : Qt.rgba(th.text.r, th.text.g, th.text.b, 0.03)
                        readonly property var r: reqs[reqs.length - 1 - index] || ({})
                        Rectangle { x: reqCard.colW(reqCard.cols[0], table.width) + 2; width: 8; height: 8; radius: 4; anchors.verticalCenter: parent.verticalCenter; color: th.s(r.slot || 0) }
                        Row {
                            id: rowL
                            anchors.verticalCenter: parent.verticalCenter
                            Repeater {
                                model: reqCard.cols
                                QQC2.Label {
                                    width: reqCard.colW(modelData, table.width)
                                    text: modelData[2](r)
                                    color: r.aborted ? th.muted : th.text
                                    horizontalAlignment: index < 2 ? Text.AlignLeft : Text.AlignRight
                                    rightPadding: win.gap
                                    leftPadding: index === 1 ? 14 : 0
                                    elide: Text.ElideRight
                                    font.italic: r.aborted && index === reqCard.cols.length - 1
                                }
                            }
                        }
                    }
                }
            }

            // ------------------------------------------- server metrics, one half each
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: false
                Layout.leftMargin: win.gap
                Layout.rightMargin: win.gap
                Layout.bottomMargin: win.gap
                spacing: win.gap
                visible: servers.some(s => (s.metrics || []).length > 0)
                Repeater {
                    model: servers.length
                    ChartCard {
                        id: mc
                        th: win.th
                        Layout.preferredWidth: 1
                        Layout.alignment: Qt.AlignTop
                        readonly property var s: win.servers[index] || ({})
                        swatch: th.s(s.slot || 0)
                        title: "Server metrics  ·  " + (s.label || "")
                        subtitle: (s.metrics || []).length ? (s.backend === "Radiance" ? "radiance /metrics (a selection), as last read" : "llama-server /metrics, as last read") : "not available"
                        Flow {
                            Layout.fillWidth: true
                            spacing: Kirigami.Units.smallSpacing
                            Repeater {
                                model: (mc.s.metrics || []).length
                                Rectangle {
                                    readonly property var mt: (mc.s.metrics || [])[index] || ["", null]
                                    radius: 6
                                    color: Qt.rgba(th.text.r, th.text.g, th.text.b, 0.04)
                                    border.color: th.border
                                    width: mcol.implicitWidth + 12
                                    height: mcol.implicitHeight + 6
                                    ColumnLayout {
                                        id: mcol
                                        x: 6; y: 3
                                        spacing: 0
                                        QQC2.Label { text: mt[0].replace(/_/g, " "); color: th.muted; font.pointSize: win.th.small - 0.5 }
                                        QQC2.Label {
                                            text: mt[1] === null ? "—" : (Math.abs(mt[1]) >= 1000 ? th.fmt(mt[1]) : String(mt[1]))
                                            color: th.text
                                            font.pointSize: win.th.small
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
