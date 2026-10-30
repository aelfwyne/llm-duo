// Time-series line chart (one y-axis), drawn on a Canvas.
// xs: seconds relative to now (negative), one array shared by all series; values may hold nulls (gaps).
// Hover shows a crosshair and the value of every series at that moment.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Item {
    id: chart
    property var th
    property var xs: []
    property var series: []          // [{name, color, values}]
    property real xMin: -600
    property real xMax: 0
    property var yMin: 0             // number, or null = from the data
    property var yMax: null          // number, or null = from the data
    property real yFloorMax: 1       // the auto maximum is at least this
    property string unit: ""
    property int decimals: 0
    property var refs: []            // [{v, label}] dashed reference lines (limits, totals)
    property string emptyText: "No data yet"
    property int hoverIdx: -1

    implicitHeight: Kirigami.Units.gridUnit * 9
    Layout.fillWidth: true

    readonly property real padL: Kirigami.Units.gridUnit * 2.6
    readonly property real padR: Kirigami.Units.smallSpacing * 2
    readonly property real padT: Kirigami.Units.smallSpacing * 2
    readonly property real padB: Kirigami.Units.gridUnit * 1.2
    readonly property real pw: Math.max(10, width - padL - padR)
    readonly property real ph: Math.max(10, height - padT - padB)

    function hasData() {
        for (let s of series) for (let v of (s.values || [])) if (v !== null && v !== undefined) return true
        return false
    }
    function nice(x) {
        if (x <= 0) return 1
        const e = Math.pow(10, Math.floor(Math.log10(x)))
        const f = x / e
        return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * e
    }
    function yRange() {
        let lo = Infinity, hi = -Infinity
        for (let s of series) for (let v of (s.values || [])) if (v !== null && v !== undefined) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
        for (let r of refs) if (r.v !== null && r.v !== undefined) { hi = Math.max(hi, r.v); lo = Math.min(lo, r.v) }
        if (lo === Infinity) { lo = 0; hi = 1 }
        let a = (yMin === null || yMin === undefined) ? lo : yMin
        let b = (yMax === null || yMax === undefined) ? Math.max(hi, yFloorMax) : yMax
        if (yMin === null || yMin === undefined) {
            const pad = (b - a) * 0.15 || 1
            a = Math.floor((a - pad) / nice((b - a + 2 * pad) / 4)) * nice((b - a + 2 * pad) / 4)
        }
        if (yMax === null || yMax === undefined) {
            const step = nice((b - a) / 4)
            b = Math.ceil((b + (b - a) * 0.05) / step) * step
        }
        if (b <= a) b = a + 1
        return [a, b]
    }
    function px(x) { return padL + (x - xMin) / (xMax - xMin) * pw }
    function py(v, r) { return padT + ph - (v - r[0]) / (r[1] - r[0]) * ph }
    function fmtY(v) {
        const d = Math.abs(v) < 10 && decimals === 0 && v % 1 !== 0 ? 1 : decimals
        return Number(v).toLocaleString(Qt.locale(), "f", d)
    }
    function fmtX(x) {
        if (x > -1) return "now"
        const m = -x / 60
        return m >= 1 ? "−" + Math.round(m) + " min" : "−" + Math.round(-x) + " s"
    }

    onSeriesChanged: canvas.requestPaint()
    onXsChanged: canvas.requestPaint()
    onRefsChanged: canvas.requestPaint()
    onHoverIdxChanged: canvas.requestPaint()
    onWidthChanged: canvas.requestPaint()
    onHeightChanged: canvas.requestPaint()
    Connections { target: chart.th; function onDarkChanged() { canvas.requestPaint() } }

    Canvas {
        id: canvas
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative
        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            const th = chart.th
            const r = chart.yRange()
            const font = chart.th.small
            ctx.font = font + "pt sans-serif"
            // grid + y labels
            const step = chart.nice((r[1] - r[0]) / 4)
            ctx.lineWidth = 1
            ctx.textBaseline = "middle"
            ctx.textAlign = "right"
            for (let v = Math.ceil(r[0] / step) * step; v <= r[1] + 1e-9; v += step) {
                const y = Math.round(chart.py(v, r)) + 0.5
                ctx.strokeStyle = Math.abs(v) < 1e-9 ? th.axis : th.grid
                ctx.beginPath(); ctx.moveTo(chart.padL, y); ctx.lineTo(chart.padL + chart.pw, y); ctx.stroke()
                ctx.fillStyle = th.muted
                ctx.fillText(chart.fmtY(v), chart.padL - 6, y)
            }
            // x labels
            ctx.textAlign = "center"
            ctx.textBaseline = "top"
            const span = chart.xMax - chart.xMin
            const xstep = span > 1200 ? 600 : span > 600 ? 300 : span > 240 ? 120 : 60
            for (let x = chart.xMax; x >= chart.xMin - 1e-9; x -= xstep) {
                const X = chart.px(x)
                ctx.fillStyle = th.muted
                ctx.textAlign = x === chart.xMax ? "right" : (x - xstep < chart.xMin ? "left" : "center")
                ctx.fillText(chart.fmtX(x), X, chart.padT + chart.ph + 4)
            }
            // reference lines
            for (let ref of chart.refs) {
                if (ref.v === null || ref.v === undefined) continue
                const y = Math.round(chart.py(ref.v, r)) + 0.5
                ctx.strokeStyle = th.axis
                ctx.setLineDash([4, 4])
                ctx.beginPath(); ctx.moveTo(chart.padL, y); ctx.lineTo(chart.padL + chart.pw, y); ctx.stroke()
                ctx.setLineDash([])
                if (ref.label) {
                    // alternate ends so close reference lines don't print over each other
                    const right = chart.refs.indexOf(ref) % 2 === 1
                    ctx.fillStyle = th.muted
                    ctx.textAlign = right ? "right" : "left"; ctx.textBaseline = "bottom"
                    ctx.fillText(ref.label, right ? chart.padL + chart.pw - 4 : chart.padL + 4, y - 2)
                }
            }
            if (!chart.hasData()) {
                ctx.fillStyle = th.muted
                ctx.textAlign = "center"; ctx.textBaseline = "middle"
                ctx.fillText(chart.emptyText, chart.padL + chart.pw / 2, chart.padT + chart.ph / 2)
                return
            }
            ctx.save()
            ctx.beginPath(); ctx.rect(chart.padL, chart.padT - 2, chart.pw, chart.ph + 4); ctx.clip()
            const xs = chart.xs
            for (let si = 0; si < chart.series.length; si++) {
                const s = chart.series[si]
                const vals = s.values || []
                // area wash for a single series
                if (chart.series.length === 1) {
                    ctx.fillStyle = s.color
                    ctx.globalAlpha = 0.10
                    let open = false, lastX = 0
                    ctx.beginPath()
                    for (let i = 0; i < vals.length; i++) {
                        const v = vals[i]
                        if (v === null || v === undefined) {
                            if (open) { ctx.lineTo(lastX, chart.py(r[0], r)); ctx.closePath(); open = false }
                            continue
                        }
                        const X = chart.px(xs[i]), Y = chart.py(v, r)
                        if (!open) { ctx.moveTo(X, chart.py(r[0], r)); ctx.lineTo(X, Y); open = true } else ctx.lineTo(X, Y)
                        lastX = X
                    }
                    if (open) { ctx.lineTo(lastX, chart.py(r[0], r)); ctx.closePath() }
                    ctx.fill()
                    ctx.globalAlpha = 1
                }
                ctx.strokeStyle = s.color
                ctx.lineWidth = 2
                ctx.lineJoin = "round"; ctx.lineCap = "round"
                ctx.beginPath()
                let pen = false
                for (let i = 0; i < vals.length; i++) {
                    const v = vals[i]
                    if (v === null || v === undefined) { pen = false; continue }
                    const X = chart.px(xs[i]), Y = chart.py(v, r)
                    if (!pen) { ctx.moveTo(X, Y); pen = true } else ctx.lineTo(X, Y)
                }
                ctx.stroke()
                // end dot with a surface ring
                for (let i = vals.length - 1; i >= 0; i--) {
                    if (vals[i] === null || vals[i] === undefined) continue
                    const X = chart.px(xs[i]), Y = chart.py(vals[i], r)
                    ctx.fillStyle = th.surface; ctx.beginPath(); ctx.arc(X, Y, 6, 0, 2 * Math.PI); ctx.fill()
                    ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X, Y, 4, 0, 2 * Math.PI); ctx.fill()
                    break
                }
            }
            ctx.restore()
            // hover crosshair
            const hi = chart.hoverIdx
            if (hi >= 0 && hi < xs.length) {
                const X = Math.round(chart.px(xs[hi])) + 0.5
                ctx.strokeStyle = th.axis; ctx.lineWidth = 1
                ctx.beginPath(); ctx.moveTo(X, chart.padT); ctx.lineTo(X, chart.padT + chart.ph); ctx.stroke()
                for (let s of chart.series) {
                    const v = (s.values || [])[hi]
                    if (v === null || v === undefined) continue
                    const Y = chart.py(v, r)
                    ctx.fillStyle = th.surface; ctx.beginPath(); ctx.arc(X, Y, 6, 0, 2 * Math.PI); ctx.fill()
                    ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X, Y, 4, 0, 2 * Math.PI); ctx.fill()
                }
            }
        }
    }

    MouseArea {
        id: mouse
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        onExited: chart.hoverIdx = -1
        onPositionChanged: function (e) {
            const xs = chart.xs
            if (!xs || xs.length === 0 || e.x < chart.padL) { chart.hoverIdx = -1; return }
            const x = chart.xMin + (e.x - chart.padL) / chart.pw * (chart.xMax - chart.xMin)
            let best = -1, bd = Infinity
            for (let i = 0; i < xs.length; i++) { const d = Math.abs(xs[i] - x); if (d < bd) { bd = d; best = i } }
            chart.hoverIdx = best
        }
    }

    Rectangle {
        id: tip
        visible: chart.hoverIdx >= 0 && chart.hoverIdx < chart.xs.length
        color: chart.th.surface
        border.color: chart.th.border
        radius: 6
        z: 10
        width: tipCol.implicitWidth + 16
        height: tipCol.implicitHeight + 12
        x: {
            if (!visible) return 0
            const X = chart.px(chart.xs[chart.hoverIdx])
            return X + width + 12 > chart.width ? X - width - 10 : X + 10
        }
        y: chart.padT
        Column {
            id: tipCol
            x: 8; y: 6
            spacing: 2
            QQC2.Label {
                text: tip.visible ? chart.fmtX(chart.xs[chart.hoverIdx]) : ""
                color: chart.th.muted
                font.pointSize: chart.th.small
            }
            Repeater {
                model: tip.visible ? chart.series : []
                Row {
                    spacing: 6
                    readonly property var v: (modelData.values || [])[chart.hoverIdx]
                    Rectangle { width: 10; height: 3; radius: 1.5; color: modelData.color; anchors.verticalCenter: parent.verticalCenter }
                    QQC2.Label {
                        text: (chart.series.length > 1 ? modelData.name + "  " : "")
                              + (v === null || v === undefined ? "—" : Number(v).toLocaleString(Qt.locale(), "f", chart.decimals) + (chart.unit ? " " + chart.unit : ""))
                        color: chart.th.text
                        font.pointSize: chart.th.small
                    }
                }
            }
        }
    }
}
