// Column chart over a sequence (e.g. one column per request), drawn on a Canvas.
// values may hold nulls (drawn as a short muted stub, e.g. an aborted request).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Item {
    id: chart
    property var th
    property var values: []
    property color color: th ? th.s(0) : "steelblue"
    property var colors: []          // optional colour per column (e.g. which server)
    property var tips: []            // tooltip text per column
    property var refs: []            // [{v, label}]
    property string unit: ""
    property int decimals: 1
    property string emptyText: "No requests yet"
    property string xLabelFirst: ""
    property string xLabelLast: ""
    property int hoverIdx: -1

    implicitHeight: Kirigami.Units.gridUnit * 9
    Layout.fillWidth: true

    readonly property real padL: Kirigami.Units.gridUnit * 2.6
    readonly property real padR: Kirigami.Units.smallSpacing * 2
    readonly property real padT: Kirigami.Units.smallSpacing * 2
    readonly property real padB: Kirigami.Units.gridUnit * 1.2
    readonly property real pw: Math.max(10, width - padL - padR)
    readonly property real ph: Math.max(10, height - padT - padB)
    readonly property real slot: values.length > 0 ? pw / values.length : pw
    readonly property real barW: Math.max(1, Math.min(24, slot > 6 ? slot - 2 : slot * 0.8))

    function nice(x) {
        if (x <= 0) return 1
        const e = Math.pow(10, Math.floor(Math.log10(x)))
        const f = x / e
        return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * e
    }
    function yTop() {
        let hi = 0
        for (let v of values) if (v !== null && v !== undefined) hi = Math.max(hi, v)
        for (let r of refs) if (r.v) hi = Math.max(hi, r.v)
        const step = nice(Math.max(hi, 1) / 4)
        return Math.ceil(Math.max(hi, 1) * 1.05 / step) * step
    }
    function py(v, t) { return padT + ph - v / t * ph }
    function cx(i) { return padL + (i + 0.5) * slot }

    onValuesChanged: canvas.requestPaint()
    onRefsChanged: canvas.requestPaint()
    onHoverIdxChanged: canvas.requestPaint()
    onWidthChanged: canvas.requestPaint()
    onHeightChanged: canvas.requestPaint()

    Canvas {
        id: canvas
        anchors.fill: parent
        renderStrategy: Canvas.Cooperative
        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            const th = chart.th
            const t = chart.yTop()
            ctx.font = chart.th.small + "pt sans-serif"
            const step = chart.nice(t / 4)
            ctx.lineWidth = 1
            ctx.textAlign = "right"; ctx.textBaseline = "middle"
            for (let v = 0; v <= t + 1e-9; v += step) {
                const y = Math.round(chart.py(v, t)) + 0.5
                ctx.strokeStyle = v === 0 ? th.axis : th.grid
                ctx.beginPath(); ctx.moveTo(chart.padL, y); ctx.lineTo(chart.padL + chart.pw, y); ctx.stroke()
                ctx.fillStyle = th.muted
                ctx.fillText(Number(v).toLocaleString(Qt.locale(), "f", v % 1 === 0 ? 0 : 1), chart.padL - 6, y)
            }
            ctx.textBaseline = "top"
            ctx.fillStyle = th.muted
            ctx.textAlign = "left"; ctx.fillText(chart.xLabelFirst, chart.padL, chart.padT + chart.ph + 4)
            ctx.textAlign = "right"; ctx.fillText(chart.xLabelLast, chart.padL + chart.pw, chart.padT + chart.ph + 4)
            if (chart.values.length === 0) {
                ctx.textAlign = "center"; ctx.textBaseline = "middle"
                ctx.fillText(chart.emptyText, chart.padL + chart.pw / 2, chart.padT + chart.ph / 2)
                return
            }
            const base = chart.py(0, t)
            for (let i = 0; i < chart.values.length; i++) {
                const v = chart.values[i]
                const w = chart.barW
                const x = chart.cx(i) - w / 2
                if (v === null || v === undefined) {
                    ctx.fillStyle = th.axis
                    ctx.fillRect(x, base - 3, w, 3)
                    continue
                }
                const y = chart.py(v, t)
                const rr = Math.min(4, w / 2, (base - y))
                ctx.fillStyle = chart.colors[i] || chart.color
                ctx.globalAlpha = chart.hoverIdx >= 0 && chart.hoverIdx !== i ? 0.55 : 1
                ctx.beginPath()
                ctx.moveTo(x, base)
                ctx.lineTo(x, y + rr)
                ctx.arcTo(x, y, x + rr, y, rr)
                ctx.lineTo(x + w - rr, y)
                ctx.arcTo(x + w, y, x + w, y + rr, rr)
                ctx.lineTo(x + w, base)
                ctx.closePath()
                ctx.fill()
                ctx.globalAlpha = 1
            }
            for (let ref of chart.refs) {
                if (!ref.v) continue
                const y = Math.round(chart.py(ref.v, t)) + 0.5
                ctx.strokeStyle = th.text2
                ctx.setLineDash([4, 4])
                ctx.beginPath(); ctx.moveTo(chart.padL, y); ctx.lineTo(chart.padL + chart.pw, y); ctx.stroke()
                ctx.setLineDash([])
                if (ref.label) {
                    // labels alternate ends so two reference lines never overlap
                    const left = chart.refs.indexOf(ref) % 2 === 1
                    const tw = ctx.measureText(ref.label).width
                    const lx = left ? chart.padL + 2 : chart.padL + chart.pw - 2 - tw
                    ctx.fillStyle = th.surface
                    ctx.fillRect(lx - 2, y - 2 - chart.th.small * 1.5, tw + 4, chart.th.small * 1.4)
                    ctx.fillStyle = th.text2
                    ctx.textAlign = "left"; ctx.textBaseline = "bottom"
                    ctx.fillText(ref.label, lx, y - 2)
                }
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        onExited: chart.hoverIdx = -1
        onPositionChanged: function (e) {
            if (chart.values.length === 0 || e.x < chart.padL) { chart.hoverIdx = -1; return }
            chart.hoverIdx = Math.max(0, Math.min(chart.values.length - 1, Math.floor((e.x - chart.padL) / chart.slot)))
        }
    }

    Rectangle {
        id: tip
        visible: chart.hoverIdx >= 0 && chart.hoverIdx < chart.values.length
        color: chart.th.surface
        border.color: chart.th.border
        radius: 6
        z: 10
        width: tipText.implicitWidth + 16
        height: tipText.implicitHeight + 12
        x: {
            if (!visible) return 0
            const X = chart.cx(chart.hoverIdx)
            return X + width + 14 > chart.width ? X - width - 12 : X + 12
        }
        y: chart.padT
        QQC2.Label {
            id: tipText
            x: 8; y: 6
            color: chart.th.text
            font.pointSize: chart.th.small
            text: tip.visible ? (chart.tips[chart.hoverIdx] || "") : ""
        }
    }
}
