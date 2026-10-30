// Scatter plot (one series): points [{x, y, tip}], e.g. generation speed against context depth.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Item {
    id: chart
    property var th
    property var points: []
    property color color: th ? th.s(0) : "steelblue"
    property string xUnit: ""
    property string emptyText: "No data yet"
    property int hoverIdx: -1

    implicitHeight: Kirigami.Units.gridUnit * 9
    Layout.fillWidth: true

    readonly property real padL: Kirigami.Units.gridUnit * 2.6
    readonly property real padR: Kirigami.Units.gridUnit * 0.8
    readonly property real padT: Kirigami.Units.smallSpacing * 2
    readonly property real padB: Kirigami.Units.gridUnit * 1.2
    readonly property real pw: Math.max(10, width - padL - padR)
    readonly property real ph: Math.max(10, height - padT - padB)

    function nice(x) {
        if (x <= 0) return 1
        const e = Math.pow(10, Math.floor(Math.log10(x)))
        const f = x / e
        return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * e
    }
    function ranges() {
        let xh = 0, yh = 0
        for (let p of points) { xh = Math.max(xh, p.x); yh = Math.max(yh, p.y) }
        const xs = nice(Math.max(xh, 1) / 4), ys = nice(Math.max(yh, 1) / 4)
        return [Math.ceil(Math.max(xh, 1) * 1.05 / xs) * xs, Math.ceil(Math.max(yh, 1) * 1.08 / ys) * ys, xs, ys]
    }
    function px(x, r) { return padL + x / r[0] * pw }
    function py(y, r) { return padT + ph - y / r[1] * ph }
    function compact(v) { return v >= 1000 ? (v / 1000).toLocaleString(Qt.locale(), "f", v % 1000 === 0 ? 0 : 1) + "K" : String(v) }

    onPointsChanged: canvas.requestPaint()
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
            const r = chart.ranges()
            ctx.font = chart.th.small + "pt sans-serif"
            ctx.lineWidth = 1
            ctx.textAlign = "right"; ctx.textBaseline = "middle"
            for (let v = 0; v <= r[1] + 1e-9; v += r[3]) {
                const y = Math.round(chart.py(v, r)) + 0.5
                ctx.strokeStyle = v === 0 ? th.axis : th.grid
                ctx.beginPath(); ctx.moveTo(chart.padL, y); ctx.lineTo(chart.padL + chart.pw, y); ctx.stroke()
                ctx.fillStyle = th.muted
                ctx.fillText(Number(v).toLocaleString(Qt.locale(), "f", v % 1 === 0 ? 0 : 1), chart.padL - 6, y)
            }
            ctx.textAlign = "center"; ctx.textBaseline = "top"
            for (let v = 0; v <= r[0] + 1e-9; v += r[2]) {
                const X = chart.px(v, r)
                ctx.fillStyle = th.muted
                ctx.textAlign = v === 0 ? "left" : (v + r[2] > r[0] ? "right" : "center")
                ctx.fillText(chart.compact(v) + (v + r[2] > r[0] && chart.xUnit ? " " + chart.xUnit : ""), X, chart.padT + chart.ph + 4)
            }
            if (chart.points.length === 0) {
                ctx.textAlign = "center"; ctx.textBaseline = "middle"
                ctx.fillText(chart.emptyText, chart.padL + chart.pw / 2, chart.padT + chart.ph / 2)
                return
            }
            for (let i = 0; i < chart.points.length; i++) {
                const p = chart.points[i]
                const X = chart.px(p.x, r), Y = chart.py(p.y, r)
                const last = i === chart.points.length - 1
                ctx.fillStyle = th.surface; ctx.beginPath(); ctx.arc(X, Y, 6, 0, 2 * Math.PI); ctx.fill()
                ctx.fillStyle = p.color || chart.color
                ctx.globalAlpha = chart.hoverIdx >= 0 && chart.hoverIdx !== i ? 0.45 : (last ? 1 : 0.8)
                ctx.beginPath(); ctx.arc(X, Y, last || chart.hoverIdx === i ? 5 : 4, 0, 2 * Math.PI); ctx.fill()
                ctx.globalAlpha = 1
            }
        }
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        onExited: chart.hoverIdx = -1
        onPositionChanged: function (e) {
            const r = chart.ranges()
            let best = -1, bd = 14 * 14
            for (let i = 0; i < chart.points.length; i++) {
                const dx = chart.px(chart.points[i].x, r) - e.x, dy = chart.py(chart.points[i].y, r) - e.y
                const d = dx * dx + dy * dy
                if (d < bd) { bd = d; best = i }
            }
            chart.hoverIdx = best
        }
    }

    Rectangle {
        id: tip
        visible: chart.hoverIdx >= 0 && chart.hoverIdx < chart.points.length
        color: chart.th.surface
        border.color: chart.th.border
        radius: 6
        z: 10
        width: tipText.implicitWidth + 16
        height: tipText.implicitHeight + 12
        x: {
            if (!visible) return 0
            const X = chart.px(chart.points[chart.hoverIdx].x, chart.ranges())
            return X + width + 14 > chart.width ? X - width - 12 : X + 12
        }
        y: chart.padT
        QQC2.Label {
            id: tipText
            x: 8; y: 6
            color: chart.th.text
            font.pointSize: chart.th.small
            text: tip.visible ? (chart.points[chart.hoverIdx].tip || "") : ""
        }
    }
}
