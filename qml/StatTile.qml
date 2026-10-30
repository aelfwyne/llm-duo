// A stat tile: label, big value with unit, a secondary line and an optional sparkline.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Rectangle {
    id: tile
    property var th
    property string label: ""
    property string value: "—"
    property string unit: ""
    property string sub: ""
    property bool hero: false
    property var spark: []           // recent values, oldest first
    property color sparkColor: th ? th.s(0) : "steelblue"
    property string level: ""        // optional status: good / warning / serious / critical

    color: th.surface
    radius: 8
    border.color: th.border
    border.width: 1
    Layout.fillWidth: true
    Layout.preferredWidth: Kirigami.Units.gridUnit * (hero ? 12 : 9)
    Layout.minimumWidth: Kirigami.Units.gridUnit * 7
    implicitHeight: col.implicitHeight + 2 * Kirigami.Units.largeSpacing

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing
        spacing: 2
        RowLayout {
            spacing: 6
            QQC2.Label { text: tile.label; color: tile.th.text2; elide: Text.ElideRight; Layout.fillWidth: true }
            Rectangle {
                visible: tile.level.length > 0
                width: badge.implicitWidth + 10; height: badge.implicitHeight + 2; radius: height / 2
                color: tile.th.status(tile.level)
                QQC2.Label {
                    id: badge
                    anchors.centerIn: parent
                    text: tile.th.statusIcon(tile.level)
                    color: "#111"
                    font.pointSize: tile.th.small
                    font.weight: Font.Bold
                }
            }
        }
        RowLayout {
            spacing: 4
            QQC2.Label {
                text: tile.value
                color: tile.th.text
                font.pointSize: Math.round(tile.th.base * (tile.hero ? 3.0 : 1.9))
                font.weight: Font.DemiBold
            }
            QQC2.Label {
                visible: tile.unit.length > 0
                text: tile.unit
                color: tile.th.muted
                Layout.alignment: Qt.AlignBaseline
            }
            Item { Layout.fillWidth: true }
            Canvas {
                id: spark
                visible: tile.spark.length >= 2
                Layout.preferredWidth: Kirigami.Units.gridUnit * 4
                Layout.preferredHeight: Kirigami.Units.gridUnit * 1.6
                Layout.alignment: Qt.AlignBottom
                onPaint: {
                    const ctx = getContext("2d")
                    ctx.reset()
                    const v = tile.spark.filter(x => x !== null && x !== undefined)
                    if (v.length < 2) return
                    const lo = Math.min(...v), hi = Math.max(...v)
                    const span = hi - lo || 1
                    ctx.strokeStyle = tile.sparkColor
                    ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round"
                    ctx.beginPath()
                    for (let i = 0; i < v.length; i++) {
                        const x = 3 + i / (v.length - 1) * (width - 6)
                        const y = height - 3 - (v[i] - lo) / span * (height - 6)
                        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
                    }
                    ctx.stroke()
                    const x = width - 3, y = height - 3 - (v[v.length - 1] - lo) / span * (height - 6)
                    ctx.fillStyle = tile.sparkColor; ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill()
                }
                Connections { target: tile; function onSparkChanged() { spark.requestPaint() } }
            }
        }
        QQC2.Label {
            visible: tile.sub.length > 0
            text: tile.sub
            color: tile.th.muted
            font.pointSize: tile.th.small
            wrapMode: Text.Wrap
            maximumLineCount: 2
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
    }
}
