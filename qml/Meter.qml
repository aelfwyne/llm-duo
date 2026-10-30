// Thin meter: label + value text above a bar. The fill turns to a status colour past the
// thresholds; the track is a faint wash of the same colour.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

ColumnLayout {
    id: m
    property var th
    property string label: ""
    property string text: ""
    property real value: 0          // 0..1
    property color color: th ? th.s(0) : "steelblue"
    property real warnAt: 2         // > 1 = never
    property real critAt: 2
    property real marker: -1        // 0..1: a tick on the track (e.g. the power cap when the scale goes past it)
    readonly property color fill: value >= critAt ? th.critical : value >= warnAt ? th.warning : color
    spacing: 3
    Layout.fillWidth: true

    RowLayout {
        Layout.fillWidth: true
        QQC2.Label { text: m.label; color: m.th.text2; font.pointSize: m.th.small }
        Item { Layout.fillWidth: true }
        QQC2.Label { text: m.text; color: m.th.text; font.pointSize: m.th.small }
    }
    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 6
        radius: 3
        color: Qt.rgba(m.fill.r, m.fill.g, m.fill.b, 0.16)
        Rectangle {
            width: parent.width * Math.max(0, Math.min(1, m.value))
            height: parent.height
            radius: 3
            color: m.fill
            Behavior on width { NumberAnimation { duration: 400; easing.type: Easing.OutCubic } }
        }
        Rectangle {
            visible: m.marker > 0 && m.marker < 1
            x: parent.width * m.marker - 1
            y: -3
            width: 2
            height: parent.height + 6
            radius: 1
            color: m.th.text2
        }
    }
}
