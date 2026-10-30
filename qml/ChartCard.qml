// A card: title, optional subtitle and legend, then the content (a chart, tiles, a table...).
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Rectangle {
    id: card
    property var th
    property string title: ""
    property string subtitle: ""
    // [{name, color}] - shown for two or more series
    property var legend: []
    default property alias content: body.data
    // optional colour key for the entity this card is about
    property color swatch: "transparent"
    // true: the content stretches to the card's height (a list); false: it sits at the top
    property bool fillBody: false

    color: th.surface
    radius: 8
    border.color: th.border
    border.width: 1
    implicitHeight: col.implicitHeight + 2 * Kirigami.Units.largeSpacing
    Layout.fillWidth: true

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing
        spacing: Kirigami.Units.smallSpacing
        RowLayout {
            Layout.fillWidth: true
            visible: card.title.length > 0
            spacing: Kirigami.Units.largeSpacing
            ColumnLayout {
                spacing: 0
                Layout.fillWidth: true
                QQC2.Label { text: card.title; font.weight: Font.DemiBold; color: card.th.text; elide: Text.ElideRight; Layout.fillWidth: true }
                QQC2.Label {
                    visible: card.subtitle.length > 0
                    text: card.subtitle; color: card.th.muted; font.pointSize: card.th.small
                    elide: Text.ElideRight; Layout.fillWidth: true
                }
            }
            Repeater {
                model: card.legend.length >= 2 ? card.legend.length : 0
                RowLayout {
                    readonly property var lg: card.legend[index] || ({})
                    spacing: 4
                    Rectangle { width: 12; height: 3; radius: 1.5; color: lg.color; Layout.alignment: Qt.AlignVCenter }
                    QQC2.Label { text: lg.name; color: card.th.text2; font.pointSize: card.th.small }
                }
            }
            Rectangle { visible: card.swatch.a > 0; width: 14; height: 4; radius: 2; color: card.swatch; Layout.alignment: Qt.AlignVCenter }
        }
        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: card.fillBody
            spacing: Kirigami.Units.smallSpacing
        }
        Item { visible: !card.fillBody; Layout.fillHeight: true }
    }
}
