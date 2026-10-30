// Colours for the dashboards: chrome follows the Plasma colour scheme, data colours come from
// a fixed, colour-vision-checked palette (separate steps for light and dark schemes).
// Identity is fixed per entity: the first GPU is always slot 1 (blue), the second slot 2 (orange).
import QtQuick
import org.kde.kirigami as Kirigami

QtObject {
    id: pal
    // luminance of the window background decides which steps to use
    readonly property color bg: Kirigami.Theme.backgroundColor
    readonly property bool dark: (0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b) < 0.5

    readonly property color plane: bg                                     // window behind the cards
    readonly property color surface: dark ? Qt.lighter(bg, 1.28) : Qt.lighter(bg, 1.045) // card surface
    readonly property color text: Kirigami.Theme.textColor
    readonly property color text2: Qt.rgba(text.r, text.g, text.b, 0.72)
    readonly property color muted: Qt.rgba(text.r, text.g, text.b, 0.50)
    readonly property color grid: Qt.rgba(text.r, text.g, text.b, dark ? 0.10 : 0.09)
    readonly property color axis: Qt.rgba(text.r, text.g, text.b, dark ? 0.22 : 0.25)
    readonly property color border: Qt.rgba(text.r, text.g, text.b, dark ? 0.12 : 0.10)

    readonly property var series: dark
        ? ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
        : ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    function s(i) { return series[i % series.length] }

    // status colours are reserved for state and always come with an icon + label
    readonly property color good: "#0ca30c"
    readonly property color warning: "#fab219"
    readonly property color serious: "#ec835a"
    readonly property color critical: "#d03b3b"
    function status(level) {
        switch (level) {
        case "good": return good
        case "warning": return warning
        case "serious": return serious
        case "critical": return critical
        default: return Kirigami.Theme.linkColor
        }
    }
    function statusIcon(level) {
        switch (level) {
        case "good": return "✓"
        case "warning": return "!"
        case "serious": return "!!"
        case "critical": return "✕"
        default: return "i"
        }
    }

    // font sizes (points) - from the application font, which follows the Plasma setting
    readonly property real base: Qt.application.font.pointSize > 0 ? Qt.application.font.pointSize : 10
    readonly property real small: Math.max(7, base - 1.5)

    // number formatting
    function fmt(v, d) {
        if (v === null || v === undefined || isNaN(v)) return "—"
        return Number(v).toLocaleString(Qt.locale(), "f", d === undefined ? 0 : d)
    }
    function compact(v) {
        if (v === null || v === undefined || isNaN(v)) return "—"
        const a = Math.abs(v)
        if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + "B"
        if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M"
        if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "K"
        return Math.round(v).toLocaleString(Qt.locale(), "f", 0)
    }
    function dur(secs) {
        if (secs === null || secs === undefined || isNaN(secs)) return "—"
        secs = Math.round(secs)
        const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60), s = secs % 60
        if (h > 0) return h + " h " + m + " min"
        if (m > 0) return m + " min " + s + " s"
        return s + " s"
    }
}
