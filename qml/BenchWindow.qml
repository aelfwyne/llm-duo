// BetterBench window: options for `betterbench run` against a pane's running model, the run's
// output, and links to its report. The run keeps going when the window is hidden.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo: BetterBench"
    width: Kirigami.Units.gridUnit * 50
    height: Kirigami.Units.gridUnit * 44

    property var help: ({})
    property var b: ({})
    property bool loaded: false
    function h(k) { return help[k] || "" }
    function call(action, arg) {
        Backend.call("bench", action, JSON.stringify(arg || {}))
    }
    function appendLog(text) {
        logArea.append(text)
        logArea.cursorPosition = logArea.length
    }
    function clearLog() { logArea.clear() }

    // ---------------------------------------------------------------- form
    property var catChecks: ({})
    function loadForm(o) {
        if (!o) return
        cmdField.text = o.command || ""
        quickBox.checked = !!o.quick
        passesSpin.value = Number(o.passes) || 20
        warmupSpin.value = Number(o.warmup) || 0
        decodeBox.checked = !!o.decode
        prefillBox.checked = !!o.prefill
        concBox.checked = !!o.concurrency
        const cats = o.categories || []
        const c = {}
        for (const k of (b.categories || [])) c[k] = cats.indexOf(k) >= 0
        catChecks = c
        greedyBox.checked = !!o.greedy
        seedField.text = String(o.seed ?? "")
        mmlField.text = String(o.maxModelLen ?? "")
        configField.text = o.config || ""
        nameField.text = o.name || ""
        notesField.text = o.notes || ""
        duoNotesBox.checked = !!o.duoNotes
        htmlBox.checked = !!o.html
        openBox.checked = !!o.openReport
        updateBox.checked = !!o.updateCheck
        extraField.text = o.extra || ""
    }
    function collectForm() {
        const cats = []
        for (const k of (b.categories || [])) if (catChecks[k]) cats.push(k)
        return {
            command: cmdField.text, quick: quickBox.checked, passes: passesSpin.value, warmup: warmupSpin.value,
            decode: decodeBox.checked, prefill: prefillBox.checked, concurrency: concBox.checked,
            categories: cats, greedy: greedyBox.checked, seed: seedField.text, maxModelLen: mmlField.text,
            config: configField.text, name: nameField.text, notes: notesField.text, duoNotes: duoNotesBox.checked,
            html: htmlBox.checked, openReport: openBox.checked, updateCheck: updateBox.checked, extra: extraField.text,
        }
    }

    visible: !!b.visible
    onVisibleChanged: if (visible && !loaded) { loadForm(b.opts); loaded = true }
    onClosing: function (close) {
        close.accepted = false
        Qt.callLater(function () { win.call("close") })
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Kirigami.Units.largeSpacing

        QQC2.Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            textFormat: Text.StyledText
            text: "<b>" + (win.b.model || "(no model)") + "</b>  on the " + (win.b.side || "") + " pane  ("
                + (win.b.backend || "") + (win.b.gpu ? ", " + win.b.gpu : "") + ")<br>" + (win.b.endpoint || "")
                + (win.b.ready || win.b.running ? "" : "<br><i>No model is ready on this pane - load one first.</i>")
        }
        Kirigami.InlineMessage {
            Layout.fillWidth: true
            visible: !!win.b.missing
            type: Kirigami.MessageType.Error
            text: (win.b.missing || "").replace(/\n/g, "<br>")
            onLinkActivated: function (link) { Qt.openUrlExternally(link) }
            actions: [
                Kirigami.Action {
                    text: "Open BetterBench on GitHub"
                    icon.name: "internet-web-browser"
                    onTriggered: Qt.openUrlExternally(win.b.url)
                }
            ]
        }

        QQC2.ScrollView {
            id: formScroll
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(form.implicitHeight, win.height * 0.55)
            contentWidth: availableWidth
            enabled: !win.b.running

            GridLayout {
                id: form
                width: formScroll.availableWidth
                columns: 2

                OptLabel { text: "Run length"; help: win.h("bb_length") }
                RowLayout {
                    QQC2.CheckBox { id: quickBox; text: "Quick smoke run"; HelpTip { text: win.h("bb_quick"); shown: quickBox.hovered } }
                    QQC2.Label { text: "Passes"; enabled: !quickBox.checked }
                    QQC2.SpinBox { id: passesSpin; from: 1; to: 1000; editable: true; enabled: !quickBox.checked
                        HelpTip { text: win.h("bb_passes"); shown: passesSpin.hovered } }
                    QQC2.Label { text: "Warmup"; enabled: !quickBox.checked }
                    QQC2.SpinBox { id: warmupSpin; from: 0; to: 100; editable: true; enabled: !quickBox.checked
                        HelpTip { text: win.h("bb_warmup"); shown: warmupSpin.hovered } }
                }

                OptLabel { text: "Phases"; help: win.h("bb_phases") }
                RowLayout {
                    QQC2.CheckBox { id: decodeBox; text: "Decode (single stream)" }
                    QQC2.CheckBox { id: prefillBox; text: "Prefill depth sweep" }
                    QQC2.CheckBox { id: concBox; text: "Concurrency sweep" }
                }

                OptLabel { text: "Prompt categories"; help: win.h("bb_categories"); Layout.alignment: Qt.AlignTop }
                Flow {
                    Layout.fillWidth: true
                    spacing: Kirigami.Units.smallSpacing
                    enabled: decodeBox.checked || concBox.checked
                    Repeater {
                        model: win.b.categories || []
                        QQC2.CheckBox {
                            required property string modelData
                            text: modelData
                            checked: !!win.catChecks[modelData]
                            onToggled: {
                                const c = Object.assign({}, win.catChecks)
                                c[modelData] = checked
                                win.catChecks = c
                            }
                        }
                    }
                }

                OptLabel { text: "Sampling"; help: win.h("bb_sampling") }
                RowLayout {
                    QQC2.CheckBox { id: greedyBox; text: "Greedy (temperature 0)" }
                    QQC2.Label { text: "Seed" }
                    QQC2.TextField { id: seedField; placeholderText: "none"; Layout.preferredWidth: Kirigami.Units.gridUnit * 6
                        validator: RegularExpressionValidator { regularExpression: /-?[0-9]*/ } }
                }

                OptLabel { text: "Max model length"; help: win.h("bb_mml") }
                QQC2.TextField { id: mmlField; placeholderText: "auto-detect from /v1/models"; Layout.fillWidth: true
                    validator: RegularExpressionValidator { regularExpression: /[0-9]*/ } }

                OptLabel { text: "Config file"; help: win.h("bb_config") }
                PathField { id: configField; browse: "file"; help: win.h("bb_config"); Layout.fillWidth: true }

                OptLabel { text: "Run name"; help: win.h("bb_name") }
                QQC2.TextField { id: nameField; placeholderText: "the model name"; Layout.fillWidth: true }

                OptLabel { text: "Notes"; help: win.h("bb_notes") }
                ColumnLayout {
                    Layout.fillWidth: true
                    QQC2.TextField { id: notesField; placeholderText: "KEY=VALUE KEY=VALUE …"; Layout.fillWidth: true }
                    QQC2.CheckBox { id: duoNotesBox; text: "Also record backend, GPU and model file"
                        HelpTip { text: win.h("bb_duo_notes"); shown: duoNotesBox.hovered } }
                }

                OptLabel { text: "Report" }
                RowLayout {
                    QQC2.CheckBox { id: htmlBox; text: "Write HTML report" }
                    QQC2.CheckBox { id: openBox; text: "Open it when done"; enabled: htmlBox.checked }
                    QQC2.CheckBox { id: updateBox; text: "Check for BetterBench updates" }
                }

                OptLabel { text: "Extra args"; help: win.h("bb_extra") }
                QQC2.TextField { id: extraField; Layout.fillWidth: true }

                OptLabel { text: "BetterBench command"; help: win.h("bb_command") }
                QQC2.TextField { id: cmdField; placeholderText: "betterbench"; Layout.fillWidth: true }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            QQC2.Button {
                text: "Start"
                icon.name: "media-playback-start"
                enabled: !win.b.running && !!win.b.ready
                onClicked: win.call("start", { opts: win.collectForm() })
            }
            QQC2.Button { text: "Stop"; icon.name: "media-playback-stop"; enabled: !!win.b.running; onClicked: win.call("stop") }
            QQC2.Button {
                id: defBtn
                text: "Defaults"
                icon.name: "edit-undo"
                enabled: !win.b.running
                onClicked: win.loadForm(win.b.defaults)
                HelpTip { text: "Put every option back to BetterBench's defaults (saved when you next press Start)"; shown: defBtn.hovered }
            }
            Item { Layout.fillWidth: true }
            QQC2.Button { text: "Open report"; icon.name: "document-preview"; enabled: !!win.b.report; onClicked: win.call("openReport") }
            QQC2.Button { text: "Runs folder"; icon.name: "folder-open"; onClicked: win.call("openRuns") }
            QQC2.Button { text: "Hide"; onClicked: win.call("hide") }
        }
        QQC2.Label { Layout.fillWidth: true; text: win.b.status || "Idle"; elide: Text.ElideRight }
        QQC2.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: Kirigami.Units.gridUnit * 6
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
