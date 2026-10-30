// Settings: backends per pane, binaries and folders, ports, GPU assignment.
import QtQuick
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import org.kde.kirigami as Kirigami
import io.github.llmduo

QQC2.ApplicationWindow {
    id: win
    title: "LLM Duo settings"
    width: Kirigami.Units.gridUnit * 50
    height: Kirigami.Units.gridUnit * 54
    visible: false
    modality: Qt.ApplicationModal

    property var help: ({})
    property var sdata: ({})
    property var deviceLabels: []
    property var backendLabels: []
    property string experimentalLabel: "vLLM (Experimental)"

    function h(k) { return help[k] || "" }

    function load(d) {
        sdata = d
        const f = d.fields || {}
        for (const k in fieldMap) fieldMap[k].text = f[k] !== undefined ? String(f[k]) : ""
        backendLabels = d.backendLabels || []
        experimentalLabel = d.experimentalLabel || experimentalLabel
        leftBackend.currentIndex = backendLabels.indexOf(d.backends[0])
        rightBackend.currentIndex = backendLabels.indexOf(d.backends[1])
        leftBackend.enabled = !d.backendLocked[0]
        rightBackend.enabled = !d.backendLocked[1]
        autoLoad.checked = !!d.autoLoad
        chime.checked = !!d.chime
        setDevices(d.devices || [], d.leftDevice || "", d.rightDevice || "")
        vkLabel.text = ""
        if (d.showVk) Qt.callLater(checkVk)
        errorMsg.visible = false
    }

    function setDevices(labels, left, right) {
        deviceLabels = labels
        leftDev.currentIndex = labels.indexOf(left)
        rightDev.currentIndex = labels.indexOf(right)
    }

    function checkVk() {
        vkLabel.text = JSON.parse(Backend.call("app", "settingsVk", "{}"))
    }

    function save() {
        const fields = {}
        for (const k in fieldMap) fields[k] = fieldMap[k].text
        fields["chime_min_secs"] = String(chimeSecs.value)
        const arg = {
            fields: fields,
            autoLoad: autoLoad.checked,
            chime: chime.checked,
            backends: [leftBackend.currentText, rightBackend.currentText],
            leftDevice: leftDev.currentIndex >= 0 ? leftDev.currentText : "",
            rightDevice: rightDev.currentIndex >= 0 ? rightDev.currentText : ""
        }
        const r = JSON.parse(Backend.call("app", "settingsSave", JSON.stringify(arg)))
        if (r && r.error) {
            errorMsg.text = r.error
            errorMsg.visible = true
            return
        }
        win.close()
    }

    property var fieldMap: ({
        "models_dir": modelsDir, "server_bin": serverBin, "server_bin_vulkan": serverBinVk, "host": host,
        "left_port": leftPort, "right_port": rightPort, "overhead_mib": overhead, "fit_margin_mib": fitMargin,
        "vllm_bin": vllmBin, "vllm_models_dir": vllmDir, "vllm_env": vllmEnv,
        "vllm_left_port": vllmLeft, "vllm_right_port": vllmRight,
        "kobold_rocm_bin": koboldRocm, "kobold_vulkan_bin": koboldVk,
        "kobold_left_port": koboldLeft, "kobold_right_port": koboldRight,
        "radiance_bin": radBin, "radiance_home": radHome, "radiance_models_dir": radDir,
        "radiance_recipes_dir": radRecipes, "radiance_env": radEnv,
        "radiance_left_port": radLeft, "radiance_right_port": radRight,
        "llama_api_key": llamaKey, "kobold_password": koboldPw, "vllm_api_key": vllmKey, "radiance_api_key": radKey
    })

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Kirigami.InlineMessage {
            id: errorMsg
            Layout.fillWidth: true
            Layout.margins: Kirigami.Units.smallSpacing
            type: Kirigami.MessageType.Error
            visible: false
            showCloseButton: true
        }

        QQC2.ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: availableWidth

            Kirigami.FormLayout {
                width: parent.width

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "Backends" }
                RowLayout {
                    Kirigami.FormData.label: "Left pane:"
                    BackendBox { id: leftBackend; labels: win.backendLabels; experimental: win.experimentalLabel; help: win.h("backend") }
                    QQC2.Label {
                        visible: leftBackend.currentText === win.experimentalLabel
                        text: "⚠ Experimental - not everything is ironed out yet"
                        color: Kirigami.Theme.negativeTextColor
                    }
                    QQC2.Label { visible: !leftBackend.enabled; text: "(stop its model to change)"; color: Kirigami.Theme.disabledTextColor }
                }
                RowLayout {
                    Kirigami.FormData.label: "Right pane:"
                    BackendBox { id: rightBackend; labels: win.backendLabels; experimental: win.experimentalLabel; help: win.h("backend") }
                    QQC2.Label {
                        visible: rightBackend.currentText === win.experimentalLabel
                        text: "⚠ Experimental - not everything is ironed out yet"
                        color: Kirigami.Theme.negativeTextColor
                    }
                    QQC2.Label { visible: !rightBackend.enabled; text: "(stop its model to change)"; color: Kirigami.Theme.disabledTextColor }
                }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "llama.cpp" }
                PathField { id: modelsDir; Kirigami.FormData.label: "Models folder:"; browse: "folder"; help: win.h("models_dir") }
                PathField { id: serverBin; Kirigami.FormData.label: "llama-server ROCm binary:"; browse: "file"; help: win.h("server_bin") }
                PathField { id: serverBinVk; Kirigami.FormData.label: "llama-server Vulkan binary:"; browse: "file"; help: win.h("server_bin_vulkan") }
                PathField { id: host; Kirigami.FormData.label: "Listen host:"; help: win.h("host") }
                PathField { id: leftPort; Kirigami.FormData.label: "Default port (left):"; help: win.h("left_port") }
                PathField { id: rightPort; Kirigami.FormData.label: "Default port (right):"; help: win.h("right_port") }
                PathField { id: overhead; Kirigami.FormData.label: "Runtime overhead (MiB):"; help: win.h("overhead_mib") }
                PathField { id: fitMargin; Kirigami.FormData.label: "Fit margin (MiB):"; help: win.h("fit_margin_mib") }
                PathField { id: llamaKey; Kirigami.FormData.label: "llama-server API key:"; secret: true; help: win.h("llama_api_key") }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "vLLM" }
                PathField { id: vllmBin; Kirigami.FormData.label: "vLLM binary:"; browse: "file"; help: win.h("vllm_bin") }
                PathField { id: vllmDir; Kirigami.FormData.label: "vLLM models folder:"; browse: "folder"; help: win.h("vllm_models_dir") }
                PathField { id: vllmEnv; Kirigami.FormData.label: "vLLM environment:"; help: win.h("vllm_env") }
                PathField { id: vllmLeft; Kirigami.FormData.label: "vLLM port (left):"; help: win.h("vllm_left_port") }
                PathField { id: vllmRight; Kirigami.FormData.label: "vLLM port (right):"; help: win.h("vllm_right_port") }
                PathField { id: vllmKey; Kirigami.FormData.label: "vLLM API key:"; secret: true; help: win.h("vllm_api_key") }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "KoboldCpp" }
                PathField { id: koboldRocm; Kirigami.FormData.label: "KoboldCpp ROCm binary:"; browse: "file"; help: win.h("kobold_rocm_bin") }
                PathField { id: koboldVk; Kirigami.FormData.label: "KoboldCpp Vulkan binary:"; browse: "file"; help: win.h("kobold_vulkan_bin") }
                PathField { id: koboldLeft; Kirigami.FormData.label: "KoboldCpp port (left):"; help: win.h("kobold_left_port") }
                PathField { id: koboldRight; Kirigami.FormData.label: "KoboldCpp port (right):"; help: win.h("kobold_right_port") }
                PathField { id: koboldPw; Kirigami.FormData.label: "KoboldCpp password:"; secret: true; help: win.h("kobold_password") }
                RowLayout {
                    Kirigami.FormData.label: "Vulkan devices:"
                    QQC2.Label {
                        id: vkLabel
                        Layout.fillWidth: true
                        color: Kirigami.Theme.disabledTextColor
                        wrapMode: Text.Wrap
                    }
                    QQC2.Button {
                        id: vkBtn
                        text: "Check"
                        onClicked: win.checkVk()
                        HelpTip {
                            text: "How each GPU maps to KoboldCpp's Vulkan device number (from vulkaninfo).\nThe Vulkan build selects cards by this number, which can differ from ROCm's."
                            shown: vkBtn.hovered
                        }
                    }
                }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "Radiance" }
                PathField { id: radBin; Kirigami.FormData.label: "Radiance binary:"; browse: "file"; help: win.h("radiance_bin") }
                PathField { id: radHome; Kirigami.FormData.label: "Radiance home (plugins):"; browse: "folder"; help: win.h("radiance_home") }
                PathField { id: radDir; Kirigami.FormData.label: "Radiance models folder:"; browse: "folder"; help: win.h("radiance_models_dir") }
                PathField { id: radRecipes; Kirigami.FormData.label: "Recipes folder:"; browse: "folder"; help: win.h("radiance_recipes_dir") }
                PathField { id: radEnv; Kirigami.FormData.label: "Radiance environment:"; help: win.h("radiance_env") }
                PathField { id: radLeft; Kirigami.FormData.label: "Radiance port (left):"; help: win.h("radiance_left_port") }
                PathField { id: radRight; Kirigami.FormData.label: "Radiance port (right):"; help: win.h("radiance_right_port") }
                PathField { id: radKey; Kirigami.FormData.label: "Radiance API key:"; secret: true; help: win.h("radiance_api_key") }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "Behaviour" }
                QQC2.CheckBox {
                    id: autoLoad
                    text: "Auto load last used models on startup"
                    HelpTip { text: win.h("auto_load"); shown: autoLoad.hovered }
                }
                RowLayout {
                    QQC2.CheckBox {
                        id: chime
                        text: "Chime when a response finishes"
                        HelpTip { text: win.h("chime"); shown: chime.hovered }
                    }
                    QQC2.Label { text: "if it took at least" }
                    QQC2.SpinBox {
                        id: chimeSecs
                        from: 0; to: 600
                        editable: true
                        value: Number(fieldMapValue("chime_min_secs")) || 0
                        function fieldMapValue(k) { return win.sdata && win.sdata.fields ? win.sdata.fields[k] : 5 }
                        HelpTip { text: win.h("chime_min_secs"); shown: chimeSecs.hovered }
                    }
                    QQC2.Label { text: "seconds" }
                    QQC2.Button { text: "Test"; onClicked: Backend.call("app", "testChime", "{}") }
                }

                Kirigami.Separator { Kirigami.FormData.isSection: true; Kirigami.FormData.label: "GPUs" }
                QQC2.ComboBox {
                    id: leftDev
                    Kirigami.FormData.label: "Left pane GPU:"
                    Layout.fillWidth: true
                    model: win.deviceLabels
                }
                QQC2.ComboBox {
                    id: rightDev
                    Kirigami.FormData.label: "Right pane GPU:"
                    Layout.fillWidth: true
                    model: win.deviceLabels
                }
                QQC2.Button {
                    text: "Re-detect GPUs"
                    icon.name: "view-refresh"
                    onClicked: Backend.call("app", "settingsRedetect", JSON.stringify({ serverBin: serverBin.text }))
                }
            }
        }

        QQC2.DialogButtonBox {
            Layout.fillWidth: true
            standardButtons: QQC2.DialogButtonBox.Save | QQC2.DialogButtonBox.Cancel
            onAccepted: win.save()
            onRejected: win.close()
        }
    }
}
