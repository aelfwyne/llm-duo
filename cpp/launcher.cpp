// QApplication + QML engine for LLM Duo.
//
// Written in C++ because the KDE desktop style (qqc2-desktop-style, "org.kde.desktop") needs a
// QApplication (widgets), which cxx-qt-lib does not wrap; everything else is Rust/QML.

#include <QApplication>
#include <QClipboard>
#include <QFileInfo>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQuickStyle>
#include <QUrl>
#include <cstddef>

extern "C" void llmduo_set_clipboard(const char *data, std::size_t len)
{
    if (QClipboard *cb = QGuiApplication::clipboard()) {
        cb->setText(QString::fromUtf8(data, static_cast<qsizetype>(len)));
    }
}

static bool hasQmlModule(const QStringList &importPaths, const QString &relDir)
{
    for (const QString &p : importPaths) {
        if (QFileInfo::exists(p + QLatin1Char('/') + relDir + QStringLiteral("/qmldir"))) {
            return true;
        }
    }
    return false;
}

extern "C" int llmduo_run_gui(int argc, char **argv)
{
    static int s_argc = argc; // QApplication keeps a reference to argc
    QApplication app(s_argc, argv);
    QApplication::setApplicationName(QStringLiteral("llm-duo"));
    QApplication::setApplicationDisplayName(QStringLiteral("LLM Duo"));
    QApplication::setApplicationVersion(QStringLiteral("2.0.0"));
    QApplication::setDesktopFileName(QStringLiteral("io.github.llmduo"));
    QApplication::setWindowIcon(QIcon::fromTheme(QStringLiteral("io.github.llmduo"),
                                                 QIcon::fromTheme(QStringLiteral("cpu"))));

    QQmlApplicationEngine engine;

    // Plasma look: use the KDE desktop style when it is installed (qml6-module-org-kde-desktop),
    // unless the user picked a style with QT_QUICK_CONTROLS_STYLE.
    if (qEnvironmentVariableIsEmpty("QT_QUICK_CONTROLS_STYLE")) {
        if (hasQmlModule(engine.importPathList(), QStringLiteral("org/kde/desktop"))) {
            QQuickStyle::setStyle(QStringLiteral("org.kde.desktop"));
        } else {
            QQuickStyle::setStyle(QStringLiteral("Fusion"));
        }
    }
    if (!hasQmlModule(engine.importPathList(), QStringLiteral("org/kde/kirigami"))) {
        qCritical("LLM Duo needs Kirigami (Ubuntu package: qml6-module-org-kde-kirigami)");
    }

    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed, &app, []() { QCoreApplication::exit(1); },
        Qt::QueuedConnection);
    engine.load(QUrl(QStringLiteral("qrc:/qt/qml/io/github/llmduo/qml/Main.qml")));
    if (engine.rootObjects().isEmpty()) {
        return 1;
    }
    return app.exec();
}
