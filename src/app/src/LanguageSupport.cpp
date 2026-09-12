#include "LanguageSupport.h"
#include "WorkbenchTranslation.h"

#include "ApplicationSettingsService.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QLocale>
#include <QStringList>
#include <QTranslator>

namespace aitrain_app {
namespace {

QString normalizeLanguageCode(const QString& languageCode)
{
    if (languageCode == QStringLiteral("en") || languageCode == QStringLiteral("en_US")) {
        return QStringLiteral("en_US");
    }
    return QStringLiteral("zh_CN");
}

QStringList translationSearchRoots()
{
    const QString appDir = QApplication::applicationDirPath();
    return {
        QStringLiteral(":/translations"),
        QDir(appDir).absoluteFilePath(QStringLiteral("translations")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../translations")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../../translations")),
        QDir(appDir).absoluteFilePath(QStringLiteral("../src/app/translations")),
        QDir::current().absoluteFilePath(QStringLiteral("translations"))
    };
}

} // namespace

QString languageSettingsKey()
{
    return QStringLiteral("settings/language");
}

QString defaultLanguageCode()
{
    const QString system = QLocale::system().name();
    return system.startsWith(QStringLiteral("en")) ? QStringLiteral("en_US") : QStringLiteral("zh_CN");
}

QString configuredLanguageCode()
{
    ApplicationSettingsService settings;
    return normalizeLanguageCode(settings.languageCode(defaultLanguageCode()));
}

void storeLanguageCode(const QString& languageCode)
{
    ApplicationSettingsService settings;
    settings.setLanguageCode(normalizeLanguageCode(languageCode));
}

QString languageDisplayName(const QString& languageCode)
{
    return normalizeLanguageCode(languageCode) == QStringLiteral("en_US")
        ? QStringLiteral("English")
        : QStringLiteral("中文");
}

bool loadTranslator(QApplication& app, QTranslator* translator, const QString& languageCode)
{
    if (!translator) {
        return false;
    }

    const QString normalized = normalizeLanguageCode(languageCode);
    if (normalized == QStringLiteral("zh_CN")) {
        return false;
    }

    const QString fileName = QStringLiteral("aitrain_%1.qm").arg(normalized);
    for (const QString& root : translationSearchRoots()) {
        if (translator->load(fileName, root)) {
            app.installTranslator(translator);
            return true;
        }
    }
    return false;
}

QString translateText(const char* context, const QString& text)
{
    if (text.isEmpty()) {
        return text;
    }
    const QByteArray source = text.toUtf8();
    const QString translated = QCoreApplication::translate(context, source.constData());
    return translated == text ? workbenchText(text) : translated;
}

} // namespace aitrain_app
