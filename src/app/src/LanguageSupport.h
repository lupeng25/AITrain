#pragma once

#include <QString>

class QApplication;
class QTranslator;

namespace aitrain_app {

QString languageSettingsKey();
QString defaultLanguageCode();
QString configuredLanguageCode();
void storeLanguageCode(const QString& languageCode);
QString languageDisplayName(const QString& languageCode);
bool loadTranslator(QApplication& app, QTranslator* translator, const QString& languageCode);
QString translateText(const char* context, const QString& text);

} // namespace aitrain_app
