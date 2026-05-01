#pragma once

#include <QString>

class QApplication;
class QTranslator;
class QWidget;

namespace aitrain_app {

QString languageSettingsKey();
QString defaultLanguageCode();
QString configuredLanguageCode();
void storeLanguageCode(const QString& languageCode);
QString languageDisplayName(const QString& languageCode);
bool loadTranslator(QApplication& app, QTranslator* translator, const QString& languageCode);
QString translateText(const char* context, const QString& text);
void translateWidgetTree(QWidget* root, const char* context = "MainWindow");

} // namespace aitrain_app
