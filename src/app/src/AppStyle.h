#pragma once

#include <QString>
class QApplication;

namespace AppStyle {

QString configuredTheme();
void apply(QApplication& app, const QString& theme = QString());

}

