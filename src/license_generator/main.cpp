#include "AppStyle.h"
#include "LicenseGeneratorWindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("AITrain License Generator"));
    QApplication::setOrganizationName(QStringLiteral("AITrain"));
    AppStyle::apply(app);

    LicenseGeneratorWindow window;
    window.resize(920, 680);
    window.show();
    return app.exec();
}
