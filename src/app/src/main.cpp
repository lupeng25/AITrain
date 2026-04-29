#include "AppStyle.h"
#include "MainWindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("AITrain Studio"));
    QApplication::setOrganizationName(QStringLiteral("AITrain"));
    AppStyle::apply(app);

    MainWindow window;
    window.resize(1280, 820);
    window.show();

    return app.exec();
}
