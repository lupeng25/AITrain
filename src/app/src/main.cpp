#include "AppStyle.h"
#include "MainWindow.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("AITrain Studio"));
    QApplication::setOrganizationName(QStringLiteral("AITrain"));
    QApplication::setWindowIcon(QIcon(QStringLiteral(":/icons/app.ico")));
    AppStyle::apply(app);

    MainWindow window;
    window.setWindowIcon(QIcon(QStringLiteral(":/icons/app.ico")));
    window.resize(1280, 820);
    window.show();

    return app.exec();
}
