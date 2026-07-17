#include "AppStyle.h"
#include "AiTrainAppConfig.h"
#include "LanguageSupport.h"
#include "MainWindow.h"
#include "RegistrationDialog.h"
#include "LicenseTokenStore.h"
#include "aitrain/core/LicenseManager.h"
#include "aitrain/core/LicenseSecurity.h"

#include <QApplication>
#include <QIcon>
#include <QDir>
#include <QStandardPaths>
#include <QTranslator>

#include <memory>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("AITrain Studio"));
    QApplication::setOrganizationName(QStringLiteral("AITrain"));
    QApplication::setWindowIcon(QIcon(QStringLiteral(":/icons/app.ico")));
    QTranslator translator;
    aitrain_app::loadTranslator(app, &translator, aitrain_app::configuredLanguageCode());
    AppStyle::apply(app);

    const QByteArray publicKeyBase64(AITRAIN_LICENSE_PUBLIC_KEY_B64);
    const QString trustedClockPath = QDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation))
        .filePath(QStringLiteral("license/trusted-utc.dat"));
    const QString storedToken = aitrain_app::LicenseTokenStore().read();
    aitrain::LicenseValidationResult license =
        aitrain::validateLicenseTokenWithTrustedClock(
            storedToken,
            publicKeyBase64,
            trustedClockPath);
    if (!license.isValid()) {
        RegistrationDialog dialog(publicKeyBase64);
        dialog.setWindowIcon(QIcon(QStringLiteral(":/icons/app.ico")));
        if (dialog.exec() != QDialog::Accepted) {
            return 0;
        }
        license.payload = dialog.activatedPayload();
        license.status = aitrain::LicenseStatus::Valid;
    }

    const QString licenseExpiry = license.payload.expiresAt.isValid()
        ? license.payload.expiresAt.toLocalTime().date().toString(Qt::ISODate)
        : QString();
    auto window = std::make_unique<MainWindow>(license.payload.customer, licenseExpiry);
    window->setWindowIcon(QIcon(QStringLiteral(":/icons/app.ico")));
    window->resize(1280, 820);
    window->show();

    return app.exec();
}
