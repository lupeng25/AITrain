#include "LicenseTokenStore.h"

#include <QSettings>

namespace aitrain_app {

QString LicenseTokenStore::read() const
{
    QSettings settings;
    return settings.value(QStringLiteral("license/token")).toString().trimmed();
}

void LicenseTokenStore::write(const QString& token) const
{
    QSettings settings;
    settings.setValue(QStringLiteral("license/token"), token.trimmed());
    settings.sync();
}

} // namespace aitrain_app
