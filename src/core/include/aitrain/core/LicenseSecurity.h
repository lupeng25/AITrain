#pragma once

#include "aitrain/core/LicenseManager.h"

#include <QByteArray>
#include <QDateTime>
#include <QString>

namespace aitrain {

constexpr qint64 kTrustedClockRollbackToleranceSeconds = 5 * 60;

bool writeProtectedLicenseKeyFile(
    const QString& path,
    const LicenseKeyPair& keyPair,
    QString* error = nullptr);

bool readProtectedLicenseKeyFile(
    const QString& path,
    LicenseKeyPair* keyPair,
    QString* error = nullptr);

LicenseValidationResult validateLicenseTokenWithTrustedClock(
    const QString& token,
    const QByteArray& publicKeyBase64,
    const QString& trustedClockPath,
    const QString& expectedMachineCode = currentMachineCode(),
    const QDateTime& nowUtc = QDateTime::currentDateTimeUtc(),
    qint64 rollbackToleranceSeconds = kTrustedClockRollbackToleranceSeconds);

} // namespace aitrain
