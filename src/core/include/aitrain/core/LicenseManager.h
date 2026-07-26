#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QString>

namespace aitrain {

struct LicensePayload {
    QString product;
    QString customer;
    QString machineCode;
    QString licenseId;
    QDateTime issuedAt;
    QDateTime expiresAt;
};

struct LicenseKeyPair {
    QByteArray publicKeyBase64;
    QByteArray privateKeyBase64;
};

enum class LicenseStatus {
    Valid,
    MissingToken,
    MissingPublicKey,
    MalformedToken,
    PayloadInvalid,
    ProductMismatch,
    MachineMismatch,
    Expired,
    SignatureInvalid,
    CryptoUnavailable,
    ClockRollbackDetected,
    ProtectedStorageCorrupted
};

struct LicenseValidationResult {
    LicenseStatus status = LicenseStatus::MissingToken;
    QString message;
    LicensePayload payload;

    bool isValid() const { return status == LicenseStatus::Valid; }
};

struct MachineCodeResult final {
    QString machineCode;
    QString unavailableReason;

    bool isAvailable() const { return !machineCode.isEmpty(); }
};

QString licenseProductName();
MachineCodeResult currentMachineCodeResult();
QString currentMachineCode();
QString normalizeMachineCode(const QString& machineCode);
bool licenseCryptoAvailable();

LicenseValidationResult validateLicenseToken(
    const QString& token,
    const QByteArray& publicKeyBase64,
    const QString& expectedMachineCode = currentMachineCode(),
    const QDateTime& nowUtc = QDateTime::currentDateTimeUtc());

bool generateLicenseKeyPair(LicenseKeyPair* keyPair, QString* error = nullptr);
QByteArray publicKeyFromPrivateKey(const QByteArray& privateKeyBase64, QString* error = nullptr);
QString createLicenseToken(const LicensePayload& payload, const QByteArray& privateKeyBase64, QString* error = nullptr);

} // namespace aitrain
