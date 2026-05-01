#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QJsonObject>
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
    CryptoUnavailable
};

struct LicenseValidationResult {
    LicenseStatus status = LicenseStatus::MissingToken;
    QString message;
    LicensePayload payload;

    bool isValid() const { return status == LicenseStatus::Valid; }
};

QString licenseProductName();
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

QJsonObject licensePayloadToJson(const LicensePayload& payload);
bool licensePayloadFromJson(const QJsonObject& object, LicensePayload* payload, QString* error = nullptr);
QString licenseStatusText(LicenseStatus status);

} // namespace aitrain
