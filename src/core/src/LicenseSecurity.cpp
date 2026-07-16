#include "aitrain/core/LicenseSecurity.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <aclapi.h>
#include <dpapi.h>
#endif

namespace aitrain {
namespace {

const QByteArray kPrivateKeyEntropy = QByteArrayLiteral("AITrain/LicenseGenerator/PrivateKey/v1");
const QByteArray kTrustedClockEntropy = QByteArrayLiteral("AITrain/Studio/TrustedUtc/v1");

LicenseValidationResult securityResult(LicenseStatus status, const QString& message)
{
    LicenseValidationResult result;
    result.status = status;
    result.message = message;
    return result;
}

void clearSensitive(QByteArray* data)
{
    if (!data || data->isEmpty()) {
        return;
    }
#ifdef Q_OS_WIN
    SecureZeroMemory(data->data(), static_cast<SIZE_T>(data->size()));
#else
    volatile char* bytes = data->data();
    for (int i = 0; i < data->size(); ++i) {
        bytes[i] = 0;
    }
#endif
    data->clear();
}

#ifdef Q_OS_WIN

QString windowsError(const QString& operation, DWORD code)
{
    return QStringLiteral("%1 failed with Windows error %2").arg(operation).arg(code);
}

bool currentUserOnlyAcl(const QString& path, QString* error)
{
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        if (error) {
            *error = windowsError(QStringLiteral("OpenProcessToken"), GetLastError());
        }
        return false;
    }

    DWORD size = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    QByteArray tokenData(static_cast<int>(size), '\0');
    if (size == 0 || !GetTokenInformation(token, TokenUser, tokenData.data(), size, &size)) {
        const DWORD code = GetLastError();
        CloseHandle(token);
        if (error) {
            *error = windowsError(QStringLiteral("GetTokenInformation"), code);
        }
        return false;
    }
    CloseHandle(token);

    const auto* tokenUser = reinterpret_cast<const TOKEN_USER*>(tokenData.constData());
    EXPLICIT_ACCESSW access = {};
    access.grfAccessPermissions = GENERIC_ALL;
    access.grfAccessMode = SET_ACCESS;
    access.grfInheritance = NO_INHERITANCE;
    access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    access.Trustee.TrusteeType = TRUSTEE_IS_USER;
    access.Trustee.ptstrName = static_cast<LPWSTR>(tokenUser->User.Sid);

    PACL acl = nullptr;
    DWORD status = SetEntriesInAclW(1, &access, nullptr, &acl);
    if (status == ERROR_SUCCESS) {
        status = SetNamedSecurityInfoW(
            const_cast<LPWSTR>(reinterpret_cast<LPCWSTR>(path.utf16())),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            nullptr,
            nullptr,
            acl,
            nullptr);
    }
    if (acl) {
        LocalFree(acl);
    }
    if (status != ERROR_SUCCESS) {
        if (error) {
            *error = windowsError(QStringLiteral("SetNamedSecurityInfo"), status);
        }
        return false;
    }
    return true;
}

bool protectForCurrentUser(
    const QByteArray& clearText,
    const QByteArray& entropy,
    QByteArray* protectedData,
    QString* error)
{
    DATA_BLOB input = {
        static_cast<DWORD>(clearText.size()),
        reinterpret_cast<BYTE*>(const_cast<char*>(clearText.constData()))};
    DATA_BLOB optionalEntropy = {
        static_cast<DWORD>(entropy.size()),
        reinterpret_cast<BYTE*>(const_cast<char*>(entropy.constData()))};
    DATA_BLOB output = {};
    if (!CryptProtectData(
            &input,
            L"AITrain protected local data",
            &optionalEntropy,
            nullptr,
            nullptr,
            CRYPTPROTECT_UI_FORBIDDEN,
            &output)) {
        if (error) {
            *error = windowsError(QStringLiteral("CryptProtectData"), GetLastError());
        }
        return false;
    }
    *protectedData = QByteArray(reinterpret_cast<const char*>(output.pbData), static_cast<int>(output.cbData));
    SecureZeroMemory(output.pbData, output.cbData);
    LocalFree(output.pbData);
    return true;
}

bool unprotectForCurrentUser(
    const QByteArray& protectedData,
    const QByteArray& entropy,
    QByteArray* clearText,
    QString* error)
{
    DATA_BLOB input = {
        static_cast<DWORD>(protectedData.size()),
        reinterpret_cast<BYTE*>(const_cast<char*>(protectedData.constData()))};
    DATA_BLOB optionalEntropy = {
        static_cast<DWORD>(entropy.size()),
        reinterpret_cast<BYTE*>(const_cast<char*>(entropy.constData()))};
    DATA_BLOB output = {};
    if (!CryptUnprotectData(
            &input,
            nullptr,
            &optionalEntropy,
            nullptr,
            nullptr,
            CRYPTPROTECT_UI_FORBIDDEN,
            &output)) {
        if (error) {
            *error = windowsError(QStringLiteral("CryptUnprotectData"), GetLastError());
        }
        return false;
    }
    *clearText = QByteArray(reinterpret_cast<const char*>(output.pbData), static_cast<int>(output.cbData));
    SecureZeroMemory(output.pbData, output.cbData);
    LocalFree(output.pbData);
    return true;
}

#endif

bool writeProtectedData(
    const QString& path,
    const QByteArray& clearText,
    const QByteArray& entropy,
    const QString& kind,
    const QJsonObject& metadata,
    QString* error)
{
#ifndef Q_OS_WIN
    Q_UNUSED(path);
    Q_UNUSED(clearText);
    Q_UNUSED(entropy);
    Q_UNUSED(kind);
    Q_UNUSED(metadata);
    if (error) {
        *error = QStringLiteral("DPAPI protected storage is only available on Windows");
    }
    return false;
#else
    QByteArray encrypted;
    if (!protectForCurrentUser(clearText, entropy, &encrypted, error)) {
        return false;
    }
    QJsonObject envelope = metadata;
    envelope.insert(QStringLiteral("type"), kind);
    envelope.insert(QStringLiteral("protection"), QStringLiteral("windows-dpapi-current-user"));
    envelope.insert(QStringLiteral("version"), 1);
    envelope.insert(QStringLiteral("protectedData"), QString::fromLatin1(encrypted.toBase64()));

    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        if (error) {
            *error = QStringLiteral("Failed to create protected storage directory");
        }
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error) {
            *error = file.errorString();
        }
        return false;
    }
    if (file.write(QJsonDocument(envelope).toJson(QJsonDocument::Compact)) < 0 || !file.commit()) {
        if (error) {
            *error = file.errorString();
        }
        return false;
    }
    if (!currentUserOnlyAcl(path, error)) {
        QFile::remove(path);
        return false;
    }
    return true;
#endif
}

bool readProtectedData(
    const QString& path,
    const QByteArray& entropy,
    const QString& expectedKind,
    QByteArray* clearText,
    QJsonObject* metadata,
    QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) {
            *error = file.errorString();
        }
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (!document.isObject()) {
        if (error) {
            *error = QStringLiteral("Protected storage envelope is invalid JSON: %1").arg(parseError.errorString());
        }
        return false;
    }
    const QJsonObject envelope = document.object();
    if (envelope.value(QStringLiteral("type")).toString() != expectedKind
        || envelope.value(QStringLiteral("protection")).toString() != QStringLiteral("windows-dpapi-current-user")
        || envelope.value(QStringLiteral("version")).toInt() != 1) {
        if (error) {
            *error = QStringLiteral("Protected storage envelope has an unsupported type or version");
        }
        return false;
    }
    const QByteArray encrypted = QByteArray::fromBase64(
        envelope.value(QStringLiteral("protectedData")).toString().toLatin1());
    if (encrypted.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Protected storage payload is empty or invalid");
        }
        return false;
    }
#ifndef Q_OS_WIN
    Q_UNUSED(entropy);
    Q_UNUSED(clearText);
    if (error) {
        *error = QStringLiteral("DPAPI protected storage is only available on Windows");
    }
    return false;
#else
    if (!unprotectForCurrentUser(encrypted, entropy, clearText, error)) {
        return false;
    }
    if (metadata) {
        *metadata = envelope;
    }
    return true;
#endif
}

} // namespace

bool writeProtectedLicenseKeyFile(
    const QString& path,
    const LicenseKeyPair& keyPair,
    QString* error)
{
    if (keyPair.privateKeyBase64.trimmed().isEmpty() || keyPair.publicKeyBase64.trimmed().isEmpty()) {
        if (error) {
            *error = QStringLiteral("License key pair is incomplete");
        }
        return false;
    }
    QJsonObject metadata;
    metadata.insert(QStringLiteral("curve"), QStringLiteral("P-256"));
    metadata.insert(QStringLiteral("publicKey"), QString::fromLatin1(keyPair.publicKeyBase64.trimmed()));
    metadata.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    return writeProtectedData(
        path,
        keyPair.privateKeyBase64.trimmed(),
        kPrivateKeyEntropy,
        QStringLiteral("aitrain-protected-license-key"),
        metadata,
        error);
}

bool readProtectedLicenseKeyFile(
    const QString& path,
    LicenseKeyPair* keyPair,
    QString* error)
{
    if (!keyPair) {
        if (error) {
            *error = QStringLiteral("Missing license key pair output");
        }
        return false;
    }
    QByteArray privateKey;
    QJsonObject metadata;
    if (!readProtectedData(
            path,
            kPrivateKeyEntropy,
            QStringLiteral("aitrain-protected-license-key"),
            &privateKey,
            &metadata,
            error)) {
        return false;
    }
    const QByteArray publicKey = metadata.value(QStringLiteral("publicKey")).toString().toLatin1().trimmed();
    QString deriveError;
    const QByteArray derivedPublicKey = publicKeyFromPrivateKey(privateKey, &deriveError);
    if (publicKey.isEmpty() || derivedPublicKey.isEmpty() || derivedPublicKey != publicKey) {
        clearSensitive(&privateKey);
        if (error) {
            *error = deriveError.isEmpty()
                ? QStringLiteral("Protected private key does not match its public key")
                : deriveError;
        }
        return false;
    }
    keyPair->privateKeyBase64 = privateKey;
    keyPair->publicKeyBase64 = publicKey;
    return true;
}

LicenseValidationResult validateLicenseTokenWithTrustedClock(
    const QString& token,
    const QByteArray& publicKeyBase64,
    const QString& trustedClockPath,
    const QString& expectedMachineCode,
    const QDateTime& nowUtc,
    qint64 rollbackToleranceSeconds)
{
    if (trustedClockPath.trimmed().isEmpty()) {
        return securityResult(LicenseStatus::ProtectedStorageCorrupted, QStringLiteral("Trusted clock path is empty"));
    }
    const QDateTime checkedAt = nowUtc.isValid() ? nowUtc.toUTC() : QDateTime::currentDateTimeUtc();
    QDateTime trustedAt;
    if (QFileInfo::exists(trustedClockPath)) {
        QByteArray clearText;
        QString error;
        if (!readProtectedData(
                trustedClockPath,
                kTrustedClockEntropy,
                QStringLiteral("aitrain-trusted-utc"),
                &clearText,
                nullptr,
                &error)) {
            return securityResult(LicenseStatus::ProtectedStorageCorrupted, error);
        }
        const QJsonDocument document = QJsonDocument::fromJson(clearText);
        if (!document.isObject()) {
            return securityResult(LicenseStatus::ProtectedStorageCorrupted, QStringLiteral("Trusted clock payload is invalid"));
        }
        trustedAt = QDateTime::fromString(document.object().value(QStringLiteral("utc")).toString(), Qt::ISODateWithMs);
        if (!trustedAt.isValid()) {
            trustedAt = QDateTime::fromString(document.object().value(QStringLiteral("utc")).toString(), Qt::ISODate);
        }
        if (!trustedAt.isValid()) {
            return securityResult(LicenseStatus::ProtectedStorageCorrupted, QStringLiteral("Trusted clock UTC is invalid"));
        }
        trustedAt = trustedAt.toUTC();
        if (checkedAt.addSecs(qMax<qint64>(0, rollbackToleranceSeconds)) < trustedAt) {
            return securityResult(LicenseStatus::ClockRollbackDetected, QStringLiteral("System clock is earlier than the protected trusted UTC"));
        }
    }

    LicenseValidationResult validation = validateLicenseToken(
        token,
        publicKeyBase64,
        expectedMachineCode,
        checkedAt);
    if (validation.status != LicenseStatus::Valid && validation.status != LicenseStatus::Expired) {
        return validation;
    }

    const QDateTime nextTrustedAt = trustedAt.isValid() && trustedAt > checkedAt ? trustedAt : checkedAt;
    const QJsonObject clockObject{{QStringLiteral("utc"), nextTrustedAt.toString(Qt::ISODateWithMs)}};
    QString writeError;
    if (!writeProtectedData(
            trustedClockPath,
            QJsonDocument(clockObject).toJson(QJsonDocument::Compact),
            kTrustedClockEntropy,
            QStringLiteral("aitrain-trusted-utc"),
            QJsonObject(),
            &writeError)) {
        return securityResult(LicenseStatus::ProtectedStorageCorrupted, writeError);
    }
    return validation;
}

} // namespace aitrain
