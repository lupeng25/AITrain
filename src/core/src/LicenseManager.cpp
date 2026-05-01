#include "aitrain/core/LicenseManager.h"

#include <QCryptographicHash>
#include <QJsonDocument>
#include <QJsonValue>
#include <QSysInfo>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

namespace aitrain {
namespace {

constexpr const char* kTokenVersion = "AITRAIN1";
constexpr int kMachineCodeHexLength = 20;

QByteArray base64UrlEncode(const QByteArray& input)
{
    return input.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals);
}

QByteArray base64UrlDecode(QByteArray input, bool* ok)
{
    input = input.trimmed();
    while (input.size() % 4 != 0) {
        input.append('=');
    }
    const QByteArray decoded = QByteArray::fromBase64(input, QByteArray::Base64UrlEncoding);
    if (ok) {
        *ok = !decoded.isEmpty() || input == QByteArrayLiteral("");
    }
    return decoded;
}

QString toIsoUtc(const QDateTime& value)
{
    return value.isValid() ? value.toUTC().toString(Qt::ISODate) : QString();
}

QDateTime fromIsoUtc(const QString& value)
{
    if (value.trimmed().isEmpty()) {
        return {};
    }
    QDateTime parsed = QDateTime::fromString(value, Qt::ISODate);
    if (!parsed.isValid()) {
        parsed = QDateTime::fromString(value, Qt::ISODateWithMs);
    }
    return parsed.isValid() ? parsed.toUTC() : QDateTime();
}

LicenseValidationResult result(LicenseStatus status, const QString& message)
{
    LicenseValidationResult output;
    output.status = status;
    output.message = message;
    return output;
}

QString compactJsonString(const QJsonObject& object)
{
    return QString::fromUtf8(QJsonDocument(object).toJson(QJsonDocument::Compact));
}

QByteArray machineSeed()
{
    QByteArray seed = QSysInfo::machineUniqueId();
    if (seed.isEmpty()) {
        seed = QSysInfo::machineHostName().toUtf8();
    }
    if (seed.isEmpty()) {
        seed = qgetenv("COMPUTERNAME");
    }
    if (seed.isEmpty()) {
        seed = qgetenv("HOSTNAME");
    }
    return seed;
}

QByteArray sha256(const QByteArray& input)
{
    return QCryptographicHash::hash(input, QCryptographicHash::Sha256);
}

#ifdef Q_OS_WIN

QString ntStatusText(NTSTATUS status)
{
    return QStringLiteral("CNG status 0x%1").arg(static_cast<unsigned long>(status), 8, 16, QLatin1Char('0'));
}

bool ntSuccess(NTSTATUS status)
{
    return status >= 0;
}

class AlgorithmHandle {
public:
    explicit AlgorithmHandle(LPCWSTR algorithm)
    {
        status_ = BCryptOpenAlgorithmProvider(&handle_, algorithm, nullptr, 0);
    }

    ~AlgorithmHandle()
    {
        if (handle_) {
            BCryptCloseAlgorithmProvider(handle_, 0);
        }
    }

    BCRYPT_ALG_HANDLE get() const { return handle_; }
    bool ok() const { return ntSuccess(status_); }
    NTSTATUS status() const { return status_; }

private:
    BCRYPT_ALG_HANDLE handle_ = nullptr;
    NTSTATUS status_ = 0;
};

class KeyHandle {
public:
    ~KeyHandle()
    {
        if (handle_) {
            BCryptDestroyKey(handle_);
        }
    }

    BCRYPT_KEY_HANDLE* out() { return &handle_; }
    BCRYPT_KEY_HANDLE get() const { return handle_; }

private:
    BCRYPT_KEY_HANDLE handle_ = nullptr;
};

bool exportKey(BCRYPT_KEY_HANDLE key, LPCWSTR blobType, QByteArray* output, QString* error)
{
    DWORD size = 0;
    NTSTATUS status = BCryptExportKey(key, nullptr, blobType, nullptr, 0, &size, 0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to size exported key: %1").arg(ntStatusText(status));
        }
        return false;
    }
    output->resize(static_cast<int>(size));
    status = BCryptExportKey(
        key,
        nullptr,
        blobType,
        reinterpret_cast<PUCHAR>(output->data()),
        size,
        &size,
        0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to export key: %1").arg(ntStatusText(status));
        }
        output->clear();
        return false;
    }
    output->resize(static_cast<int>(size));
    return true;
}

bool importKey(
    BCRYPT_ALG_HANDLE algorithm,
    LPCWSTR blobType,
    const QByteArray& blob,
    BCRYPT_KEY_HANDLE* key,
    QString* error)
{
    NTSTATUS status = BCryptImportKeyPair(
        algorithm,
        nullptr,
        blobType,
        key,
        reinterpret_cast<PUCHAR>(const_cast<char*>(blob.constData())),
        static_cast<ULONG>(blob.size()),
        0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to import key: %1").arg(ntStatusText(status));
        }
        return false;
    }
    return true;
}

bool signDigest(const QByteArray& privateKeyBlob, const QByteArray& digest, QByteArray* signature, QString* error)
{
    AlgorithmHandle algorithm(BCRYPT_ECDSA_P256_ALGORITHM);
    if (!algorithm.ok()) {
        if (error) {
            *error = QStringLiteral("Failed to open ECDSA P-256 provider: %1").arg(ntStatusText(algorithm.status()));
        }
        return false;
    }

    KeyHandle key;
    if (!importKey(algorithm.get(), BCRYPT_ECCPRIVATE_BLOB, privateKeyBlob, key.out(), error)) {
        return false;
    }

    DWORD size = 0;
    NTSTATUS status = BCryptSignHash(
        key.get(),
        nullptr,
        reinterpret_cast<PUCHAR>(const_cast<char*>(digest.constData())),
        static_cast<ULONG>(digest.size()),
        nullptr,
        0,
        &size,
        0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to size ECDSA signature: %1").arg(ntStatusText(status));
        }
        return false;
    }

    signature->resize(static_cast<int>(size));
    status = BCryptSignHash(
        key.get(),
        nullptr,
        reinterpret_cast<PUCHAR>(const_cast<char*>(digest.constData())),
        static_cast<ULONG>(digest.size()),
        reinterpret_cast<PUCHAR>(signature->data()),
        size,
        &size,
        0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to sign license token: %1").arg(ntStatusText(status));
        }
        signature->clear();
        return false;
    }
    signature->resize(static_cast<int>(size));
    return true;
}

bool verifyDigest(const QByteArray& publicKeyBlob, const QByteArray& digest, const QByteArray& signature, QString* error)
{
    AlgorithmHandle algorithm(BCRYPT_ECDSA_P256_ALGORITHM);
    if (!algorithm.ok()) {
        if (error) {
            *error = QStringLiteral("Failed to open ECDSA P-256 provider: %1").arg(ntStatusText(algorithm.status()));
        }
        return false;
    }

    KeyHandle key;
    if (!importKey(algorithm.get(), BCRYPT_ECCPUBLIC_BLOB, publicKeyBlob, key.out(), error)) {
        return false;
    }

    const NTSTATUS status = BCryptVerifySignature(
        key.get(),
        nullptr,
        reinterpret_cast<PUCHAR>(const_cast<char*>(digest.constData())),
        static_cast<ULONG>(digest.size()),
        reinterpret_cast<PUCHAR>(const_cast<char*>(signature.constData())),
        static_cast<ULONG>(signature.size()),
        0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Signature verification failed: %1").arg(ntStatusText(status));
        }
        return false;
    }
    return true;
}

#endif

} // namespace

QString licenseProductName()
{
    return QStringLiteral("AITrain Studio");
}

QString currentMachineCode()
{
    QByteArray seed = machineSeed();
    if (seed.isEmpty()) {
        seed = QByteArrayLiteral("aitrain-unknown-machine");
    }
    const QByteArray digest = sha256(seed + QByteArrayLiteral("|aitrain-license-v1"));
    const QString hex = QString::fromLatin1(digest.toHex()).toUpper().left(kMachineCodeHexLength);
    QStringList groups;
    for (int i = 0; i < hex.size(); i += 4) {
        groups.append(hex.mid(i, 4));
    }
    return groups.join(QStringLiteral("-"));
}

QString normalizeMachineCode(const QString& machineCode)
{
    QString normalized;
    normalized.reserve(machineCode.size());
    for (const QChar ch : machineCode) {
        if (ch.isLetterOrNumber()) {
            normalized.append(ch.toUpper());
        }
    }
    QStringList groups;
    for (int i = 0; i < normalized.size(); i += 4) {
        groups.append(normalized.mid(i, 4));
    }
    return groups.join(QStringLiteral("-"));
}

bool licenseCryptoAvailable()
{
#ifdef Q_OS_WIN
    return true;
#else
    return false;
#endif
}

QJsonObject licensePayloadToJson(const LicensePayload& payload)
{
    QJsonObject object;
    object.insert(QStringLiteral("product"), payload.product);
    object.insert(QStringLiteral("customer"), payload.customer);
    object.insert(QStringLiteral("machineCode"), normalizeMachineCode(payload.machineCode));
    object.insert(QStringLiteral("licenseId"), payload.licenseId);
    if (payload.issuedAt.isValid()) {
        object.insert(QStringLiteral("issuedAt"), toIsoUtc(payload.issuedAt));
    }
    if (payload.expiresAt.isValid()) {
        object.insert(QStringLiteral("expiresAt"), toIsoUtc(payload.expiresAt));
    }
    return object;
}

bool licensePayloadFromJson(const QJsonObject& object, LicensePayload* payload, QString* error)
{
    if (!payload) {
        if (error) {
            *error = QStringLiteral("Missing payload output");
        }
        return false;
    }

    LicensePayload parsed;
    parsed.product = object.value(QStringLiteral("product")).toString();
    parsed.customer = object.value(QStringLiteral("customer")).toString();
    parsed.machineCode = normalizeMachineCode(object.value(QStringLiteral("machineCode")).toString());
    parsed.licenseId = object.value(QStringLiteral("licenseId")).toString();
    parsed.issuedAt = fromIsoUtc(object.value(QStringLiteral("issuedAt")).toString());
    parsed.expiresAt = fromIsoUtc(object.value(QStringLiteral("expiresAt")).toString());

    if (parsed.product.isEmpty() || parsed.customer.isEmpty() || parsed.machineCode.isEmpty() || parsed.licenseId.isEmpty()) {
        if (error) {
            *error = QStringLiteral("License payload is missing required fields");
        }
        return false;
    }
    if (object.contains(QStringLiteral("issuedAt")) && !object.value(QStringLiteral("issuedAt")).toString().isEmpty() && !parsed.issuedAt.isValid()) {
        if (error) {
            *error = QStringLiteral("License issuedAt is invalid");
        }
        return false;
    }
    if (object.contains(QStringLiteral("expiresAt")) && !object.value(QStringLiteral("expiresAt")).toString().isEmpty() && !parsed.expiresAt.isValid()) {
        if (error) {
            *error = QStringLiteral("License expiresAt is invalid");
        }
        return false;
    }

    *payload = parsed;
    return true;
}

LicenseValidationResult validateLicenseToken(
    const QString& token,
    const QByteArray& publicKeyBase64,
    const QString& expectedMachineCode,
    const QDateTime& nowUtc)
{
    const QByteArray trimmedToken = token.trimmed().toLatin1();
    if (trimmedToken.isEmpty()) {
        return result(LicenseStatus::MissingToken, QStringLiteral("License token is missing"));
    }
    if (publicKeyBase64.trimmed().isEmpty()) {
        return result(LicenseStatus::MissingPublicKey, QStringLiteral("Application license public key is not configured"));
    }
    if (!licenseCryptoAvailable()) {
        return result(LicenseStatus::CryptoUnavailable, QStringLiteral("ECDSA license verification is unavailable on this platform"));
    }

    const QList<QByteArray> parts = trimmedToken.split('.');
    if (parts.size() != 3 || parts.at(0) != QByteArray(kTokenVersion)) {
        return result(LicenseStatus::MalformedToken, QStringLiteral("License token format is invalid"));
    }

    bool payloadOk = false;
    bool signatureOk = false;
    const QByteArray payloadJson = base64UrlDecode(parts.at(1), &payloadOk);
    const QByteArray signature = base64UrlDecode(parts.at(2), &signatureOk);
    if (!payloadOk || !signatureOk || payloadJson.isEmpty() || signature.isEmpty()) {
        return result(LicenseStatus::MalformedToken, QStringLiteral("License token encoding is invalid"));
    }

    const QJsonDocument document = QJsonDocument::fromJson(payloadJson);
    if (!document.isObject()) {
        return result(LicenseStatus::PayloadInvalid, QStringLiteral("License payload is not a JSON object"));
    }

    LicensePayload payload;
    QString parseError;
    if (!licensePayloadFromJson(document.object(), &payload, &parseError)) {
        return result(LicenseStatus::PayloadInvalid, parseError);
    }

#ifdef Q_OS_WIN
    const QByteArray publicKeyBlob = QByteArray::fromBase64(publicKeyBase64.trimmed());
    if (publicKeyBlob.isEmpty()) {
        return result(LicenseStatus::MissingPublicKey, QStringLiteral("Application license public key is invalid"));
    }

    const QByteArray signingInput = parts.at(0) + QByteArrayLiteral(".") + parts.at(1);
    QString verifyError;
    if (!verifyDigest(publicKeyBlob, sha256(signingInput), signature, &verifyError)) {
        LicenseValidationResult output = result(LicenseStatus::SignatureInvalid, verifyError);
        output.payload = payload;
        return output;
    }
#endif

    if (payload.product != licenseProductName()) {
        LicenseValidationResult output = result(LicenseStatus::ProductMismatch, QStringLiteral("License product does not match this application"));
        output.payload = payload;
        return output;
    }
    if (normalizeMachineCode(expectedMachineCode) != normalizeMachineCode(payload.machineCode)) {
        LicenseValidationResult output = result(LicenseStatus::MachineMismatch, QStringLiteral("License machine code does not match this computer"));
        output.payload = payload;
        return output;
    }

    const QDateTime checkedAt = nowUtc.isValid() ? nowUtc.toUTC() : QDateTime::currentDateTimeUtc();
    if (payload.expiresAt.isValid() && checkedAt > payload.expiresAt.toUTC()) {
        LicenseValidationResult output = result(LicenseStatus::Expired, QStringLiteral("License has expired"));
        output.payload = payload;
        return output;
    }

    LicenseValidationResult output = result(LicenseStatus::Valid, QStringLiteral("License is valid"));
    output.payload = payload;
    return output;
}

bool generateLicenseKeyPair(LicenseKeyPair* keyPair, QString* error)
{
    if (!keyPair) {
        if (error) {
            *error = QStringLiteral("Missing key pair output");
        }
        return false;
    }
#ifndef Q_OS_WIN
    if (error) {
        *error = QStringLiteral("ECDSA P-256 key generation requires Windows CNG in this build");
    }
    return false;
#else
    AlgorithmHandle algorithm(BCRYPT_ECDSA_P256_ALGORITHM);
    if (!algorithm.ok()) {
        if (error) {
            *error = QStringLiteral("Failed to open ECDSA P-256 provider: %1").arg(ntStatusText(algorithm.status()));
        }
        return false;
    }

    KeyHandle key;
    NTSTATUS status = BCryptGenerateKeyPair(algorithm.get(), key.out(), 256, 0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to generate ECDSA key pair: %1").arg(ntStatusText(status));
        }
        return false;
    }
    status = BCryptFinalizeKeyPair(key.get(), 0);
    if (!ntSuccess(status)) {
        if (error) {
            *error = QStringLiteral("Failed to finalize ECDSA key pair: %1").arg(ntStatusText(status));
        }
        return false;
    }

    QByteArray publicBlob;
    QByteArray privateBlob;
    if (!exportKey(key.get(), BCRYPT_ECCPUBLIC_BLOB, &publicBlob, error)
        || !exportKey(key.get(), BCRYPT_ECCPRIVATE_BLOB, &privateBlob, error)) {
        return false;
    }

    keyPair->publicKeyBase64 = publicBlob.toBase64();
    keyPair->privateKeyBase64 = privateBlob.toBase64();
    return true;
#endif
}

QByteArray publicKeyFromPrivateKey(const QByteArray& privateKeyBase64, QString* error)
{
#ifndef Q_OS_WIN
    if (error) {
        *error = QStringLiteral("ECDSA P-256 key parsing requires Windows CNG in this build");
    }
    return {};
#else
    QByteArray privateBlob = QByteArray::fromBase64(privateKeyBase64.trimmed());
    if (privateBlob.size() < static_cast<int>(sizeof(BCRYPT_ECCKEY_BLOB))) {
        if (error) {
            *error = QStringLiteral("Private key blob is too small");
        }
        return {};
    }
    const auto* header = reinterpret_cast<const BCRYPT_ECCKEY_BLOB*>(privateBlob.constData());
    if (header->dwMagic != BCRYPT_ECDSA_PRIVATE_P256_MAGIC || header->cbKey == 0) {
        if (error) {
            *error = QStringLiteral("Private key is not an ECDSA P-256 private blob");
        }
        return {};
    }
    const int publicSize = static_cast<int>(sizeof(BCRYPT_ECCKEY_BLOB) + (2 * header->cbKey));
    const int privateSize = static_cast<int>(sizeof(BCRYPT_ECCKEY_BLOB) + (3 * header->cbKey));
    if (privateBlob.size() < privateSize) {
        if (error) {
            *error = QStringLiteral("Private key blob is truncated");
        }
        return {};
    }

    QByteArray publicBlob = privateBlob.left(publicSize);
    auto* publicHeader = reinterpret_cast<BCRYPT_ECCKEY_BLOB*>(publicBlob.data());
    publicHeader->dwMagic = BCRYPT_ECDSA_PUBLIC_P256_MAGIC;
    return publicBlob.toBase64();
#endif
}

QString createLicenseToken(const LicensePayload& payload, const QByteArray& privateKeyBase64, QString* error)
{
    if (!licenseCryptoAvailable()) {
        if (error) {
            *error = QStringLiteral("ECDSA license signing is unavailable on this platform");
        }
        return {};
    }

    LicensePayload normalized = payload;
    normalized.product = normalized.product.isEmpty() ? licenseProductName() : normalized.product;
    normalized.machineCode = normalizeMachineCode(normalized.machineCode);
    if (!normalized.issuedAt.isValid()) {
        normalized.issuedAt = QDateTime::currentDateTimeUtc();
    }

    QString parseError;
    LicensePayload checked;
    const QJsonObject payloadObject = licensePayloadToJson(normalized);
    if (!licensePayloadFromJson(payloadObject, &checked, &parseError)) {
        if (error) {
            *error = parseError;
        }
        return {};
    }

#ifndef Q_OS_WIN
    Q_UNUSED(privateKeyBase64);
    return {};
#else
    const QByteArray privateKeyBlob = QByteArray::fromBase64(privateKeyBase64.trimmed());
    if (privateKeyBlob.isEmpty()) {
        if (error) {
            *error = QStringLiteral("Private key is empty or invalid");
        }
        return {};
    }

    const QByteArray payloadJson = QJsonDocument(payloadObject).toJson(QJsonDocument::Compact);
    const QByteArray payloadPart = base64UrlEncode(payloadJson);
    const QByteArray signingInput = QByteArray(kTokenVersion) + QByteArrayLiteral(".") + payloadPart;
    QByteArray signature;
    if (!signDigest(privateKeyBlob, sha256(signingInput), &signature, error)) {
        return {};
    }

    return QStringLiteral("%1.%2.%3")
        .arg(QString::fromLatin1(kTokenVersion),
            QString::fromLatin1(payloadPart),
            QString::fromLatin1(base64UrlEncode(signature)));
#endif
}

QString licenseStatusText(LicenseStatus status)
{
    switch (status) {
    case LicenseStatus::Valid: return QStringLiteral("Valid");
    case LicenseStatus::MissingToken: return QStringLiteral("Missing license token");
    case LicenseStatus::MissingPublicKey: return QStringLiteral("Missing public key");
    case LicenseStatus::MalformedToken: return QStringLiteral("Malformed license token");
    case LicenseStatus::PayloadInvalid: return QStringLiteral("Invalid license payload");
    case LicenseStatus::ProductMismatch: return QStringLiteral("Product mismatch");
    case LicenseStatus::MachineMismatch: return QStringLiteral("Machine mismatch");
    case LicenseStatus::Expired: return QStringLiteral("Expired license");
    case LicenseStatus::SignatureInvalid: return QStringLiteral("Invalid signature");
    case LicenseStatus::CryptoUnavailable: return QStringLiteral("Crypto unavailable");
    }
    return QStringLiteral("Unknown license status");
}

} // namespace aitrain
