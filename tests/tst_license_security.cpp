#include "aitrain/core/LicenseManager.h"
#include "aitrain/core/LicenseSecurity.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtTest>
#include <QUuid>

class LicenseSecurityTests : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void protectedPrivateKeyRoundTripAndRejectsCorruption();
    void validatesNormalPermanentAndExpiryCases();
    void rejectsTamperWrongKeyWrongMachineAndInvalidDate();
    void detectsClockRollbackWithinExplicitTolerance();
    void rejectsCorruptedTrustedClock();

private:
    aitrain::LicensePayload payload(const QDateTime& expiresAt = QDateTime()) const;
    QString tokenFor(const aitrain::LicensePayload& value) const;
    QString trustedClockPath(const QString& name) const;
    static QString replacePayloadField(const QString& token, const QString& field, const QJsonValue& value);

    QTemporaryDir temporaryDir_;
    aitrain::LicenseKeyPair keyPair_;
    aitrain::LicenseKeyPair otherKeyPair_;
    const QString machineCode_ = QStringLiteral("AAAA-BBBB-CCCC-DDDD-EEEE");
    const QDateTime nowUtc_ = QDateTime(QDate(2026, 7, 15), QTime(12, 0), Qt::UTC);
};

void LicenseSecurityTests::initTestCase()
{
    QVERIFY2(temporaryDir_.isValid(), "temporary directory must be available");
    QVERIFY2(aitrain::licenseCryptoAvailable(), "license security tests require Windows CNG and DPAPI");
    QString error;
    QVERIFY2(aitrain::generateLicenseKeyPair(&keyPair_, &error), qPrintable(error));
    QVERIFY2(aitrain::generateLicenseKeyPair(&otherKeyPair_, &error), qPrintable(error));
}

aitrain::LicensePayload LicenseSecurityTests::payload(const QDateTime& expiresAt) const
{
    aitrain::LicensePayload value;
    value.product = aitrain::licenseProductName();
    value.customer = QStringLiteral("测试客户");
    value.machineCode = machineCode_;
    value.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    value.issuedAt = nowUtc_.addDays(-1);
    value.expiresAt = expiresAt;
    return value;
}

QString LicenseSecurityTests::tokenFor(const aitrain::LicensePayload& value) const
{
    QString error;
    const QString token = aitrain::createLicenseToken(value, keyPair_.privateKeyBase64, &error);
    if (token.isEmpty()) {
        QTest::qFail(qPrintable(error), __FILE__, __LINE__);
    }
    return token;
}

QString LicenseSecurityTests::trustedClockPath(const QString& name) const
{
    return temporaryDir_.filePath(name + QStringLiteral(".dat"));
}

QString LicenseSecurityTests::replacePayloadField(
    const QString& token,
    const QString& field,
    const QJsonValue& value)
{
    QList<QByteArray> parts = token.toLatin1().split('.');
    if (parts.size() != 3) {
        return {};
    }
    QByteArray encoded = parts.at(1);
    while (encoded.size() % 4 != 0) {
        encoded.append('=');
    }
    QJsonObject object = QJsonDocument::fromJson(
        QByteArray::fromBase64(encoded, QByteArray::Base64UrlEncoding)).object();
    object.insert(field, value);
    parts[1] = QJsonDocument(object).toJson(QJsonDocument::Compact)
                   .toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals);
    return QString::fromLatin1(parts.join('.'));
}

void LicenseSecurityTests::protectedPrivateKeyRoundTripAndRejectsCorruption()
{
    const QString path = temporaryDir_.filePath(QStringLiteral("issuer.aitrainkey"));
    QString error;
    QVERIFY2(aitrain::writeProtectedLicenseKeyFile(path, keyPair_, &error), qPrintable(error));

    QFile file(path);
    QVERIFY(file.open(QIODevice::ReadOnly));
    const QByteArray stored = file.readAll();
    QVERIFY(!stored.contains(keyPair_.privateKeyBase64));
    QVERIFY(stored.contains("windows-dpapi-current-user"));
    file.close();

    aitrain::LicenseKeyPair loaded;
    QVERIFY2(aitrain::readProtectedLicenseKeyFile(path, &loaded, &error), qPrintable(error));
    QCOMPARE(loaded.publicKeyBase64, keyPair_.publicKeyBase64);
    QCOMPARE(loaded.privateKeyBase64, keyPair_.privateKeyBase64);

    QJsonObject envelope = QJsonDocument::fromJson(stored).object();
    QByteArray protectedData = QByteArray::fromBase64(
        envelope.value(QStringLiteral("protectedData")).toString().toLatin1());
    QVERIFY(!protectedData.isEmpty());
    const int changedIndex = protectedData.size() / 2;
    protectedData[changedIndex] = static_cast<char>(protectedData.at(changedIndex) ^ 0x5a);
    envelope.insert(QStringLiteral("protectedData"), QString::fromLatin1(protectedData.toBase64()));
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write(QJsonDocument(envelope).toJson(QJsonDocument::Compact));
    file.close();
    QVERIFY(!aitrain::readProtectedLicenseKeyFile(path, &loaded, &error));
    QVERIFY(!error.isEmpty());
}

void LicenseSecurityTests::validatesNormalPermanentAndExpiryCases()
{
    const QString permanent = tokenFor(payload());
    auto result = aitrain::validateLicenseTokenWithTrustedClock(
        permanent, keyPair_.publicKeyBase64, trustedClockPath(QStringLiteral("permanent")), machineCode_, nowUtc_);
    QCOMPARE(result.status, aitrain::LicenseStatus::Valid);
    QVERIFY(!result.payload.expiresAt.isValid());

    const QString future = tokenFor(payload(nowUtc_.addDays(30)));
    result = aitrain::validateLicenseTokenWithTrustedClock(
        future, keyPair_.publicKeyBase64, trustedClockPath(QStringLiteral("future")), machineCode_, nowUtc_);
    QCOMPARE(result.status, aitrain::LicenseStatus::Valid);

    const QString expired = tokenFor(payload(nowUtc_.addSecs(-1)));
    result = aitrain::validateLicenseTokenWithTrustedClock(
        expired, keyPair_.publicKeyBase64, trustedClockPath(QStringLiteral("expired")), machineCode_, nowUtc_);
    QCOMPARE(result.status, aitrain::LicenseStatus::Expired);
}

void LicenseSecurityTests::rejectsTamperWrongKeyWrongMachineAndInvalidDate()
{
    const QString token = tokenFor(payload(nowUtc_.addDays(1)));
    QList<QByteArray> tamperedParts = token.toLatin1().split('.');
    QVERIFY(tamperedParts.size() == 3);
    QVERIFY(!tamperedParts[2].isEmpty());
    tamperedParts[2][0] = tamperedParts[2].at(0) == 'A' ? 'B' : 'A';
    const QString tampered = QString::fromLatin1(tamperedParts.join('.'));
    QCOMPARE(
        aitrain::validateLicenseToken(tampered, keyPair_.publicKeyBase64, machineCode_, nowUtc_).status,
        aitrain::LicenseStatus::SignatureInvalid);
    QCOMPARE(
        aitrain::validateLicenseToken(token, otherKeyPair_.publicKeyBase64, machineCode_, nowUtc_).status,
        aitrain::LicenseStatus::SignatureInvalid);
    QCOMPARE(
        aitrain::validateLicenseToken(token, keyPair_.publicKeyBase64, QStringLiteral("FFFF-EEEE-DDDD-CCCC-BBBB"), nowUtc_).status,
        aitrain::LicenseStatus::MachineMismatch);

    const QString invalidDate = replacePayloadField(token, QStringLiteral("expiresAt"), QStringLiteral("not-a-date"));
    QCOMPARE(
        aitrain::validateLicenseToken(invalidDate, keyPair_.publicKeyBase64, machineCode_, nowUtc_).status,
        aitrain::LicenseStatus::PayloadInvalid);
}

void LicenseSecurityTests::detectsClockRollbackWithinExplicitTolerance()
{
    const QString path = trustedClockPath(QStringLiteral("rollback"));
    const QString token = tokenFor(payload(nowUtc_.addDays(2)));
    QCOMPARE(
        aitrain::validateLicenseTokenWithTrustedClock(
            token, keyPair_.publicKeyBase64, path, machineCode_, nowUtc_, 300).status,
        aitrain::LicenseStatus::Valid);
    QCOMPARE(
        aitrain::validateLicenseTokenWithTrustedClock(
            token, keyPair_.publicKeyBase64, path, machineCode_, nowUtc_.addSecs(-299), 300).status,
        aitrain::LicenseStatus::Valid);
    QCOMPARE(
        aitrain::validateLicenseTokenWithTrustedClock(
            token, keyPair_.publicKeyBase64, path, machineCode_, nowUtc_.addSecs(-301), 300).status,
        aitrain::LicenseStatus::ClockRollbackDetected);
}

void LicenseSecurityTests::rejectsCorruptedTrustedClock()
{
    const QString path = trustedClockPath(QStringLiteral("corrupt-clock"));
    const QString token = tokenFor(payload(nowUtc_.addDays(2)));
    QCOMPARE(
        aitrain::validateLicenseTokenWithTrustedClock(
            token, keyPair_.publicKeyBase64, path, machineCode_, nowUtc_).status,
        aitrain::LicenseStatus::Valid);

    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Truncate));
    file.write("{\"type\":\"aitrain-trusted-utc\",\"protectedData\":\"corrupted\"}");
    file.close();
    QCOMPARE(
        aitrain::validateLicenseTokenWithTrustedClock(
            token, keyPair_.publicKeyBase64, path, machineCode_, nowUtc_.addSecs(1)).status,
        aitrain::LicenseStatus::ProtectedStorageCorrupted);
}

QTEST_GUILESS_MAIN(LicenseSecurityTests)
#include "tst_license_security.moc"
