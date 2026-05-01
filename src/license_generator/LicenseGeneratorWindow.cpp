#include "LicenseGeneratorWindow.h"

#include "aitrain/core/LicenseManager.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDate>
#include <QDateEdit>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTime>
#include <QVBoxLayout>
#include <QUuid>

namespace {

QPushButton* primaryButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("PrimaryButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

bool writeJsonFile(const QString& path, const QJsonObject& object, QString* error)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error) {
            *error = file.errorString();
        }
        return false;
    }
    file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    return true;
}

} // namespace

LicenseGeneratorWindow::LicenseGeneratorWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("AITrain 注册码生成器"));
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(22, 22, 22, 22);
    layout->setSpacing(14);

    auto* title = new QLabel(QStringLiteral("离线注册码签发"));
    title->setObjectName(QStringLiteral("PageTitle"));
    auto* hint = new QLabel(QStringLiteral("生成或加载 ECDSA P-256 私钥，输入客户名称和机器码后签发注册码。生成器默认不随客户包安装。"));
    hint->setObjectName(QStringLiteral("MutedText"));
    hint->setWordWrap(true);

    auto* keyRow = new QHBoxLayout;
    keyPathEdit_ = new QLineEdit;
    keyPathEdit_->setReadOnly(true);
    keyPathEdit_->setPlaceholderText(QStringLiteral("尚未加载私钥文件"));
    auto* loadKeyButton = new QPushButton(QStringLiteral("加载私钥"));
    auto* generateKeyButton = primaryButton(QStringLiteral("生成私钥"));
    keyRow->addWidget(keyPathEdit_, 1);
    keyRow->addWidget(loadKeyButton);
    keyRow->addWidget(generateKeyButton);

    publicKeyEdit_ = new QPlainTextEdit;
    publicKeyEdit_->setReadOnly(true);
    publicKeyEdit_->setPlaceholderText(QStringLiteral("公钥会显示在这里，用于配置 CMake 变量 AITRAIN_LICENSE_PUBLIC_KEY"));
    publicKeyEdit_->setFixedHeight(88);
    auto* copyPublicButton = new QPushButton(QStringLiteral("复制公钥"));

    customerEdit_ = new QLineEdit;
    customerEdit_->setPlaceholderText(QStringLiteral("客户或组织名称"));
    machineCodeEdit_ = new QLineEdit;
    machineCodeEdit_->setPlaceholderText(QStringLiteral("来自注册窗口的机器码"));
    expiryCheck_ = new QCheckBox(QStringLiteral("设置到期日期"));
    expiryDateEdit_ = new QDateEdit(QDate::currentDate().addYears(1));
    expiryDateEdit_->setCalendarPopup(true);
    expiryDateEdit_->setEnabled(false);

    auto* form = new QFormLayout;
    form->setLabelAlignment(Qt::AlignRight);
    form->addRow(QStringLiteral("客户名称"), customerEdit_);
    form->addRow(QStringLiteral("机器码"), machineCodeEdit_);
    auto* expiryRow = new QHBoxLayout;
    expiryRow->addWidget(expiryCheck_);
    expiryRow->addWidget(expiryDateEdit_);
    expiryRow->addStretch();
    form->addRow(QStringLiteral("有效期"), expiryRow);

    licenseEdit_ = new QPlainTextEdit;
    licenseEdit_->setPlaceholderText(QStringLiteral("生成的注册码会显示在这里"));
    licenseEdit_->setFixedHeight(150);

    auto* actionRow = new QHBoxLayout;
    auto* generateLicenseButton = primaryButton(QStringLiteral("生成注册码"));
    auto* copyLicenseButton = new QPushButton(QStringLiteral("复制注册码"));
    auto* saveLicenseButton = new QPushButton(QStringLiteral("保存注册码"));
    actionRow->addWidget(generateLicenseButton);
    actionRow->addWidget(copyLicenseButton);
    actionRow->addWidget(saveLicenseButton);
    actionRow->addStretch();

    statusLabel_ = new QLabel(QStringLiteral("准备就绪。"));
    statusLabel_->setObjectName(QStringLiteral("InlineStatus"));
    statusLabel_->setWordWrap(true);

    layout->addWidget(title);
    layout->addWidget(hint);
    layout->addWidget(new QLabel(QStringLiteral("私钥文件")));
    layout->addLayout(keyRow);
    layout->addWidget(new QLabel(QStringLiteral("应用公钥")));
    layout->addWidget(publicKeyEdit_);
    layout->addWidget(copyPublicButton, 0, Qt::AlignRight);
    layout->addLayout(form);
    layout->addLayout(actionRow);
    layout->addWidget(licenseEdit_);
    layout->addWidget(statusLabel_);
    setCentralWidget(central);

    connect(generateKeyButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::generateKeyFile);
    connect(loadKeyButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::loadKeyFile);
    connect(copyPublicButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::copyPublicKey);
    connect(generateLicenseButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::generateLicense);
    connect(copyLicenseButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::copyLicense);
    connect(saveLicenseButton, &QPushButton::clicked, this, &LicenseGeneratorWindow::saveLicense);
    connect(expiryCheck_, &QCheckBox::toggled, expiryDateEdit_, &QDateEdit::setEnabled);
}

void LicenseGeneratorWindow::generateKeyFile()
{
    const QString path = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("保存私钥文件"),
        QDir::home().filePath(QStringLiteral("aitrain-license-private-key.json")),
        QStringLiteral("JSON (*.json)"));
    if (path.isEmpty()) {
        return;
    }

    QString error;
    aitrain::LicenseKeyPair keyPair;
    if (!aitrain::generateLicenseKeyPair(&keyPair, &error)) {
        QMessageBox::critical(this, QStringLiteral("生成私钥"), error);
        return;
    }

    QJsonObject object;
    object.insert(QStringLiteral("type"), QStringLiteral("aitrain-license-key"));
    object.insert(QStringLiteral("curve"), QStringLiteral("P-256"));
    object.insert(QStringLiteral("createdAt"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate));
    object.insert(QStringLiteral("publicKey"), QString::fromLatin1(keyPair.publicKeyBase64));
    object.insert(QStringLiteral("privateKey"), QString::fromLatin1(keyPair.privateKeyBase64));
    if (!writeJsonFile(path, object, &error)) {
        QMessageBox::critical(this, QStringLiteral("保存私钥"), error);
        return;
    }

    updateKeyFields(keyPair.privateKeyBase64, keyPair.publicKeyBase64, path);
    setStatus(QStringLiteral("私钥已生成并保存。请妥善保管该文件，不要随客户包分发。"));
}

void LicenseGeneratorWindow::loadKeyFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("加载私钥文件"),
        QDir::homePath(),
        QStringLiteral("JSON or key files (*.json *.key *.txt);;All files (*.*)"));
    if (path.isEmpty()) {
        return;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, QStringLiteral("加载私钥"), file.errorString());
        return;
    }
    const QByteArray data = file.readAll();
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(data, &parseError);

    QByteArray privateKey;
    QByteArray publicKey;
    if (document.isObject()) {
        const QJsonObject object = document.object();
        privateKey = object.value(QStringLiteral("privateKey")).toString().toLatin1();
        publicKey = object.value(QStringLiteral("publicKey")).toString().toLatin1();
    } else {
        privateKey = data.trimmed();
    }
    QString error;
    if (privateKey.isEmpty()) {
        QMessageBox::critical(this, QStringLiteral("加载私钥"), QStringLiteral("文件中没有 privateKey 字段。"));
        return;
    }
    if (publicKey.isEmpty()) {
        publicKey = aitrain::publicKeyFromPrivateKey(privateKey, &error);
        if (publicKey.isEmpty()) {
            QMessageBox::critical(this, QStringLiteral("加载私钥"), error);
            return;
        }
    }

    updateKeyFields(privateKey, publicKey, path);
    setStatus(QStringLiteral("私钥已加载。"));
}

void LicenseGeneratorWindow::copyPublicKey()
{
    if (publicKeyBase64_.isEmpty()) {
        setStatus(QStringLiteral("没有可复制的公钥。"));
        return;
    }
    QApplication::clipboard()->setText(QString::fromLatin1(publicKeyBase64_));
    setStatus(QStringLiteral("公钥已复制。"));
}

void LicenseGeneratorWindow::generateLicense()
{
    if (!ensurePrivateKeyLoaded()) {
        return;
    }
    const QString customer = customerEdit_->text().trimmed();
    const QString machineCode = aitrain::normalizeMachineCode(machineCodeEdit_->text().trimmed());
    if (customer.isEmpty() || machineCode.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("生成注册码"), QStringLiteral("客户名称和机器码不能为空。"));
        return;
    }

    aitrain::LicensePayload payload;
    payload.product = aitrain::licenseProductName();
    payload.customer = customer;
    payload.machineCode = machineCode;
    payload.licenseId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    payload.issuedAt = QDateTime::currentDateTimeUtc();
    if (expiryCheck_->isChecked()) {
        payload.expiresAt = QDateTime(expiryDateEdit_->date(), QTime(23, 59, 59), Qt::LocalTime).toUTC();
    }

    QString error;
    const QString token = aitrain::createLicenseToken(payload, privateKeyBase64_, &error);
    if (token.isEmpty()) {
        QMessageBox::critical(this, QStringLiteral("生成注册码"), error);
        return;
    }
    licenseEdit_->setPlainText(token);
    setStatus(QStringLiteral("注册码已生成。"));
}

void LicenseGeneratorWindow::copyLicense()
{
    const QString token = licenseEdit_->toPlainText().trimmed();
    if (token.isEmpty()) {
        setStatus(QStringLiteral("没有可复制的注册码。"));
        return;
    }
    QApplication::clipboard()->setText(token);
    setStatus(QStringLiteral("注册码已复制。"));
}

void LicenseGeneratorWindow::saveLicense()
{
    const QString token = licenseEdit_->toPlainText().trimmed();
    if (token.isEmpty()) {
        setStatus(QStringLiteral("请先生成注册码。"));
        return;
    }
    const QString path = QFileDialog::getSaveFileName(
        this,
        QStringLiteral("保存注册码"),
        QDir::home().filePath(QStringLiteral("aitrain-license.txt")),
        QStringLiteral("Text (*.txt);;All files (*.*)"));
    if (path.isEmpty()) {
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        QMessageBox::critical(this, QStringLiteral("保存注册码"), file.errorString());
        return;
    }
    file.write(token.toUtf8());
    setStatus(QStringLiteral("注册码已保存。"));
}

void LicenseGeneratorWindow::setStatus(const QString& message)
{
    statusLabel_->setText(message);
}

void LicenseGeneratorWindow::updateKeyFields(
    const QByteArray& privateKeyBase64,
    const QByteArray& publicKeyBase64,
    const QString& sourcePath)
{
    privateKeyBase64_ = privateKeyBase64.trimmed();
    publicKeyBase64_ = publicKeyBase64.trimmed();
    keyPathEdit_->setText(QDir::toNativeSeparators(sourcePath));
    publicKeyEdit_->setPlainText(QString::fromLatin1(publicKeyBase64_));
}

bool LicenseGeneratorWindow::ensurePrivateKeyLoaded()
{
    if (!privateKeyBase64_.isEmpty()) {
        return true;
    }
    QMessageBox::warning(this, QStringLiteral("私钥"), QStringLiteral("请先生成或加载私钥文件。"));
    return false;
}
