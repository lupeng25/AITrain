#include "RegistrationDialog.h"

#include "LanguageSupport.h"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>

namespace {

QPushButton* primaryDialogButton(const QString& text)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("PrimaryButton"));
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

} // namespace

RegistrationDialog::RegistrationDialog(const QByteArray& publicKeyBase64, QWidget* parent)
    : QDialog(parent)
    , publicKeyBase64_(publicKeyBase64)
    , machineCode_(aitrain::currentMachineCode())
{
    setWindowTitle(tr("AITrain Studio 注册"));
    setModal(true);
    resize(620, 460);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(22, 22, 22, 22);
    layout->setSpacing(14);

    auto* title = new QLabel(tr("离线授权验证"));
    title->setObjectName(QStringLiteral("PageTitle"));
    auto* description = new QLabel(tr("请把本机机器码发送给授权方，收到注册码后粘贴到下方。验证通过后会进入主界面。"));
    description->setObjectName(QStringLiteral("MutedText"));
    description->setWordWrap(true);

    auto* languageRow = new QHBoxLayout;
    auto* languageLabel = new QLabel(tr("界面语言"));
    languageCombo_ = new QComboBox;
    languageCombo_->addItem(QStringLiteral("中文"), QStringLiteral("zh_CN"));
    languageCombo_->addItem(QStringLiteral("English"), QStringLiteral("en_US"));
    const QString language = aitrain_app::configuredLanguageCode();
    const int languageIndex = languageCombo_->findData(language);
    if (languageIndex >= 0) {
        languageCombo_->setCurrentIndex(languageIndex);
    }
    languageRow->addWidget(languageLabel);
    languageRow->addWidget(languageCombo_, 1);

    auto* machineRow = new QHBoxLayout;
    machineCodeLabel_ = new QLabel(machineCode_);
    machineCodeLabel_->setObjectName(QStringLiteral("InlineStatus"));
    machineCodeLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* copyButton = new QPushButton(tr("复制机器码"));
    copyButton->setCursor(Qt::PointingHandCursor);
    machineRow->addWidget(machineCodeLabel_, 1);
    machineRow->addWidget(copyButton);

    auto* tokenLabel = new QLabel(tr("注册码"));
    tokenEdit_ = new QPlainTextEdit;
    tokenEdit_->setPlaceholderText(tr("粘贴 AITRAIN1 开头的离线注册码"));
    tokenEdit_->setFixedHeight(140);
    statusLabel_ = new QLabel(tr("未注册：请输入注册码。"));
    statusLabel_->setObjectName(QStringLiteral("InlineStatus"));
    statusLabel_->setWordWrap(true);

    auto* actions = new QHBoxLayout;
    auto* cancelButton = new QPushButton(tr("退出"));
    cancelButton->setCursor(Qt::PointingHandCursor);
    auto* activateButton = primaryDialogButton(tr("验证并启动"));
    actions->addStretch();
    actions->addWidget(cancelButton);
    actions->addWidget(activateButton);

    layout->addWidget(title);
    layout->addWidget(description);
    layout->addLayout(languageRow);
    layout->addWidget(new QLabel(tr("本机机器码")));
    layout->addLayout(machineRow);
    layout->addWidget(tokenLabel);
    layout->addWidget(tokenEdit_);
    layout->addWidget(statusLabel_);
    layout->addLayout(actions);

    connect(copyButton, &QPushButton::clicked, this, &RegistrationDialog::copyMachineCode);
    connect(cancelButton, &QPushButton::clicked, this, &RegistrationDialog::reject);
    connect(activateButton, &QPushButton::clicked, this, &RegistrationDialog::activateLicense);
    connect(languageCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RegistrationDialog::handleLanguageChanged);

    aitrain_app::translateWidgetTree(this, "RegistrationDialog");
}

aitrain::LicensePayload RegistrationDialog::activatedPayload() const
{
    return activatedPayload_;
}

void RegistrationDialog::copyMachineCode()
{
    QApplication::clipboard()->setText(machineCode_);
    statusLabel_->setText(tr("机器码已复制。"));
}

void RegistrationDialog::activateLicense()
{
    const QString token = tokenEdit_->toPlainText().trimmed();
    const aitrain::LicenseValidationResult validation =
        aitrain::validateLicenseToken(token, publicKeyBase64_, machineCode_);
    if (!validation.isValid()) {
        statusLabel_->setText(localizedLicenseMessage(validation.status, validation.message));
        return;
    }

    QSettings settings;
    settings.setValue(QStringLiteral("license/token"), token);
    settings.setValue(QStringLiteral("license/customer"), validation.payload.customer);
    settings.setValue(QStringLiteral("license/licenseId"), validation.payload.licenseId);
    if (validation.payload.expiresAt.isValid()) {
        settings.setValue(QStringLiteral("license/expiresAt"), validation.payload.expiresAt.toUTC().toString(Qt::ISODate));
    } else {
        settings.remove(QStringLiteral("license/expiresAt"));
    }
    settings.sync();

    activatedPayload_ = validation.payload;
    statusLabel_->setText(tr("注册成功，正在启动主界面。"));
    accept();
}

void RegistrationDialog::handleLanguageChanged(int index)
{
    const QString selected = languageCombo_->itemData(index).toString();
    if (selected.isEmpty()) {
        return;
    }
    const QString previous = aitrain_app::configuredLanguageCode();
    aitrain_app::storeLanguageCode(selected);
    if (previous != selected) {
        statusLabel_->setText(tr("语言设置已保存，重启后生效。"));
    }
}

QString RegistrationDialog::localizedLicenseMessage(aitrain::LicenseStatus status, const QString& fallback) const
{
    switch (status) {
    case aitrain::LicenseStatus::MissingToken:
        return tr("请输入注册码。");
    case aitrain::LicenseStatus::MissingPublicKey:
        return tr("应用未配置授权公钥，请检查构建配置。");
    case aitrain::LicenseStatus::MalformedToken:
        return tr("注册码格式不正确。");
    case aitrain::LicenseStatus::PayloadInvalid:
        return tr("注册码载荷无效。");
    case aitrain::LicenseStatus::ProductMismatch:
        return tr("注册码不属于 AITrain Studio。");
    case aitrain::LicenseStatus::MachineMismatch:
        return tr("注册码与本机机器码不匹配。");
    case aitrain::LicenseStatus::Expired:
        return tr("注册码已过期。");
    case aitrain::LicenseStatus::SignatureInvalid:
        return tr("注册码签名验证失败。");
    case aitrain::LicenseStatus::CryptoUnavailable:
        return tr("当前平台不支持离线授权验证。");
    case aitrain::LicenseStatus::Valid:
        return tr("注册成功。");
    }
    return fallback.isEmpty() ? tr("注册验证失败。") : fallback;
}
