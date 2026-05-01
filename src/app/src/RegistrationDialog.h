#pragma once

#include "aitrain/core/LicenseManager.h"

#include <QDialog>
#include <QByteArray>

class QComboBox;
class QLabel;
class QPlainTextEdit;

class RegistrationDialog : public QDialog {
    Q_OBJECT

public:
    explicit RegistrationDialog(const QByteArray& publicKeyBase64, QWidget* parent = nullptr);
    aitrain::LicensePayload activatedPayload() const;

private slots:
    void copyMachineCode();
    void activateLicense();
    void handleLanguageChanged(int index);

private:
    QString localizedLicenseMessage(aitrain::LicenseStatus status, const QString& fallback) const;

    QByteArray publicKeyBase64_;
    QString machineCode_;
    aitrain::LicensePayload activatedPayload_;

    QLabel* machineCodeLabel_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QPlainTextEdit* tokenEdit_ = nullptr;
    QComboBox* languageCombo_ = nullptr;
};
