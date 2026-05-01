#pragma once

#include <QMainWindow>
#include <QByteArray>

class QCheckBox;
class QDateEdit;
class QLabel;
class QLineEdit;
class QPlainTextEdit;

class LicenseGeneratorWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit LicenseGeneratorWindow(QWidget* parent = nullptr);

private slots:
    void generateKeyFile();
    void loadKeyFile();
    void copyPublicKey();
    void generateLicense();
    void copyLicense();
    void saveLicense();

private:
    void setStatus(const QString& message);
    void updateKeyFields(const QByteArray& privateKeyBase64, const QByteArray& publicKeyBase64, const QString& sourcePath);
    bool ensurePrivateKeyLoaded();

    QByteArray privateKeyBase64_;
    QByteArray publicKeyBase64_;

    QLineEdit* keyPathEdit_ = nullptr;
    QLineEdit* customerEdit_ = nullptr;
    QLineEdit* machineCodeEdit_ = nullptr;
    QCheckBox* expiryCheck_ = nullptr;
    QDateEdit* expiryDateEdit_ = nullptr;
    QPlainTextEdit* publicKeyEdit_ = nullptr;
    QPlainTextEdit* licenseEdit_ = nullptr;
    QLabel* statusLabel_ = nullptr;
};
