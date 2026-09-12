#include "SettingsPageController.h"

#include "LanguageSupport.h"
#include "AppStyle.h"
#include <QApplication>
#include <QSettings>
#include "MainWindowSupport.h"
#include "SettingsPage.h"
#include "aitrain/core/CapabilityRegistry.h"

#include <QDir>
#include <QFileDialog>
#include <QMessageBox>

using namespace aitrain_app;

SettingsPageController::SettingsPageController(QString defaultProjectPath,
    QObject* parent)
    : QObject(parent)
    , defaultProjectPath_(QDir::cleanPath(std::move(defaultProjectPath)))
{
}

void SettingsPageController::attach(SettingsWorkspacePage* page)
{
    page_ = page;
    connect(page_, &SettingsWorkspacePage::themeRequested, this, [](const QString& theme) {
        QSettings settings; settings.setValue(QStringLiteral("settings/theme"), theme); settings.sync();
        AppStyle::apply(*qApp, theme);
    });
    connect(page_, &SettingsWorkspacePage::refreshCapabilitiesRequested,
        this, &SettingsPageController::refreshCapabilities);
    connect(page_, &SettingsWorkspacePage::languageRequested,
        this, &SettingsPageController::setLanguageCode);
    connect(page_, &SettingsWorkspacePage::browseDefaultProjectPathRequested,
        this, &SettingsPageController::browseDefaultProjectPath);
    connect(page_, &SettingsWorkspacePage::saveDefaultProjectPathRequested,
        this, &SettingsPageController::saveDefaultProjectPath);
    connect(page_, &SettingsWorkspacePage::resetDefaultProjectPathRequested,
        this, &SettingsPageController::resetDefaultProjectPath);
    refresh();
}

QString SettingsPageController::configuredDefaultProjectPath() const
{
    return normalizedDefaultProjectPath(
        settings_.defaultProjectPath(defaultProjectPath_));
}

void SettingsPageController::refresh()
{
    if (!page_) {
        return;
    }
    page_->setDefaultProjectPathText(configuredDefaultProjectPath());
    page_->setLanguageCode(configuredLanguageCode());
    page_->setThemeCode(AppStyle::configuredTheme());
    refreshCapabilities();
}

void SettingsPageController::refreshCapabilities()
{
    if (!page_) {
        return;
    }
    QVector<SettingsCapabilityRow> rows;
    SettingsCapabilitySummary summary;
    QStringList datasetFormats;
    QStringList exportFormats;
    const QVector<aitrain::CapabilityDescriptor> capabilities =
        aitrain::BuiltinCapabilityRegistry::instance().capabilities();
    rows.reserve(capabilities.size());
    for (const aitrain::CapabilityDescriptor& capability : capabilities) {
        SettingsCapabilityRow row;
        row.id = capability.id;
        row.displayName = capability.displayName;
        row.taskTypes = compactListSummary(capability.taskTypes, 4);
        row.datasetFormats = compactListSummary(capability.datasetFormats, 4);
        row.backendIds = compactListSummary(capability.backendIds, 4);
        rows.append(row);
        datasetFormats.append(capability.datasetFormats);
        for (const QString& backendId : capability.backendIds) {
            const aitrain::BackendDescriptor backend =
                aitrain::BuiltinCapabilityRegistry::instance().backend(backendId);
            exportFormats.append(backend.exportFormats);
            if (backend.devicePolicy == QStringLiteral("gpu_required")
                || backend.devicePolicy == QStringLiteral("gpu_recommended")) {
                ++summary.gpuCapabilityCount;
            }
        }
    }
    summary.capabilityCount = capabilities.size();
    summary.datasetFormatCount = uniqueStringCount(datasetFormats);
    summary.exportFormatCount = uniqueStringCount(exportFormats);
    summary.datasetFormats = compactListSummary(datasetFormats, 12);
    summary.exportFormats = compactListSummary(exportFormats, 12);
    page_->setCapabilities(rows, summary);
}

void SettingsPageController::showTab(int tabIndex)
{
    if (page_) {
        page_->showTab(tabIndex);
    }
}

void SettingsPageController::browseDefaultProjectPath()
{
    const QString directory = QFileDialog::getExistingDirectory(
        page_, tr("请选择默认项目目录"),
        QDir::fromNativeSeparators(page_->defaultProjectPathText().trimmed()));
    if (!directory.isEmpty()) {
        page_->setDefaultProjectPathText(directory);
    }
}

void SettingsPageController::saveDefaultProjectPath()
{
    const QString normalized =
        normalizedDefaultProjectPath(page_->defaultProjectPathText());
    if (normalized.isEmpty() || normalized == QStringLiteral(".")) {
        QMessageBox::warning(page_, tr("默认项目目录"), tr("目录不能为空。"));
        return;
    }
    settings_.setDefaultProjectPath(normalized);
    page_->setDefaultProjectPathText(normalized);
    page_->setDefaultProjectPathStatus(tr("默认项目目录已保存。"));
    emit defaultProjectPathChanged(normalized);
}

void SettingsPageController::resetDefaultProjectPath()
{
    settings_.setDefaultProjectPath(defaultProjectPath_);
    page_->setDefaultProjectPathText(defaultProjectPath_);
    page_->setDefaultProjectPathStatus(tr("默认项目目录已恢复。"));
    emit defaultProjectPathChanged(defaultProjectPath_);
}

void SettingsPageController::setLanguageCode(const QString& languageCode)
{
    const QString previous = configuredLanguageCode();
    storeLanguageCode(languageCode);
    const QString current = configuredLanguageCode();
    page_->setLanguageCode(current);
    emit languageChanged(current);
    if (previous != current) {
        QMessageBox::information(page_, tr("界面语言"),
            tr("语言设置已保存，重启 AITrain Studio 后生效。"));
    }
}

QString SettingsPageController::normalizedDefaultProjectPath(
    const QString& path) const
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty()) {
        return QString();
    }
    return QDir::cleanPath(QDir::fromNativeSeparators(trimmed));
}
