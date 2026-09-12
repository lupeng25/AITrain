#include "WorkbenchTranslation.h"
#include "TrainingPageController.h"
#include "TrainingPage.h"
#include "ApplicationSettingsService.h"
#include "ProjectObjectSelectors.h"
#include "aitrain/product/ProductCapabilityContract.h"
#include "aitrain/workflow/ProjectQueryService.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QDialog>
#include <QDialogButtonBox>
#include <QSignalBlocker>
#include <QTimer>

namespace {
bool draftControl(const QString& name)
{
    return name == QStringLiteral("TrainingBackend") || name == QStringLiteral("TrainingModelPreset")
        || name == QStringLiteral("TrainingEpochs") || name == QStringLiteral("TrainingBatchSize")
        || name == QStringLiteral("TrainingImageSize") || name.contains(QStringLiteral("TrainArg_"))
        || name.startsWith(QStringLiteral("YoloTrainExportArg_"));
}
}

void TrainingPageController::initializeDraftPersistence()
{
    initialDraftControls_ = draftControls();
    page_->findChild<QPushButton*>(QStringLiteral("TrainingSaveDraft"))->setEnabled(!draftProjectId_.isEmpty());
    page_->findChild<QPushButton*>(QStringLiteral("TrainingDiscardDraft"))->setEnabled(!draftProjectId_.isEmpty());
    draftTimer_ = new QTimer(this); draftTimer_->setSingleShot(true); draftTimer_->setInterval(500);
    connect(draftTimer_, &QTimer::timeout, this, &TrainingPageController::saveDraft);
    connect(qApp, &QCoreApplication::aboutToQuit, this, &TrainingPageController::saveDraft);
    for (auto* widget : page_->findChildren<QWidget*>()) {
        if (!draftControl(widget->objectName())) continue;
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) connect(edit, &QLineEdit::textChanged, this, &TrainingPageController::scheduleDraftSave);
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) connect(combo, &QComboBox::currentTextChanged, this, &TrainingPageController::scheduleDraftSave);
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) connect(check, &QCheckBox::toggled, this, &TrainingPageController::scheduleDraftSave);
    }
    connect(page_->findChild<QPushButton*>(QStringLiteral("TrainingSaveDraft")), &QPushButton::clicked, this, [this]() { draftDirty_ = true; saveDraft(); });
    connect(page_->findChild<QPushButton*>(QStringLiteral("TrainingDiscardDraft")), &QPushButton::clicked, this, &TrainingPageController::discardDraft);
}

QJsonObject TrainingPageController::draftControls() const
{
    QJsonObject result;
    if (!page_) return result;
    for (auto* widget : page_->findChildren<QWidget*>()) {
        const QString name = widget->objectName(); if (!draftControl(name)) continue;
        const bool pending = advancedBackup_.contains(name);
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) result.insert(name, pending ? advancedBackup_.value(name).toString() : edit->text());
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) result.insert(name, pending ? advancedBackup_.value(name).toBool() : check->isChecked());
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            const int index = pending ? advancedBackup_.value(name).toInt() : combo->currentIndex();
            // 非编辑项保存稳定 data，不保存会随语言变化的显示标签或索引。
            result.insert(name, combo->isEditable() ? combo->currentText() : combo->itemData(index).toString());
        }
    }
    return result;
}

void TrainingPageController::scheduleDraftSave()
{
    if (restoringDraft_ || !page_ || draftProjectId_.isEmpty()) return;
    draftDirty_ = true; if (draftTimer_) draftTimer_->start();
}

void TrainingPageController::saveDraft()
{
    if (!draftDirty_ || restoringDraft_ || !page_ || draftProjectId_.isEmpty()) return;
    if (draftTimer_) draftTimer_->stop();
    QJsonObject draft{{QStringLiteral("controls"), draftControls()},
        {QStringLiteral("datasetId"), binding_.datasetId}, {QStringLiteral("datasetVersionId"), binding_.datasetVersionId},
        {QStringLiteral("datasetSnapshotId"), binding_.snapshotId}, {QStringLiteral("datasetSnapshotArtifactId"), binding_.snapshotArtifactId},
        {QStringLiteral("sample"), binding_.deploymentSampleRelativePath}};
    aitrain_app::ApplicationSettingsService settings;
    const bool saved = settings.saveTrainingDraft(draftProjectId_, draft);
    draftDirty_ = !saved;
    page_->findChild<QLabel*>(QStringLiteral("TrainingDraftStatus"))->setText(saved
        ? aitrain_app::workbenchText(QStringLiteral("草稿已保存。重启后恢复已应用的参数。")) : aitrain_app::workbenchText(QStringLiteral("草稿保存失败，请检查用户设置目录。")));
}

void TrainingPageController::applyDraftControls(const QJsonObject& controls)
{
    if (!page_) return;
    auto* backend = page_->findChild<QComboBox*>(QStringLiteral("TrainingBackend"));
    const int index = backend->findData(controls.value(QStringLiteral("TrainingBackend")).toString());
    if (index >= 0) backend->setCurrentIndex(index);
    synchronizeBackend();
    for (auto* widget : page_->findChildren<QWidget*>()) {
        const QString name = widget->objectName();
        if (!draftControl(name) || name == QStringLiteral("TrainingBackend") || !controls.contains(name)) continue;
        const QJsonValue value = controls.value(name); const QSignalBlocker blocker(widget);
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) { if (value.isString() && value.toString().size() <= 8192) edit->setText(value.toString()); }
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) { if (value.isBool()) check->setChecked(value.toBool()); }
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            if (!value.isString() || value.toString().size() > 8192) continue;
            const int item = combo->findData(value.toString());
            if (combo->isEditable()) combo->setCurrentText(value.toString()); else if (item >= 0) combo->setCurrentIndex(item);
        }
    }
    if (backend->currentData().toString() == QStringLiteral("anomalib_efficientad")) page_->findChild<QLineEdit*>(QStringLiteral("TrainingBatchSize"))->setText(QStringLiteral("1"));
    refreshSummary();
}

void TrainingPageController::resetDraftControls()
{
    modelPresetBackend_.clear(); refreshDefaults(); applyDraftControls(initialDraftControls_);
}

void TrainingPageController::restoreDraft()
{
    if (!page_ || draftProjectId_.isEmpty()) return;
    aitrain_app::ApplicationSettingsService settings; QJsonObject draft; QString error;
    auto* status = page_->findChild<QLabel*>(QStringLiteral("TrainingDraftStatus"));
    status->setText(aitrain_app::workbenchText(QStringLiteral("草稿按项目自动保存，可在重启后恢复。")));
    if (!settings.readTrainingDraft(draftProjectId_, &draft, &error)) { if (!error.isEmpty()) status->setText(error); return; }
    TrainingDatasetBinding binding; QString label;
    const bool hadBinding = !draft.value(QStringLiteral("datasetSnapshotId")).toString().isEmpty();
    const bool validBinding = historyBinding(draft, &binding, &label);
    if (validBinding) {
        const QString sample = draft.value(QStringLiteral("sample")).toString();
        aitrain::ArtifactId artifact; QString cursor;
        if (aitrain_app::isImageMember(sample) && aitrain::ArtifactId::parse(binding.snapshotArtifactId, &artifact, &error)) do {
            const auto files = queryService_->artifactFiles(artifact, {100, cursor}, &error);
            if (!error.isEmpty()) break;
            for (const auto& file : files.items) if (file.relativePath == sample) binding.deploymentSampleRelativePath = sample;
            cursor = files.hasMore ? files.nextCursor : QString();
        } while (binding.deploymentSampleRelativePath.isEmpty() && !cursor.isEmpty());
        binding_ = binding; refreshDefaults();
    }
    const auto controls = draft.value(QStringLiteral("controls")).toObject();
    aitrain::TrainingBackendContract backend;
    const bool validBackend = aitrain::ProductCapabilityContract::instance().resolveTrainingBackend(
        controls.value(QStringLiteral("TrainingBackend")).toString(), &backend)
        && (!validBinding || backend.datasetFormat == binding.datasetFormat);
    if (!validBackend) { binding_ = {}; refreshDefaults(); }
    applyDraftControls(controls);
    status->setText(hadBinding && !validBinding ? aitrain_app::workbenchText(QStringLiteral("已恢复参数；原数据版本已失效，请重新选择数据。"))
        : validBinding && !draft.value(QStringLiteral("sample")).toString().isEmpty() && binding_.deploymentSampleRelativePath.isEmpty()
        ? aitrain_app::workbenchText(QStringLiteral("已恢复参数和数据版本；原样本已失效，请重新选择样本。"))
        : aitrain_app::workbenchText(QStringLiteral("已恢复本项目草稿。检查参数后可开始训练。")));
    if (!validBackend) status->setText(aitrain_app::workbenchText(QStringLiteral("原后端与当前合同不匹配，请重新选择算法和数据版本。")));
}

void TrainingPageController::discardDraft()
{
    if (!page_ || draftProjectId_.isEmpty()) return;
    QDialog confirmation(page_);
    confirmation.setObjectName(QStringLiteral("TrainingDiscardConfirmation"));
    confirmation.setWindowTitle(aitrain_app::workbenchText(QStringLiteral("丢弃草稿")));
    auto* layout = new QVBoxLayout(&confirmation);
    auto* message = new QLabel(aitrain_app::workbenchText(QStringLiteral("清除本项目草稿并恢复默认参数？")));
    message->setWordWrap(true); layout->addWidget(message);
    auto* buttons = new QDialogButtonBox;
    auto* discard = buttons->addButton(aitrain_app::workbenchText(QStringLiteral("丢弃草稿")), QDialogButtonBox::AcceptRole);
    discard->setObjectName(QStringLiteral("TrainingConfirmDiscard"));
    auto* cancel = buttons->addButton(aitrain_app::workbenchText(QStringLiteral("取消")), QDialogButtonBox::RejectRole);
    cancel->setDefault(true); discard->setAutoDefault(false);
    layout->addWidget(buttons); confirmation.resize(440, 140);
    connect(buttons, &QDialogButtonBox::accepted, &confirmation, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &confirmation, &QDialog::reject);
    if (confirmation.exec() != QDialog::Accepted) return;
    restoringDraft_ = true; draftTimer_->stop();
    aitrain_app::ApplicationSettingsService().removeTrainingDraft(draftProjectId_);
    advancedBackup_.clear(); binding_ = {}; resetDraftControls(); draftDirty_ = false; restoringDraft_ = false;
    page_->findChild<QLabel*>(QStringLiteral("TrainingDraftStatus"))->setText(aitrain_app::workbenchText(QStringLiteral("草稿已丢弃。")));
}
