#include "WorkbenchTranslation.h"
#include "TrainingPageController.h"
#include "TrainingPage.h"
#include "TaskArtifactPresenter.h"
#include "ProjectObjectSelectors.h"
#include "MainWindowSupport.h"
#include "aitrain/product/ProductCapabilityContract.h"
#include <QCheckBox>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDialog>
#include <QDialogButtonBox>
#include <QPlainTextEdit>
#include <QComboBox>
#include <QDateTime>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QSet>
#include <QSignalBlocker>
#include <cmath>
using namespace aitrain_app;

void TrainingPageController::setQueryService(const aitrain::ProjectQueryService* query)
{
    queryService_ = query;
    if (history_) delete history_;
    history_ = new TaskArtifactPresenter(query, this);
}

void TrainingPageController::refreshHistory()
{
    if (!page_ || !history_) return;
    if (projectOpen_) {
        QStringList trainingKinds;
        for (const auto& backend : aitrain::ProductCapabilityContract::instance().trainingBackends()) trainingKinds.append(backend.taskType);
        trainingKinds.removeDuplicates(); history_->setCatalogFilter({historySearch_, trainingKinds, {}});
        historyConfigurations_.remove(activeTaskId_);
        const int count = history_->taskCount();
        history_->refresh();
        while (history_->hasMoreTasks() && history_->taskCount() < count) if (!history_->loadMore()) break;
    }
    renderHistory();
}

void TrainingPageController::renderHistory()
{
    if (!page_ || !history_) return;
    auto* table = page_->historyTable;
    const QString previous = table->currentRow() >= 0
        ? table->item(table->currentRow(), 0)->data(Qt::UserRole).toString() : QString();
    const QSignalBlocker blocker(table);
    table->setRowCount(0);
    QSet<QString> trainingTypes;
    for (const auto& backend : aitrain::ProductCapabilityContract::instance().trainingBackends())
        trainingTypes.insert(backend.taskType);
    if (projectOpen_) for (const auto& item : history_->taskRows()) {
        if (!trainingTypes.contains(item.taskType)) continue;
        const int row = table->rowCount(); table->insertRow(row);
        const QString date = item.updatedAt;
        const QJsonObject configuration = historyParameters(item.taskId);
        TrainingDatasetBinding dataset; QString datasetVersion;
        const bool bound = historyBinding(configuration, &dataset, &datasetVersion);
        const QString model = configuration.value(QStringLiteral("modelPreset")).toString(configuration.value(QStringLiteral("model")).toString());
        const QStringList values = {QStringLiteral("%1 · %2").arg(taskTypeLabel(item.taskType), date), bound ? datasetVersion : aitrain_app::workbenchText(QStringLiteral("尚无已保存绑定")), model.isEmpty() ? taskTypeLabel(item.taskType) : model, item.stateLabel, date};
        for (int c = 0; c < values.size(); ++c) table->setItem(row, c, new QTableWidgetItem(values[c]));
        table->item(row, 0)->setData(Qt::UserRole, item.taskId);
        if (item.taskId == previous) table->selectRow(row);
    }
    page_->moreHistory->setVisible(projectOpen_ && history_->hasMoreTasks());
    page_->historyStatus->setText(!projectOpen_ ? aitrain_app::workbenchText(QStringLiteral("请先打开项目。"))
        : !history_->lastError().isEmpty() ? aitrain_app::workbenchText(QStringLiteral("读取失败：%1")).arg(history_->lastError())
        : table->rowCount() > 0 ? aitrain_app::workbenchText(QStringLiteral("已显示 %1 条训练记录。按更新时间排列，可继续载入更早记录。")).arg(table->rowCount())
        : history_->hasMoreTasks() ? aitrain_app::workbenchText(QStringLiteral("当前批次没有训练记录，可继续查找更早记录。")) : aitrain_app::workbenchText(QStringLiteral("还没有训练记录。选择数据版本后开始训练。")));
}

void TrainingPageController::openHistory(const QString& taskId)
{
    if (!history_ || !history_->selectTask(taskId)) return;
    selectedTaskId_ = taskId;
    page_->resetRuntimeProjection();
    const auto& details = history_->details();
    page_->setPhase(details.summary);
    page_->appendLog(details.summary);
    for (const auto& metric : details.metrics) page_->addMetric(metric.name, metric.value);
    page_->cancelTaskButton->setEnabled(false);
    const auto configuration = historyParameters(taskId); TrainingDatasetBinding dataset; QString version;
    historyBinding(configuration, &dataset, &version);
    page_->setLiveValue(QStringLiteral("TrainingMonitorContext"), configuration.value(QStringLiteral("modelPreset")).toString() + QStringLiteral(" · ") + version);
    refreshResultModel();
    page_->setMode(TrainingWorkspacePage::Monitor);
    // 历史记录只投影已持久化指标；原始日志和全部分页产物在任务详情中读取。
}

void TrainingPageController::chooseDataset()
{
    DatasetSelection selected;
    if (!selectProjectDataset(page_, queryService_, &selected, true)) return;
    TrainingDatasetBinding binding;
    binding.datasetId = selected.snapshot.datasetId.toString();
    binding.datasetVersionId = selected.snapshot.datasetVersionId.toString();
    binding.snapshotId = selected.snapshot.snapshotId.toString();
    binding.snapshotArtifactId = selected.snapshot.artifactId.toString();
    binding.datasetFormat = selected.snapshot.datasetFormat;
    binding.displayName = selected.displayName;
    binding.deploymentSampleRelativePath = selected.sampleRelativePath;
    setDatasetBinding(binding);
}

void TrainingPageController::editAdvanced()
{
    advancedBackup_.clear();
    for (auto* widget : page_->findChildren<QWidget*>()) {
        const QString name = widget->objectName();
        if (!name.contains(QStringLiteral("TrainArg_")) && !name.startsWith(QStringLiteral("YoloTrainExportArg_"))) continue;
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) advancedBackup_[name] = edit->text();
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) advancedBackup_[name] = combo->currentIndex();
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) advancedBackup_[name] = check->isChecked();
    }
    page_->setMode(TrainingWorkspacePage::Advanced, TrainingWorkspacePage::Configuration);
    page_->backButton->hide();
}

void TrainingPageController::cancelAdvanced()
{
    for (auto it = advancedBackup_.cbegin(); it != advancedBackup_.cend(); ++it) {
        auto* widget = page_->findChild<QWidget*>(it.key());
        if (!widget) continue;
        const QSignalBlocker blocker(widget);
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) edit->setText(it.value().toString());
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) combo->setCurrentIndex(it.value().toInt());
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) check->setChecked(it.value().toBool());
    }
    advancedBackup_.clear();
    page_->setMode(TrainingWorkspacePage::Configuration);
}

bool TrainingPageController::validateParameters()
{
    QString error;
    for (const QString& name : {QStringLiteral("TrainingEpochs"), QStringLiteral("TrainingBatchSize"), QStringLiteral("TrainingImageSize")}) {
        const auto* field = page_->findChild<QLineEdit*>(name);
        if (!field || !field->hasAcceptableInput()) { error = aitrain_app::workbenchText(QStringLiteral("轮数、批次和输入尺寸必须填写正整数。")); break; }
    }
    const QString backend = page_->formData().backendId;
    const QString prefix = backend.startsWith(QStringLiteral("ultralytics_")) ? QStringLiteral("YoloTrainArg_")
        : backend.startsWith(QStringLiteral("smp_")) ? QStringLiteral("SmpTrainArg_")
        : backend.startsWith(QStringLiteral("anomalib_")) ? QStringLiteral("AnomalyTrainArg_") : QString();
    const QSet<QString> strings = {QStringLiteral("device"), QStringLiteral("backbone"), QStringLiteral("layers"), QStringLiteral("imagenetDir")};
    const QSet<QString> integers = {QStringLiteral("seed"), QStringLiteral("workers"), QStringLiteral("patience"), QStringLiteral("save_period"), QStringLiteral("nbs"), QStringLiteral("max_det"), QStringLiteral("close_mosaic"), QStringLiteral("mask_ratio"), QStringLiteral("ignoreIndex"), QStringLiteral("numNeighbors")};
    if (error.isEmpty() && !prefix.isEmpty()) for (const auto* edit : page_->findChildren<QLineEdit*>()) {
        if (!edit->objectName().startsWith(prefix)) continue;
        const QString key = edit->objectName().mid(prefix.size()), value = edit->text().trimmed();
        if (value.isEmpty() || strings.contains(key)) continue;
        bool valid = true;
        if (key == QStringLiteral("classes") || key == QStringLiteral("freeze")) {
            for (const QString& part : value.split(QLatin1Char(','))) { bool ok; const int n = part.trimmed().toInt(&ok); valid = valid && ok && n >= 0; }
        } else if (integers.contains(key)) { value.toInt(&valid); }
        else { const double number = value.toDouble(&valid); valid = valid && std::isfinite(number); }
        if (!valid) { error = aitrain_app::workbenchText(QStringLiteral("参数 %1 的数值格式不正确。")).arg(key); break; }
    }
    if (auto* label = page_->findChild<QLabel*>(QStringLiteral("TrainingFormError"))) label->setText(error);
    if (!error.isEmpty() && page_->views->currentIndex() == TrainingWorkspacePage::Advanced) QMessageBox::warning(page_, aitrain_app::workbenchText(QStringLiteral("检查参数")), error);
    return error.isEmpty();
}


void TrainingPageController::captureParameterDefaults()
{
    for (auto* widget : page_->findChildren<QWidget*>()) {
        const QString name = widget->objectName();
        if (!name.contains(QStringLiteral("TrainArg_")) && !name.startsWith(QStringLiteral("YoloTrainExportArg_"))) continue;
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) defaultParameterControls_[name] = edit->text();
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) defaultParameterControls_[name] = combo->currentIndex();
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) defaultParameterControls_[name] = check->isChecked();
    }
}

QJsonObject TrainingPageController::historyParameters(const QString& taskId)
{
    if (historyConfigurations_.contains(taskId)) return historyConfigurations_.value(taskId);
    aitrain::TaskId id; QString error;
    if (!queryService_ || !aitrain::TaskId::parse(taskId, &id, &error)) return {};
    aitrain::TaskReadModel details;
    if (!queryService_->taskDetails(id, &details, &error)) return {};
    for (const auto& workflow : details.workflows) for (const auto& step : workflow.steps) {
        if (step.kind == QStringLiteral("Train") && step.parameterSummary.contains(QStringLiteral("trainingBackend"))) {
            historyConfigurations_.insert(taskId, step.parameterSummary);
            return step.parameterSummary;
        }
    }
    return {};
}

bool TrainingPageController::historyBinding(const QJsonObject& configuration,
    TrainingDatasetBinding* result, QString* versionLabel)
{
    const QString snapshotText = configuration.value(QStringLiteral("datasetSnapshotId")).toString();
    if (snapshotText.isEmpty() || !queryService_) return false;
    aitrain::DatasetId datasetId; QString error;
    if (!aitrain::DatasetId::parse(configuration.value(QStringLiteral("datasetId")).toString(), &datasetId, &error)) return false;
    QString name, catalogCursor;
    do {
        const auto catalog = queryService_->datasetCatalog({100, catalogCursor}, &error);
        if (!error.isEmpty()) return false;
        for (const auto& item : catalog.items) if (item.datasetId == datasetId) { name = item.displayName; break; }
        catalogCursor = catalog.hasMore ? catalog.nextCursor : QString();
    } while (name.isEmpty() && !catalogCursor.isEmpty());
    QString cursor;
    do {
        const auto snapshots = queryService_->datasetSnapshots(datasetId, {100, cursor}, &error);
        if (!error.isEmpty()) return false;
        for (const auto& snapshot : snapshots.items) {
            if (snapshot.snapshotId.toString() != snapshotText) continue;
            if (snapshot.datasetVersionId.toString() != configuration.value(QStringLiteral("datasetVersionId")).toString()
                || snapshot.artifactId.toString() != configuration.value(QStringLiteral("datasetSnapshotArtifactId")).toString()) return false;
            result->datasetId = snapshot.datasetId.toString(); result->datasetVersionId = snapshot.datasetVersionId.toString();
            result->snapshotId = snapshotText; result->snapshotArtifactId = snapshot.artifactId.toString(); result->datasetFormat = snapshot.datasetFormat;
            *versionLabel = QStringLiteral("%1 · %2").arg(name.isEmpty() ? datasetFormatLabel(snapshot.datasetFormat) : name, snapshot.createdAt.toLocalTime().toString(QStringLiteral("MM-dd HH:mm:ss")));
            result->displayName = *versionLabel;
            return true;
        }
        cursor = snapshots.hasMore ? snapshots.nextCursor : QString();
    } while (!cursor.isEmpty());
    return false;
}

void TrainingPageController::showHistoricalConfiguration()
{
    const auto configuration = historyParameters(selectedTaskId_);
    if (configuration.isEmpty()) { page_->setPhase(aitrain_app::workbenchText(QStringLiteral("该任务尚无已保存的训练配置，可在任务详情查看失败原因。"))); return; }
    QDialog dialog(page_); dialog.setWindowTitle(aitrain_app::workbenchText(QStringLiteral("训练配置（只读）"))); dialog.resize(850, 600);
    auto* layout = new QVBoxLayout(&dialog); auto* text = new QPlainTextEdit; text->setReadOnly(true);
    text->setPlainText(QString::fromUtf8(QJsonDocument(configuration).toJson(QJsonDocument::Indented))); layout->addWidget(text, 1);
    auto* close = new QDialogButtonBox(QDialogButtonBox::Close); layout->addWidget(close); connect(close, &QDialogButtonBox::rejected, &dialog, &QDialog::reject); dialog.exec();
}

void TrainingPageController::copyHistoricalConfiguration()
{
    const auto configuration = historyParameters(selectedTaskId_);
    TrainingDatasetBinding selected; QString versionLabel;
    if (configuration.isEmpty() || !historyBinding(configuration, &selected, &versionLabel)) {
        page_->setPhase(aitrain_app::workbenchText(QStringLiteral("无法恢复该训练的完整数据版本，请先确认来源快照仍在当前项目中。"))); return;
    }
    setDatasetBinding(selected);
    auto* backend = page_->findChild<QComboBox*>(QStringLiteral("TrainingBackend"));
    const int backendIndex = backend->findData(configuration.value(QStringLiteral("trainingBackend")).toString());
    if (backendIndex < 0) { page_->setPhase(aitrain_app::workbenchText(QStringLiteral("该训练后端不在当前产品合同中。"))); return; }
    backend->setCurrentIndex(backendIndex);
    const QString backendId = backend->currentData().toString();
    const auto textValue = [](const QJsonValue& value) {
        if (value.isArray()) { QStringList parts; for (const auto& part : value.toArray()) parts.append(part.toVariant().toString()); return parts.join(QLatin1Char(',')); }
        return value.toVariant().toString();
    };
    for (auto* widget : page_->findChildren<QWidget*>()) {
        const QString name = widget->objectName();
        if (!defaultParameterControls_.contains(name)) continue;
        QString prefix; QJsonObject arguments;
        if (name.startsWith(QStringLiteral("YoloTrainExportArg_"))) {
            prefix = QStringLiteral("YoloTrainExportArg_"); if (backendId.startsWith(QStringLiteral("ultralytics_"))) arguments = configuration.value(QStringLiteral("ultralyticsExportArgs")).toObject();
        } else if (name.startsWith(QStringLiteral("YoloTrainArg_"))) {
            prefix = QStringLiteral("YoloTrainArg_"); if (backendId.startsWith(QStringLiteral("ultralytics_"))) arguments = configuration.value(QStringLiteral("ultralyticsTrainArgs")).toObject();
        } else if (name.startsWith(QStringLiteral("SmpTrainArg_"))) {
            prefix = QStringLiteral("SmpTrainArg_"); if (backendId.startsWith(QStringLiteral("smp_"))) arguments = configuration;
        } else { prefix = QStringLiteral("AnomalyTrainArg_"); if (backendId.startsWith(QStringLiteral("anomalib_"))) arguments = configuration; }
        const QString key = name.mid(prefix.size()); const bool present = arguments.contains(key); const QJsonValue value = arguments.value(key);
        const QVariant baseline = defaultParameterControls_.value(name); const QSignalBlocker blocker(widget);
        if (auto* edit = qobject_cast<QLineEdit*>(widget)) edit->setText(present ? textValue(value) : baseline.toString());
        else if (auto* check = qobject_cast<QCheckBox*>(widget)) check->setChecked(present ? value.toBool() : baseline.toBool());
        else if (auto* combo = qobject_cast<QComboBox*>(widget)) {
            int index = present ? combo->findData(textValue(value)) : baseline.toInt();
            if (present && index < 0) index = combo->findText(textValue(value)); combo->setCurrentIndex(qMax(0, index));
        }
    }
    const QVector<QPair<QString, QString>> basic{{QStringLiteral("TrainingEpochs"), QStringLiteral("epochs")}, {QStringLiteral("TrainingBatchSize"), QStringLiteral("batchSize")}, {QStringLiteral("TrainingImageSize"), QStringLiteral("imageSize")}};
    for (const auto& field : basic) if (configuration.contains(field.second)) page_->findChild<QLineEdit*>(field.first)->setText(textValue(configuration.value(field.second)));
    auto* model = page_->findChild<QComboBox*>(QStringLiteral("TrainingModelPreset"));
    const QString modelValue = configuration.value(QStringLiteral("modelPreset")).toString(configuration.value(QStringLiteral("model")).toString());
    if (!modelValue.isEmpty()) model->setCurrentText(modelValue);
    if (backendId == QStringLiteral("anomalib_efficientad")) page_->findChild<QLineEdit*>(QStringLiteral("TrainingBatchSize"))->setText(QStringLiteral("1"));
    // 样本不从其他快照借用。旧配置未保存交付样本时要求用户重新选择。
    binding_.deploymentSampleRelativePath.clear();
    scheduleDraftSave();
    page_->setMode(TrainingWorkspacePage::Configuration);
    refreshSummary();
    page_->findChild<QLabel*>(QStringLiteral("TrainingFormError"))->setText(aitrain_app::workbenchText(QStringLiteral("已复制配置为新草稿。检查数据版本、样本和参数后再开始训练。")));
}

void TrainingPageController::refreshResultModel()
{
    resultModelId_.clear();
    if (!page_ || !queryService_ || !projectOpen_) return;
    QString cursor, error;
    do {
        const auto models = queryService_->modelPackages({100, cursor}, &error);
        if (!error.isEmpty()) break;
        for (const auto& model : models.items) if (model.sourceTaskId.toString() == selectedTaskId_) { resultModelId_ = model.modelPackageId.toString(); break; }
        cursor = models.hasMore ? models.nextCursor : QString();
    } while (resultModelId_.isEmpty() && !cursor.isEmpty());
    page_->modelsButton->setEnabled(!resultModelId_.isEmpty());
}
