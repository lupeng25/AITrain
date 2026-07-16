#include "MainWindow.h"
#include "ModelRegistryPresenter.h"
#include "TaskArtifactPresenter.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "TaskArtifactPanel.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

void MainWindow::updateRecentTasks()
{
    if (taskQueueTable_ && taskArtifactPresenter_ && workspace_.isOpen()) {
        taskArtifactPresenter_->refresh(200);
        updateTaskTable();
    }
    const QVector<TaskListItem> tasks = taskArtifactPresenter_
        ? taskArtifactPresenter_->taskRows() : QVector<TaskListItem>{};
    if (dashboardTaskValue_) {
        dashboardTaskValue_->setText(QString::number(tasks.size()));
    }
    if (recentTasksTable_) {
        recentTasksTable_->setRowCount(0);
        for (const TaskListItem& task : tasks) {
            const int row = recentTasksTable_->rowCount();
            recentTasksTable_->insertRow(row);
            recentTasksTable_->setItem(row, 0, new QTableWidgetItem(task.taskId.left(8)));
            recentTasksTable_->setItem(row, 1, new QTableWidgetItem(uiText(" 任务")));
            recentTasksTable_->setItem(row, 2, new QTableWidgetItem(task.capabilityId));
            recentTasksTable_->setItem(row, 3, new QTableWidgetItem(task.taskType));
            recentTasksTable_->setItem(row, 4, new QTableWidgetItem(task.stateLabel));
            recentTasksTable_->setItem(row, 5, new QTableWidgetItem(task.updatedAt));
            recentTasksTable_->setItem(row, 6, new QTableWidgetItem(task.message));
        }
    }
    updateDashboardSummary();
}
void MainWindow::updateTaskTable()
{
    if (!taskQueueTable_ || !taskArtifactPresenter_) return;
    taskQueueTable_->setRowCount(0);
    const QVector<TaskListItem>& rows = taskArtifactPresenter_->taskRows();
    if (rows.isEmpty()) {
        taskQueueTable_->insertRow(0);
        auto* empty = new QTableWidgetItem(uiText("暂无  任务记录"));
        empty->setData(Qt::UserRole + 1, QStringLiteral("empty"));
        taskQueueTable_->setItem(0, 0, empty);
        for (int column = 1; column < taskQueueTable_->columnCount(); ++column) {
            taskQueueTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        clearSelectedTaskDetails();
        return;
    }
    for (const TaskListItem& task : rows) {
        const int row = taskQueueTable_->rowCount();
        taskQueueTable_->insertRow(row);
        auto* id = new QTableWidgetItem(task.taskId.left(8));
        id->setData(Qt::UserRole, task.taskId);
        taskQueueTable_->setItem(row, 0, id);
        auto* category = new QTableWidgetItem(uiText(" 任务"));
        category->setData(Qt::UserRole, QStringLiteral(""));
        taskQueueTable_->setItem(row, 1, category);
        taskQueueTable_->setItem(row, 2, new QTableWidgetItem(task.capabilityId));
        auto* type = new QTableWidgetItem(task.taskType);
        type->setData(Qt::UserRole, task.taskType);
        taskQueueTable_->setItem(row, 3, type);
        auto* state = new QTableWidgetItem(task.stateLabel);
        state->setData(Qt::UserRole, task.state);
        taskQueueTable_->setItem(row, 4, state);
        taskQueueTable_->setItem(row, 5, new QTableWidgetItem(task.updatedAt));
        taskQueueTable_->setItem(row, 6, new QTableWidgetItem(task.message));
    }
    applyTaskFilters();
}

void MainWindow::updateDatasetList()
{
    QString error;
    const QVector<aitrain::DatasetCatalogItem> datasets = workspace_.isOpen()
        ? workspace_.datasets(50, &error) : QVector<aitrain::DatasetCatalogItem>{};
    if (datasetListTable_) {
        datasetListTable_->setRowCount(0);
    }
    if (datasetListTable_ && datasets.isEmpty()) {
        datasetListTable_->insertRow(0);
        datasetListTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无数据集记录")));
        for (int column = 1; column < datasetListTable_->columnCount(); ++column) {
            datasetListTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        updateDashboardSummary();
        return;
    }

    if (datasetListTable_) {
        for (const aitrain::DatasetCatalogItem& dataset : datasets) {
            const int row = datasetListTable_->rowCount();
            datasetListTable_->insertRow(row);
            auto* nameItem = new QTableWidgetItem(dataset.datasetId.toString().left(12));
            nameItem->setData(Qt::UserRole, dataset.datasetId.toString());
            datasetListTable_->setItem(row, 0, nameItem);
            auto* formatItem = new QTableWidgetItem(datasetFormatLabel(dataset.datasetFormat));
            formatItem->setData(Qt::UserRole, dataset.datasetFormat);
            datasetListTable_->setItem(row, 1, formatItem);
            auto* statusItem = new QTableWidgetItem(dataset.latestSnapshotId.isValid() ? uiText("已提交快照") : uiText("尚无快照"));
            statusItem->setData(Qt::UserRole, dataset.latestSnapshotId.toString());
            datasetListTable_->setItem(row, 2, statusItem);
            datasetListTable_->setItem(row, 3, new QTableWidgetItem(QString::number(dataset.latestFileCount)));
            auto* identityItem = new QTableWidgetItem(dataset.latestSnapshotId.toString());
            identityItem->setData(Qt::UserRole, dataset.latestArtifactId.toString());
            identityItem->setToolTip(uiText("Version %1\nArtifact %2\nRoot hash %3")
                .arg(dataset.latestVersionId.toString(), dataset.latestArtifactId.toString(), dataset.latestRootHash));
            datasetListTable_->setItem(row, 4, identityItem);
        }
    }
    updateDashboardSummary();
}

void MainWindow::updateAnnotationToolStatus()
{
    if (annotationToolStatusLabel_) {
        annotationToolStatusLabel_->setText(xAnyLabelingStatusText());
        annotationToolStatusLabel_->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    }
}

void MainWindow::setDatasetRepairLoopRows(const QString& summary, const QVector<QStringList>& rows)
{
    if (datasetRepairLoopLabel_) {
        datasetRepairLoopLabel_->setText(summary);
    }
    if (!datasetRepairLoopTable_) {
        return;
    }

    datasetRepairLoopTable_->setRowCount(0);
    if (rows.isEmpty()) {
        datasetRepairLoopTable_->insertRow(0);
        datasetRepairLoopTable_->setItem(0, 0, new QTableWidgetItem(uiText("等待")));
        datasetRepairLoopTable_->setItem(0, 1, new QTableWidgetItem(uiText("未开始")));
        datasetRepairLoopTable_->setItem(0, 2, new QTableWidgetItem(uiText("生成质量报告后进入修复闭环。")));
        return;
    }

    for (const QStringList& rowValues : rows) {
        const int row = datasetRepairLoopTable_->rowCount();
        datasetRepairLoopTable_->insertRow(row);
        for (int column = 0; column < datasetRepairLoopTable_->columnCount(); ++column) {
            const QString value = rowValues.value(column);
            auto* item = new QTableWidgetItem(value);
            item->setToolTip(value);
            datasetRepairLoopTable_->setItem(row, column, item);
        }
    }
}

void MainWindow::applyTaskFilters()
{
    if (!taskQueueTable_ || taskQueueTable_->columnCount() < 7) {
        return;
    }

    const QString kind = currentTaskKindFilter();
    const QString state = currentTaskStateFilter();
    const QString query = taskSearchEdit_ ? taskSearchEdit_->text().trimmed() : QString();

    for (int row = 0; row < taskQueueTable_->rowCount(); ++row) {
        const QString rowKind = taskQueueTable_->item(row, 0)
            ? taskQueueTable_->item(row, 0)->data(Qt::UserRole + 1).toString()
            : QString();
        if (rowKind == QStringLiteral("empty")) {
            taskQueueTable_->setRowHidden(row, false);
            continue;
        }

        bool visible = true;
        if (!kind.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 1)
                && taskQueueTable_->item(row, 1)->data(Qt::UserRole).toString() == kind;
        }
        if (!state.isEmpty()) {
            visible = visible && taskQueueTable_->item(row, 4)
                && taskQueueTable_->item(row, 4)->data(Qt::UserRole).toString() == state;
        }
        if (!query.isEmpty()) {
            bool matched = false;
            for (int column = 0; column < taskQueueTable_->columnCount(); ++column) {
                auto* item = taskQueueTable_->item(row, column);
                if (item && item->text().contains(query, Qt::CaseInsensitive)) {
                    matched = true;
                    break;
                }
            }
            visible = visible && matched;
        }
        taskQueueTable_->setRowHidden(row, !visible);
    }

    ensureVisibleTaskSelection();
}

void MainWindow::ensureVisibleTaskSelection()
{
    if (!taskQueueTable_) {
        return;
    }

    auto isSelectableRow = [this](int row) {
        if (row < 0 || row >= taskQueueTable_->rowCount() || taskQueueTable_->isRowHidden(row)) {
            return false;
        }
        auto* idItem = taskQueueTable_->item(row, 0);
        if (!idItem) {
            return false;
        }
        if (idItem->data(Qt::UserRole + 1).toString() == QStringLiteral("empty")) {
            return false;
        }
        return !idItem->data(Qt::UserRole).toString().isEmpty();
    };

    int selectedRow = -1;
    if (!taskQueueTable_->selectedItems().isEmpty()) {
        selectedRow = taskQueueTable_->selectedItems().first()->row();
    }
    if (isSelectableRow(selectedRow)) {
        return;
    }

    int preferredRow = -1;
    int fallbackRow = -1;
    for (int row = 0; row < taskQueueTable_->rowCount(); ++row) {
        if (!isSelectableRow(row)) {
            continue;
        }
        if (fallbackRow < 0) {
            fallbackRow = row;
        }
        auto* kindItem = taskQueueTable_->item(row, 1);
        if (kindItem && kindItem->data(Qt::UserRole).toString() == QStringLiteral("evaluate")) {
            preferredRow = row;
            break;
        }
    }

    const int rowToSelect = preferredRow >= 0 ? preferredRow : fallbackRow;
    if (rowToSelect >= 0) {
        const QSignalBlocker blocker(taskQueueTable_);
        taskQueueTable_->clearSelection();
        taskQueueTable_->selectRow(rowToSelect);
        taskQueueTable_->setCurrentCell(rowToSelect, 0);
        updateSelectedTaskDetails();
        return;
    }

    {
        const QSignalBlocker blocker(taskQueueTable_);
        taskQueueTable_->clearSelection();
    }
    clearSelectedTaskDetails();
}

void MainWindow::clearSelectedTaskDetails()
{
    if (taskArtifactPresenter_) taskArtifactPresenter_->clearSelection();
    else if (taskArtifactPanel_) taskArtifactPanel_->clear();
}

void MainWindow::updateModelRegistry()
{
    if (!workspace_.isOpen()) {
        modelRegistryPresenter_->clear();
        if (modelRegistrySummaryLabel_) modelRegistrySummaryLabel_->setText(uiText("请先打开  项目。"));
        return;
    }
    if (!modelRegistryPresenter_->refresh(200)) {
        if (modelRegistrySummaryLabel_) {
            modelRegistrySummaryLabel_->setText(uiText("读取  模型包目录失败：%1")
                .arg(modelRegistryPresenter_->lastError()));
        }
        return;
    }
    const QVector<ModelPackageListItem>& packages = modelRegistryPresenter_->modelPackages();
    if (modelRegistrySummaryLabel_) {
        modelRegistrySummaryLabel_->setText(uiText("已登记  模型包：%1。模型库只展示无路径 Manifest 与 lineage。")
            .arg(packages.size()));
    }
    if (ModelPackageTable_) {
        ModelPackageTable_->setRowCount(0);
        if (packages.isEmpty()) {
            ModelPackageTable_->insertRow(0);
            ModelPackageTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无已验证  模型包")));
            for (int column = 1; column < ModelPackageTable_->columnCount(); ++column)
                ModelPackageTable_->setItem(0, column, new QTableWidgetItem(QString()));
        } else {
            for (const ModelPackageListItem& package : packages) {
                const int row = ModelPackageTable_->rowCount();
                ModelPackageTable_->insertRow(row);
                auto* idItem = new QTableWidgetItem(package.modelPackageId);
                idItem->setData(Qt::UserRole, package.modelPackageId);
                ModelPackageTable_->setItem(row, 0, idItem);
                ModelPackageTable_->setItem(row, 1, new QTableWidgetItem(package.modelFamily));
                ModelPackageTable_->setItem(row, 2, new QTableWidgetItem(package.taskType));
                ModelPackageTable_->setItem(row, 3, new QTableWidgetItem(package.sourceBackend));
                ModelPackageTable_->setItem(row, 4, new QTableWidgetItem(package.decoder));
                ModelPackageTable_->setItem(row, 5, new QTableWidgetItem(package.createdAt));
            }
        }
    }
    const auto refreshCombo = [this, &packages](QComboBox* combo) {
        if (!combo) return;
        const QString selectedId = combo->currentData().toString();
        combo->clear();
        combo->addItem(uiText("请选择已验证  模型包"), QString());
        for (const ModelPackageListItem& package : packages) {
            combo->addItem(QStringLiteral("%1 · %2 · %3")
                .arg(package.modelFamily, package.taskType, package.modelPackageId.left(8)),
                package.modelPackageId);
        }
        const int index = combo->findData(selectedId);
        if (index >= 0) combo->setCurrentIndex(index);
    };
    refreshCombo(inferenceModelPackageCombo_);
    refreshCombo(deploymentModelPackageCombo_);
    if (modelVersionTable_) modelVersionTable_->setRowCount(0);
    if (pipelineRunTable_) pipelineRunTable_->setRowCount(0);
    if (evaluationReportView_) evaluationReportView_->clear();
    if (modelComparisonSummaryLabel_)
        modelComparisonSummaryLabel_->setText(uiText("旧裸路径模型对比已停用；后续仅按 ModelPackageId 与 committed Metric 对比。"));
    return;
}

void MainWindow::refreshModelRegistry()
{
    updateModelRegistry();
}

void MainWindow::handleDatasetConversionWorkflow(const QJsonObject& payload)
{
    const QString state = payload.value(QStringLiteral("state")).toString();
    const QString datasetId = payload.value(QStringLiteral("datasetId")).toString();
    const QString versionId = payload.value(QStringLiteral("datasetVersionId")).toString();
    const QString snapshotId = payload.value(QStringLiteral("snapshotId")).toString();
    const QString conversionArtifactId = payload.value(QStringLiteral("conversionArtifactId")).toString();
    const QString snapshotArtifactId = payload.value(QStringLiteral("snapshotArtifactId")).toString();
    const QString evidenceArtifactId = payload.value(QStringLiteral("evidenceArtifactId")).toString();
    if (datasetConversionProgressBar_ && state == QStringLiteral("succeeded"))
        datasetConversionProgressBar_->setValue(100);
    const QString summary = uiText("转换 %1：Dataset %2 | Version %3 | Snapshot %4 | Evidence %5")
        .arg(state, datasetId, versionId, snapshotId, evidenceArtifactId);
    if (datasetConversionStatusLabel_) datasetConversionStatusLabel_->setText(summary);
    if (datasetConversionResultLabel_) {
        datasetConversionResultLabel_->setText(
            uiText("Conversion Artifact %1 | Snapshot Artifact %2")
                .arg(conversionArtifactId, snapshotArtifactId));
    }
    appendDatasetConversionLog(summary);
    appendDatasetConversionLog(QString::fromUtf8(QJsonDocument(payload).toJson(QJsonDocument::Indented)));
    setDatasetConversionFormRunning(false);
    updateRecentTasks();
    updateSelectedTaskDetails();
    updateProjectSummary();
    updateDashboardSummary();
}
