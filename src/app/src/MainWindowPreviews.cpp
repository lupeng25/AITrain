#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"
#include "TaskArtifactPanel.h"
#include "aitrain/core/DetectionTrainer.h"
#include "aitrain/core/PluginInterfaces.h"

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
#include <QInputDialog>
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

QString MainWindow::selectedTaskId() const
{
    if (!taskQueueTable_ || taskQueueTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    auto* item = taskQueueTable_->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

QString MainWindow::selectedArtifactPath() const
{
    return taskArtifactPanel_ ? taskArtifactPanel_->selectedArtifactPath() : QString();
}

QString MainWindow::selectedEvaluationReportPath() const
{
    if (!evaluationReportTable_ || evaluationReportTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = evaluationReportTable_->selectedItems().first()->row();
    auto* item = evaluationReportTable_->item(row, 3);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void MainWindow::updateSelectedTaskDetails()
{
    if (!taskQueueTable_ || !repository_.isOpen() || !taskArtifactPanel_) {
        return;
    }
    if (taskQueueTable_->selectedItems().isEmpty()) {
        clearSelectedTaskDetails();
        return;
    }
    const int row = taskQueueTable_->selectedItems().first()->row();
    const QString taskId = taskQueueTable_->item(row, 0)
        ? taskQueueTable_->item(row, 0)->data(Qt::UserRole).toString()
        : QString();
    if (taskId.isEmpty()) {
        clearSelectedTaskDetails();
        return;
    }
    const QString taskKind = taskQueueTable_->item(row, 1) ? taskQueueTable_->item(row, 1)->text() : QString();
    const QString taskBackend = taskQueueTable_->item(row, 2) ? taskQueueTable_->item(row, 2)->text() : QString();
    const QString taskType = taskQueueTable_->item(row, 3) ? taskQueueTable_->item(row, 3)->text() : QString();
    const QString taskState = taskQueueTable_->item(row, 4) ? taskQueueTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
    const QString taskMessage = taskQueueTable_->item(row, 6) ? taskQueueTable_->item(row, 6)->text() : QString();

    QString error;
    const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(taskId, &error);
    const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
    const QVector<aitrain::ExportRecord> exports = repository_.exportsForTask(taskId, &error);

    QString summary = uiText("任务 %1：%2 / %3 / %4，%5 个产物，%6 个指标点，%7 条导出记录")
        .arg(taskId.left(8))
        .arg(taskKind.isEmpty() ? uiText("任务") : taskKind)
        .arg(taskType.isEmpty() ? uiText("未记录类型") : taskType)
        .arg(taskBackend.isEmpty() ? uiText("未记录后端") : taskBackend)
        .arg(artifacts.size())
        .arg(metrics.size())
        .arg(exports.size());
    if (taskState == QStringLiteral("failed")) {
        summary.append(uiText("\n失败摘要：%1\n建议：优先查看 report/log 产物；若提示缺环境或缺 Python 包，进入“环境”页自检；若提示数据错误，回到“数据集”页重新校验。")
            .arg(taskMessage.isEmpty() ? uiText("Worker 未返回详细消息。") : taskMessage));
    } else if (!taskMessage.isEmpty()) {
        summary.append(uiText("\n最新消息：%1").arg(taskMessage));
    }

    taskArtifactPanel_->setTaskSummary(summary);
    taskArtifactPanel_->setArtifacts(artifacts);
    taskArtifactPanel_->setMetrics(metrics);
    taskArtifactPanel_->setExports(exports);
}

void MainWindow::updateSelectedEvaluationReportDetails()
{
    if (!evaluationReportView_) {
        return;
    }
    const QString reportPath = selectedEvaluationReportPath();
    if (reportPath.isEmpty()) {
        evaluationReportView_->clear();
        return;
    }
    evaluationReportView_->loadReport(reportPath);
}

void MainWindow::openSelectedArtifactDirectory()
{
    const QString path = selectedArtifactPath();
    if (path.isEmpty()) {
        return;
    }
    const QFileInfo info(path);
    const QString directory = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    QDesktopServices::openUrl(QUrl::fromLocalFile(directory));
}

void MainWindow::copySelectedArtifactPath()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty()) {
        QApplication::clipboard()->setText(QDir::toNativeSeparators(path));
        statusBar()->showMessage(uiText("产物路径已复制"), 3000);
    }
}

void MainWindow::useSelectedArtifactForInference()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty() && inferenceCheckpointEdit_) {
        inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        showPage(InferencePage, tr("推理验证"));
    }
}

void MainWindow::useSelectedArtifactForExport()
{
    const QString path = selectedArtifactPath();
    if (!path.isEmpty() && conversionCheckpointEdit_) {
        conversionCheckpointEdit_->setText(QDir::toNativeSeparators(path));
        showPage(ConversionPage, tr("模型导出"));
    }
}

void MainWindow::registerSelectedArtifactAsModelVersion()
{
    if (!repository_.isOpen()) {
        QMessageBox::information(this, uiText("模型注册"), uiText("请先打开项目。"));
        return;
    }
    const QString path = selectedArtifactPath();
    if (path.isEmpty()) {
        QMessageBox::information(this, uiText("模型注册"), uiText("请先选择一个 checkpoint、ONNX 或 engine 产物。"));
        return;
    }

    const QFileInfo info(path);
    const QString suffix = info.suffix().toLower();
    const QString defaultName = currentProjectName_.isEmpty() ? QStringLiteral("model") : currentProjectName_;
    bool ok = false;
    const QString modelName = QInputDialog::getText(this, uiText("模型注册"), uiText("模型名称"), QLineEdit::Normal, defaultName, &ok).trimmed();
    if (!ok || modelName.isEmpty()) {
        return;
    }

    const int existingCount = repository_.recentModelVersions(500).size();
    const QString defaultVersion = QStringLiteral("v%1").arg(existingCount + 1);
    const QString version = QInputDialog::getText(this, uiText("模型注册"), uiText("版本号"), QLineEdit::Normal, defaultVersion, &ok).trimmed();
    if (!ok || version.isEmpty()) {
        return;
    }

    QJsonObject metricSummary;
    const QString taskId = selectedTaskId();
    if (!taskId.isEmpty()) {
        QString error;
        const QVector<aitrain::MetricPoint> metrics = repository_.metricsForTask(taskId, &error);
        for (const aitrain::MetricPoint& metric : metrics) {
            metricSummary.insert(metric.name, metric.value);
        }
    }

    aitrain::ModelVersionRecord record;
    record.modelName = modelName;
    record.version = version;
    record.sourceTaskId = taskId;
    if (!taskId.isEmpty()) {
        QString runError;
        const aitrain::ExperimentRunRecord run = repository_.experimentRunForTask(taskId, &runError);
        if (run.id > 0) {
            record.experimentRunId = run.id;
            record.datasetSnapshotId = run.datasetSnapshotId;
        }
    }
    record.status = suffix == QStringLiteral("onnx") ? QStringLiteral("exported") : QStringLiteral("draft");
    record.notes = uiText("从任务产物手动注册。");
    record.metricsJson = QString::fromUtf8(QJsonDocument(metricSummary).toJson(QJsonDocument::Compact));
    if (suffix == QStringLiteral("onnx")) {
        record.onnxPath = path;
    } else if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan")) {
        record.tensorRtEnginePath = path;
    } else {
        record.checkpointPath = path;
    }
    record.createdAt = QDateTime::currentDateTimeUtc();
    record.updatedAt = record.createdAt;

    QString error;
    const int id = repository_.upsertModelVersion(record, &error);
    if (id <= 0) {
        QMessageBox::critical(this, uiText("模型注册"), error);
        return;
    }
    updateModelRegistry();
    updateDashboardSummary();
    statusBar()->showMessage(uiText("已注册模型版本：%1:%2").arg(modelName, version), 4000);
    showPage(ModelRegistryPage, uiText("模型库"));
}
