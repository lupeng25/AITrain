#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"
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
    if (!taskArtifactTable_ || taskArtifactTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = taskArtifactTable_->selectedItems().first()->row();
    auto* item = taskArtifactTable_->item(row, 1);
    return item ? item->data(Qt::UserRole).toString() : QString();
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
    if (!taskQueueTable_ || !repository_.isOpen()) {
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

    if (selectedTaskSummaryLabel_) {
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
        selectedTaskSummaryLabel_->setText(summary);
    }

    if (taskArtifactTable_) {
        taskArtifactTable_->setRowCount(0);
        if (artifacts.isEmpty()) {
            taskArtifactTable_->insertRow(0);
            taskArtifactTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无产物")));
            for (int column = 1; column < taskArtifactTable_->columnCount(); ++column) {
                taskArtifactTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ArtifactRecord& artifact : artifacts) {
                const int artifactRow = taskArtifactTable_->rowCount();
                taskArtifactTable_->insertRow(artifactRow);
                taskArtifactTable_->setItem(artifactRow, 0, new QTableWidgetItem(artifact.kind));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(artifact.path));
                pathItem->setData(Qt::UserRole, artifact.path);
                taskArtifactTable_->setItem(artifactRow, 1, pathItem);
                taskArtifactTable_->setItem(artifactRow, 2, new QTableWidgetItem(artifact.message));
                taskArtifactTable_->setItem(artifactRow, 3, new QTableWidgetItem(artifact.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (taskMetricTable_) {
        taskMetricTable_->setRowCount(0);
        if (metrics.isEmpty()) {
            taskMetricTable_->insertRow(0);
            taskMetricTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无指标")));
            for (int column = 1; column < taskMetricTable_->columnCount(); ++column) {
                taskMetricTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::MetricPoint& metric : metrics) {
                const int metricRow = taskMetricTable_->rowCount();
                taskMetricTable_->insertRow(metricRow);
                taskMetricTable_->setItem(metricRow, 0, new QTableWidgetItem(metric.name));
                taskMetricTable_->setItem(metricRow, 1, new QTableWidgetItem(QString::number(metric.value, 'f', 6)));
                taskMetricTable_->setItem(metricRow, 2, new QTableWidgetItem(QString::number(metric.step)));
                taskMetricTable_->setItem(metricRow, 3, new QTableWidgetItem(QString::number(metric.epoch)));
            }
        }
    }

    if (taskExportTable_) {
        taskExportTable_->setRowCount(0);
        if (exports.isEmpty()) {
            taskExportTable_->insertRow(0);
            taskExportTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无导出")));
            for (int column = 1; column < taskExportTable_->columnCount(); ++column) {
                taskExportTable_->setItem(0, column, new QTableWidgetItem(QString()));
            }
        } else {
            for (const aitrain::ExportRecord& exportRecord : exports) {
                const int exportRow = taskExportTable_->rowCount();
                taskExportTable_->insertRow(exportRow);
                taskExportTable_->setItem(exportRow, 0, new QTableWidgetItem(exportRecord.format));
                auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(exportRecord.path));
                pathItem->setData(Qt::UserRole, exportRecord.path);
                taskExportTable_->setItem(exportRow, 1, pathItem);
                taskExportTable_->setItem(exportRow, 2, new QTableWidgetItem(exportRecord.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            }
        }
    }

    if (!artifacts.isEmpty() && taskArtifactTable_) {
        int preferredRow = 0;
        for (int artifactRow = 0; artifactRow < taskArtifactTable_->rowCount(); ++artifactRow) {
            auto* kindItem = taskArtifactTable_->item(artifactRow, 0);
            if (kindItem && kindItem->text() == QStringLiteral("evaluation_report")) {
                preferredRow = artifactRow;
                break;
            }
        }
        taskArtifactTable_->selectRow(preferredRow);
    }
}

void MainWindow::previewArtifactPath(const QString& path)
{
    if (!artifactPreviewText_ || !artifactImagePreviewLabel_ || !artifactPreviewStack_) {
        return;
    }

    artifactPreviewStack_->setCurrentIndex(0);
    if (artifactEvaluationReportView_) {
        artifactEvaluationReportView_->clear();
    }
    artifactImagePreviewLabel_->clear();
    artifactImagePreviewLabel_->setVisible(false);
    artifactImagePreviewLabel_->setText(uiText("暂无产物预览"));
    artifactPreviewText_->setVisible(true);
    artifactPreviewText_->clear();
    if (path.isEmpty()) {
        artifactImagePreviewLabel_->setVisible(true);
        artifactPreviewText_->setVisible(false);
        return;
    }

    const QFileInfo info(path);
    if (!info.exists()) {
        artifactPreviewText_->setPlainText(uiText("产物不存在：%1").arg(QDir::toNativeSeparators(path)));
        return;
    }

    if (info.isDir()) {
        artifactPreviewText_->setPlainText(uiText("目录产物\n路径：%1\n修改时间：%2")
            .arg(QDir::toNativeSeparators(path), info.lastModified().toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
        return;
    }

    const QString suffix = info.suffix().toLower();
    if (QStringList{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("bmp")}.contains(suffix)) {
        QPixmap image(path);
        if (!image.isNull()) {
            artifactImagePreviewLabel_->setVisible(true);
            artifactImagePreviewLabel_->setPixmap(image.scaled(
                artifactImagePreviewLabel_->size().boundedTo(QSize(520, 360)),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation));
        }
        artifactPreviewText_->setPlainText(uiText("图片产物\n路径：%1\n尺寸：%2 x %3\n大小：%4 bytes")
            .arg(QDir::toNativeSeparators(path))
            .arg(image.width())
            .arg(image.height())
            .arg(info.size()));
        return;
    }

    if (suffix == QStringLiteral("onnx")) {
        artifactPreviewText_->setPlainText(uiText("ONNX 模型\n路径：%1\n模型族：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), aitrain::inferOnnxModelFamily(path))
            .arg(info.size()));
        return;
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan") || suffix == QStringLiteral("pdparams")
        || suffix == QStringLiteral("aitrain") || suffix == QStringLiteral("param") || suffix == QStringLiteral("bin")) {
        artifactPreviewText_->setPlainText(uiText("模型产物\n路径：%1\n类型：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), suffix)
            .arg(info.size()));
        return;
    }

    if (QStringList{QStringLiteral("json"), QStringLiteral("yaml"), QStringLiteral("yml"), QStringLiteral("txt"), QStringLiteral("csv"), QStringLiteral("log")}.contains(suffix)) {
        if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("evaluation_report.json") && artifactEvaluationReportView_) {
            artifactEvaluationReportView_->loadReport(path);
            artifactPreviewStack_->setCurrentIndex(1);
            return;
        }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            artifactPreviewText_->setPlainText(uiText("无法读取文本产物：%1").arg(QDir::toNativeSeparators(path)));
            return;
        }
        const qint64 maxBytes = 256 * 1024;
        const QByteArray data = file.read(maxBytes);
        QString text = suffix == QStringLiteral("json") ? formatJsonTextForPreview(data) : QString::fromUtf8(data);
        if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("evaluation_report.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
                QStringList lines;
                lines << uiText("评估报告摘要");
                lines << uiText("任务类型：%1").arg(report.value(QStringLiteral("taskType")).toString());
                lines << uiText("真实评估：%1").arg(report.value(QStringLiteral("scaffold")).toBool() ? uiText("否，scaffold") : uiText("是"));
                lines << uiText("precision=%1 recall=%2 mAP50=%3")
                    .arg(metrics.value(QStringLiteral("precision")).toDouble(), 0, 'f', 4)
                    .arg(metrics.value(QStringLiteral("recall")).toDouble(), 0, 'f', 4)
                    .arg(metrics.value(QStringLiteral("mAP50")).toDouble(), 0, 'f', 4);
                lines << uiText("错误样本：%1；低置信样本：%2")
                    .arg(report.value(QStringLiteral("errorSamples")).toArray().size())
                    .arg(report.value(QStringLiteral("lowConfidenceSamples")).toArray().size());
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        } else if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("dataset_quality_report.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonObject severity = report.value(QStringLiteral("severityCounts")).toObject();
                const QJsonObject summary = report.value(QStringLiteral("summary")).toObject();
                QStringList lines;
                lines << uiText("数据质量报告摘要");
                lines << uiText("格式：%1；真实分析：%2")
                    .arg(report.value(QStringLiteral("format")).toString())
                    .arg(report.value(QStringLiteral("scaffold")).toBool() ? uiText("否，scaffold") : uiText("是"));
                lines << uiText("error=%1 warning=%2 info=%3 问题样本=%4 重复图片=%5")
                    .arg(severity.value(QStringLiteral("error")).toInt())
                    .arg(severity.value(QStringLiteral("warning")).toInt())
                    .arg(severity.value(QStringLiteral("info")).toInt())
                    .arg(summary.value(QStringLiteral("problemSampleCount")).toInt())
                    .arg(summary.value(QStringLiteral("duplicateImageCount")).toInt());
                lines << uiText("修复清单：%1").arg(QDir::toNativeSeparators(report.value(QStringLiteral("xAnyLabelingFixListPath")).toString()));
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        } else if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("problem_samples.json")) {
            QJsonParseError parseError;
            const QJsonObject report = QJsonDocument::fromJson(data, &parseError).object();
            if (parseError.error == QJsonParseError::NoError && !report.isEmpty()) {
                const QJsonArray samples = report.value(QStringLiteral("samples")).toArray();
                QStringList lines;
                lines << uiText("问题样本摘要");
                lines << uiText("问题样本数：%1").arg(samples.size());
                const int previewCount = qMin(5, samples.size());
                for (int index = 0; index < previewCount; ++index) {
                    const QJsonObject sample = samples.at(index).toObject();
                    lines << QStringLiteral("%1. %2 %3 %4")
                        .arg(index + 1)
                        .arg(sample.value(QStringLiteral("severity")).toString())
                        .arg(sample.value(QStringLiteral("code")).toString())
                        .arg(QDir::toNativeSeparators(sample.value(QStringLiteral("imagePath")).toString()));
                }
                lines << QString();
                text = lines.join(QLatin1Char('\n')) + text;
            }
        }
        if (info.size() > maxBytes) {
            text.append(uiText("\n\n[文件超过 256KB，仅显示前部内容]\n路径：%1").arg(QDir::toNativeSeparators(path)));
        }
        artifactPreviewText_->setPlainText(text);
        return;
    }

    artifactPreviewText_->setPlainText(uiText("不支持内联预览的产物\n路径：%1\n大小：%2 bytes")
        .arg(QDir::toNativeSeparators(path))
        .arg(info.size()));
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

void MainWindow::updateArtifactPreviewFromSelection()
{
    previewArtifactPath(selectedArtifactPath());
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
