#include "WorkbenchTranslation.h"
#include "DatasetPageController.h"

#include "ApplicationEventRouter.h"
#include "DatasetPage.h"
#include "DatasetCatalogPresenter.h"
#include <QSignalBlocker>
#include <QLineEdit>
#include "MainWindowSupport.h"
#include "ProjectObjectSelectors.h"

#include <QJsonDocument>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>

using namespace aitrain_app;

void DatasetPageController::applyTaskViewState(const TaskViewState& task)
{
    if (!page_ || task.taskId != activeTaskId_) return;
    const bool conversion = activeKind_ == QStringLiteral("dataset_conversion");
    if (conversion) {
        page_->datasetConversionProgressBar->setValue(task.progress);
        page_->datasetConversionLog->setPlainText(task.logs.join(QLatin1Char('\n')));
    }
    if (!task.terminal) return;
    const bool success = task.status == QStringLiteral("succeeded");
    const QString status = success ? aitrain_app::workbenchText(QStringLiteral("已完成"))
        : task.status == QStringLiteral("canceled") ? aitrain_app::workbenchText(QStringLiteral("已取消")) : aitrain_app::workbenchText(QStringLiteral("失败"));
    const QString message = task.terminalMessage.isEmpty() ? status
        : QStringLiteral("%1：%2").arg(status, task.terminalMessage);
    page_->operationStatusLabel->setText(message);
    if (conversion) {
        setConversionRunning(false);
        page_->datasetConversionStatusLabel->setText(message);
        page_->datasetConversionResultLabel->setText(success
            ? aitrain_app::workbenchText(QStringLiteral("转换结果已登记到数据集目录。")) : message);
    }
    if (activeKind_ == QStringLiteral("data_quality") && activeSnapshotId_ == state_.currentSnapshotId) {
        state_.latestQualityTaskId = task.taskId;
        if (success) loadQualityReport(task.taskId);
        else page_->validationSummaryLabel->setText(message);
    }
    for (const auto& artifact : task.artifacts) {
        if (artifact.kind == QStringLiteral("annotation_session")) state_.latestAnnotationSessionArtifactId = artifact.artifactId;
        if (artifact.kind == QStringLiteral("annotation_sync_report")) state_.latestAnnotationSyncReportArtifactId = artifact.artifactId;
    }
    if (activeKind_.startsWith(QStringLiteral("annotation_"))) {
        page_->datasetRepairLoopLabel->setText(message);
        emit repairLoopChanged(message, {{aitrain_app::workbenchText(QStringLiteral("标注会话")), status,
            success ? aitrain_app::workbenchText(QStringLiteral("可查看已提交结果，或返回数据集检查新版本。")) : task.terminalMessage}});
    }
    const bool followResult = success && !pendingTargetId_.isEmpty()
        && (page_->views->currentIndex() == DatasetWorkspacePage::Import
            || page_->views->currentIndex() == DatasetWorkspacePage::Conversion
            || page_->views->currentIndex() == DatasetWorkspacePage::Split
            || page_->views->currentIndex() == DatasetWorkspacePage::Catalog);
    if (followResult) {
        selectAfterRefresh_ = pendingTargetId_;
        catalogSearch_.clear(); catalogPresenter_->setCatalogFilter({});
        const QSignalBlocker blocker(page_->findChild<QLineEdit*>(QStringLiteral("DatasetCatalogSearch")));
        page_->findChild<QLineEdit*>(QStringLiteral("DatasetCatalogSearch"))->clear();
        state_.currentSnapshotId.clear();
        catalogCursors_ = {QString()};
    }
    pendingTargetId_.clear();
    refreshCatalog();
    if (followResult) page_->showView(DatasetWorkspacePage::Detail);
    emit statusChanged(message);
}

void DatasetPageController::loadQualityReport(const QString& taskText)
{
    if (!page_ || !projectOpen_ || taskText.isEmpty()) return;
    aitrain::TaskId taskId;
    QString error;
    if (!aitrain::TaskId::parse(taskText, &taskId, &error)) return;
    aitrain::TaskReadModel details;
    if (!queryService_->taskDetails(taskId, &details, &error)) {
        page_->validationSummaryLabel->setText(aitrain_app::workbenchText(QStringLiteral("读取质量记录失败：%1")).arg(error));
        return;
    }
    auto artifacts = details.artifacts;
    QString cursor = details.artifactNextCursor;
    bool hasMore = details.artifactsHasMore;
    while (hasMore) {
        const auto page = queryService_->taskArtifacts(taskId, {100, cursor}, &error);
        if (!error.isEmpty()) break;
        artifacts += page.items;
        cursor = page.nextCursor;
        hasMore = page.hasMore;
    }
    QString reportId;
    state_.latestQualityArtifactId.clear();
    state_.latestRepairArtifactId.clear();
    for (const auto& artifact : artifacts) {
        if (artifact.kind == QStringLiteral("dataset_quality_report")) reportId = artifact.id.toString();
        if (artifact.kind == QStringLiteral("dataset_repair_manifest")) state_.latestRepairArtifactId = artifact.id.toString();
    }
    state_.latestQualityArtifactId = reportId;
    if (reportId.isEmpty()) {
        page_->validationSummaryLabel->setText(details.task.failure.message.isEmpty()
            ? aitrain_app::workbenchText(QStringLiteral("该检查尚无已提交报告。"))
            : aitrain_app::workbenchText(QStringLiteral("检查未完成：%1")).arg(details.task.failure.message));
        return;
    }
    aitrain::ArtifactId artifactId;
    if (!aitrain::ArtifactId::parse(reportId, &artifactId, &error)) return;
    const quint64 generation = ++qualityGeneration_;
    const QString expectedSnapshot = state_.currentSnapshotId;
    QPointer<DatasetPageController> self(this);
    if (!queryService_->artifactFilePreviewAsync(artifactId, QStringLiteral("quality_report.json"), this,
            [self, generation, expectedSnapshot](bool ok, aitrain::ArtifactFilePreview preview, QString readError) {
                if (!self || !self->page_ || generation != self->qualityGeneration_
                    || expectedSnapshot != self->state_.currentSnapshotId) return;
                if (!ok) { self->page_->validationSummaryLabel->setText(aitrain_app::workbenchText(QStringLiteral("报告读取失败：%1")).arg(readError)); return; }
                const QJsonDocument document = QJsonDocument::fromJson(preview.content);
                const QJsonObject report = document.object();
                if (!document.isObject() || report.value(QStringLiteral("datasetSnapshotId")).toString() != expectedSnapshot) {
                    self->page_->validationSummaryLabel->setText(aitrain_app::workbenchText(QStringLiteral("报告与当前数据版本不匹配，已停止显示。")));
                    return;
                }
                self->renderQualityReport(report);
            }, 4 * 1024 * 1024, &error)) page_->validationSummaryLabel->setText(error);
}

void DatasetPageController::renderQualityReport(const QJsonObject& report)
{
    const QJsonArray issues = report.value(QStringLiteral("issues")).toArray();
    page_->validationSummaryLabel->setText(issues.isEmpty()
        ? aitrain_app::workbenchText(QStringLiteral("质量检查完成，未发现需复核的问题。"))
        : aitrain_app::workbenchText(QStringLiteral("质量检查完成，发现 %1 项待复核问题。")).arg(issues.size()));
    page_->validationOutput->setPlainText(QString::fromUtf8(QJsonDocument(report).toJson(QJsonDocument::Indented)));
    page_->validationIssuesTable->setRowCount(issues.size());
    for (int row = 0; row < issues.size(); ++row) {
        const QJsonObject issue = issues.at(row).toObject();
        const ReviewSamplePathView paths = reviewSamplePathView(issue);
        const QString file = paths.labelRelativePath.isEmpty() ? paths.imageRelativePath : paths.labelRelativePath;
        const QStringList values{issue.value(QStringLiteral("severity")).toString(), issue.value(QStringLiteral("code")).toString(), file,
            issue.contains(QStringLiteral("line")) ? QString::number(issue.value(QStringLiteral("line")).toInt()) : QString(),
            issue.value(QStringLiteral("message")).toString()};
        for (int column = 0; column < values.size(); ++column)
            page_->validationIssuesTable->setItem(row, column, new QTableWidgetItem(values.at(column)));
    }
    const int selected = page_->datasetListTable->currentRow();
    if (selected >= 0 && page_->datasetListTable->item(selected, 2)->data(Qt::UserRole).toString() == state_.currentSnapshotId) {
        page_->datasetListTable->item(selected, 2)->setText(issues.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("已检查")) : aitrain_app::workbenchText(QStringLiteral("需复核")));
        const auto count = report.value(QStringLiteral("summary")).toObject().value(QStringLiteral("sampleCount"));
        if (count.isDouble()) {
            sampleCounts_.insert(state_.currentSnapshotId, count.toInt());
            page_->datasetListTable->item(selected, 3)->setText(QString::number(count.toInt()));
        }
    }
    page_->datasetRepairLoopLabel->setText(issues.isEmpty()
        ? aitrain_app::workbenchText(QStringLiteral("当前质量报告没有待修复项。"))
        : aitrain_app::workbenchText(QStringLiteral("可从当前修复清单创建外部标注会话，完成后同步并重新检查。")));
}

void DatasetPageController::openQualityReport(bool repairList)
{
    if (!page_) return;
    const QString artifact = repairList ? state_.latestRepairArtifactId : state_.latestQualityArtifactId;
    if (artifact.isEmpty()) {
        QMessageBox::information(page_, aitrain_app::workbenchText(QStringLiteral("质量报告")), aitrain_app::workbenchText(QStringLiteral("当前数据版本尚无可打开的报告，请先检查质量。")));
        return;
    }
    showArtifactReport(page_, queryService_, artifact,
        repairList ? QStringLiteral("repair_manifest.json") : QStringLiteral("quality_report.html"),
        repairList ? aitrain_app::workbenchText(QStringLiteral("问题与修复清单")) : aitrain_app::workbenchText(QStringLiteral("数据质量报告")));
}
