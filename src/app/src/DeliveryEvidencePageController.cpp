#include "DeliveryEvidencePageController.h"

#include "DeliveryEvidencePage.h"
#include "DeliveryEvidencePresenter.h"
#include "DiagnosticBundlePresenter.h"
#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QTableWidget>
#include <QTableWidgetItem>

DeliveryEvidencePageController::DeliveryEvidencePageController(
    const aitrain::ProjectQueryService* queryService,
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , presenter_(new DeliveryEvidencePresenter(queryService, this))
    , diagnosticPresenter_(new DiagnosticBundlePresenter(queryService, this))
    , taskRuntime_(taskRuntime)
{
    connect(presenter_, &DeliveryEvidencePresenter::changed,
        this, &DeliveryEvidencePageController::render);
    connect(presenter_, &DeliveryEvidencePresenter::queryFailed,
        this, [this](const QString&) { render(); });
}

void DeliveryEvidencePageController::attach(
    DeliveryEvidenceWorkspacePage* page)
{
    page_ = page;
    refresh();
}

void DeliveryEvidencePageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
    if (!projectOpen_) presenter_->clear();
}

void DeliveryEvidencePageController::setWorkerExecutable(
    const QString& executable)
{
    workerExecutable_ = executable;
}

void DeliveryEvidencePageController::browseReport(QLineEdit* target)
{
    if (!page_ || !target) return;
    const QString selected = QFileDialog::getOpenFileName(page_,
        tr("选择文件"), projectRoot_,
        tr("PaddleOCR 官方报告 (*.json);;所有文件 (*.*)"));
    if (!selected.isEmpty()) {
        target->setText(QDir::toNativeSeparators(selected));
    }
}

void DeliveryEvidencePageController::refresh()
{
    if (!page_) return;
    if (page_->acceptanceTable->rowCount() == 0) {
        const QStringList stages = {
            tr("本机 RC"), tr("Clean Windows"), tr("TensorRT"),
            tr("客户域 OCR"), tr("包体完整性"), tr("部署验证"),
            tr("诊断包")};
        for (const QString& stage : stages) {
            const int row = page_->acceptanceTable->rowCount();
            page_->acceptanceTable->insertRow(row);
            page_->acceptanceTable->setItem(
                row, 0, new QTableWidgetItem(stage));
            page_->acceptanceTable->setItem(
                row, 1, new QTableWidgetItem(QStringLiteral("not_run")));
            page_->acceptanceTable->setItem(
                row, 2, new QTableWidgetItem(QString()));
            page_->acceptanceTable->setItem(row, 3,
                new QTableWidgetItem(
                    tr("等待导入外部结果或运行对应 Worker/脚本。")));
        }
    }
    if (projectOpen_) presenter_->refreshAsync();
    render();
}

void DeliveryEvidencePageController::render()
{
    if (!page_ || !page_->acceptanceTable) return;
    QTableWidget* table = page_->acceptanceTable;
    for (const auto& evidence : presenter_->viewModel().records) {
        QString stage = evidence.evidenceKind;
        const QString kind = evidence.evidenceKind.toLower();
        if (kind.contains(QStringLiteral("clean"))) {
            stage = tr("Clean Windows");
        } else if (kind.contains(QStringLiteral("tensor"))) {
            stage = tr("TensorRT");
        } else if (kind.contains(QStringLiteral("ocr"))) {
            stage = tr("客户域 OCR");
        }
        int row = -1;
        for (int index = 0; index < table->rowCount(); ++index) {
            if (table->item(index, 0)
                && table->item(index, 0)->text() == stage) {
                row = index;
                break;
            }
        }
        if (row < 0) {
            row = table->rowCount();
            table->insertRow(row);
            table->setItem(row, 0, new QTableWidgetItem(stage));
        }
        table->setItem(row, 1, new QTableWidgetItem(
            evidence.verified
                ? QStringLiteral("passed") : QStringLiteral("collected")));
        table->setItem(row, 2, new QTableWidgetItem(
            evidence.evidenceArtifactId.toString()));
        table->setItem(row, 3, new QTableWidgetItem(
            evidence.limitations.join(QStringLiteral(" | "))));
    }
    int passed = 0;
    int blocked = 0;
    int hardwareBlocked = 0;
    int collected = 0;
    int notRun = 0;
    for (int row = 0; row < table->rowCount(); ++row) {
        const QString status = table->item(row, 1)
            ? table->item(row, 1)->text() : QString();
        if (status == QStringLiteral("passed")) ++passed;
        else if (status == QStringLiteral("blocked")
            || status == QStringLiteral("failed")) ++blocked;
        else if (status == QStringLiteral("hardware-blocked")) {
            ++hardwareBlocked;
        } else if (status == QStringLiteral("collected")
            || status == QStringLiteral("imported")) {
            ++collected;
        } else {
            ++notRun;
        }
    }
    page_->acceptanceSummaryLabel->setText(
        tr("验收状态：passed %1 / blocked %2 / hardware-blocked %3 / collected %4 / not-run %5")
            .arg(passed).arg(blocked).arg(hardwareBlocked)
            .arg(collected).arg(notRun));
}

void DeliveryEvidencePageController::importOcrOfficialReports()
{
    if (!page_) return;
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(
            page_, tr("OCR 报告导入"), tr("Worker 正在执行任务，稍后再导入。"));
        return;
    }
    const auto source = [](QLineEdit* report, QLineEdit* snapshot,
                            QLineEdit* artifact) {
        return QJsonObject{
            {QStringLiteral("reportPath"), QDir::fromNativeSeparators(
                report->text().trimmed())},
            {QStringLiteral("snapshotId"), snapshot->text().trimmed()},
            {QStringLiteral("snapshotArtifactId"),
                artifact->text().trimmed()}};
    };
    const QJsonObject det = source(page_->ocrDetReportEdit,
        page_->ocrDetSnapshotIdEdit, page_->ocrDetSnapshotArtifactIdEdit);
    const QJsonObject rec = source(page_->ocrRecReportEdit,
        page_->ocrRecSnapshotIdEdit, page_->ocrRecSnapshotArtifactIdEdit);
    const QJsonObject system = source(page_->ocrSystemReportEdit,
        page_->ocrSystemSnapshotIdEdit,
        page_->ocrSystemSnapshotArtifactIdEdit);
    const auto valid = [](const QJsonObject& value) {
        return QFileInfo(value.value(
            QStringLiteral("reportPath")).toString()).isFile()
            && (!value.value(QStringLiteral("snapshotId")).toString().isEmpty()
                || !value.value(QStringLiteral("snapshotArtifactId"))
                    .toString().isEmpty());
    };
    if (!projectOpen_ || projectRoot_.isEmpty()
        || !valid(det) || !valid(rec) || !valid(system)
        || page_->ocrCohortIdEdit->text().trimmed().isEmpty()
        || page_->ocrDomainIdEdit->text().trimmed().isEmpty()) {
        QMessageBox::warning(page_, tr("OCR 报告导入"), tr(
            "请选择三份官方 JSON，并为每份填写 SnapshotId 或 committed Snapshot ArtifactId；验收批次和客户域不能为空。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::OcrOfficialReportImportCommand command;
    command.context = {taskId, projectRoot_};
    command.det = det;
    command.rec = rec;
    command.system = system;
    command.acceptanceCohortId =
        page_->ocrCohortIdEdit->text().trimmed();
    command.customerDomainId = page_->ocrDomainIdEdit->text().trimmed();
    command.evidenceClass = page_->ocrEvidenceClassCombo->currentText();
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("OCR 报告导入"), error);
        return;
    }
    page_->ocrStatusLabel->setText(
        tr("正在受控导入三份官方报告；原始路径不会进入结果事件。"));
    emit taskStarted(taskId.toString(), QStringLiteral("ocr_report_import"));
    emit statusChanged(tr("OCR 报告导入中"));
}

void DeliveryEvidencePageController::runOcrAcceptance()
{
    if (!page_) return;
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(
            page_, tr("OCR 验收"), tr("Worker 正在执行任务，稍后再运行验收。"));
        return;
    }
    aitrain::ArtifactId det;
    aitrain::ArtifactId rec;
    aitrain::ArtifactId system;
    QString error;
    if (!projectOpen_ || projectRoot_.isEmpty()
        || !aitrain::ArtifactId::parse(
            page_->ocrDetReportArtifactIdEdit->text().trimmed(),
            &det, &error)
        || !aitrain::ArtifactId::parse(
            page_->ocrRecReportArtifactIdEdit->text().trimmed(),
            &rec, &error)
        || !aitrain::ArtifactId::parse(
            page_->ocrSystemReportArtifactIdEdit->text().trimmed(),
            &system, &error)) {
        QMessageBox::warning(page_, tr("OCR 验收"),
            tr("验收只接受三个已提交官方报告 ArtifactId。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::OcrAcceptanceCommand command;
    command.context = {taskId, projectRoot_};
    command.detReportArtifactId = det.toString();
    command.recReportArtifactId = rec.toString();
    command.systemReportArtifactId = system.toString();
    command.thresholds = QJsonObject{
        {QStringLiteral("minimumDetSamples"), 1},
        {QStringLiteral("minimumRecSamples"), 1},
        {QStringLiteral("minimumSystemSamples"), 1},
        {QStringLiteral("minimumDetHmean"),
            page_->ocrMinDetHmeanEdit->text().toDouble()},
        {QStringLiteral("minimumRecAccuracy"),
            page_->ocrMinAccEdit->text().toDouble()},
        {QStringLiteral("maximumRecCer"),
            page_->ocrMaxCerEdit->text().toDouble()},
        {QStringLiteral("minimumSystemAccuracy"),
            page_->ocrMinSystemAccEdit->text().toDouble()}};
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("OCR 验收"), error);
        return;
    }
    page_->ocrStatusLabel->setText(tr("OCR Acceptance 四步验收运行中。"));
    emit taskStarted(taskId.toString(), QStringLiteral("ocr_acceptance"));
    emit statusChanged(tr("OCR 验收中"));
}

void DeliveryEvidencePageController::collectDiagnostics()
{
    if (!page_) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(page_, tr("诊断包"),
            tr("请先打开项目，并等待当前 Worker 任务结束。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::DiagnosticsCommand command;
    command.context = {taskId, projectRoot_};
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("诊断包"), error);
        return;
    }
    page_->diagnosticsStatusLabel->setText(tr("诊断包生成中。"));
    emit taskStarted(taskId.toString(), QStringLiteral("diagnostics"));
    emit statusChanged(tr("诊断包生成中"));
}

void DeliveryEvidencePageController::importAcceptanceEvidence()
{
    if (!page_) return;
    const QString file = QFileDialog::getOpenFileName(page_,
        tr("导入验收结果"), projectRoot_,
        tr("结构化验收证据 (*.json);;所有文件 (*.*)"));
    if (file.isEmpty()) return;
    if (taskRuntime_->isRunning() || !projectOpen_ || projectRoot_.isEmpty()) {
        QMessageBox::warning(
            page_, tr("导入验收结果"), tr("请先打开项目并等待当前任务结束。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::ExternalAcceptanceEvidenceImportCommand command;
    command.context = {taskId, projectRoot_};
    command.sourcePath =
        QDir::cleanPath(QDir::fromNativeSeparators(file));
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("导入验收结果"), error);
        return;
    }
    emit taskStarted(
        taskId.toString(), QStringLiteral("external_acceptance_evidence"));
    emit statusChanged(tr("外部验收证据校验中"));
}
