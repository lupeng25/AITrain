#include "WorkbenchTranslation.h"
#include "DeliveryEvidencePageController.h"

#include "DeliveryEvidencePage.h"
#include "DeliveryEvidencePresenter.h"
#include "DiagnosticBundlePresenter.h"
#include "TaskRuntimeController.h"
#include "ProjectObjectSelectors.h"
#include "ApplicationEventRouter.h"
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
    , queryService_(queryService)
{
    connect(this, &DeliveryEvidencePageController::taskStarted, this, [this](const QString& id, const QString&) { activeTaskId_ = id; });
    connect(presenter_, &DeliveryEvidencePresenter::changed,
        this, &DeliveryEvidencePageController::render);
    connect(presenter_, &DeliveryEvidencePresenter::queryFailed,
        this, [this](const QString&) { render(); });
}

void DeliveryEvidencePageController::attach(
    DeliveryEvidenceWorkspacePage* page)
{
    page_ = page;
    aitrain_app::bindCatalogSearch(page_->findChild<QLineEdit*>(QStringLiteral("EvidenceCatalogSearch")), this, [this](const QString& text) {
        presenter_->clear();
        presenter_->setCatalogFilter({text, {}, {}}); refresh();
    });
    connect(page_, &DeliveryEvidenceWorkspacePage::browseRequested, this, &DeliveryEvidencePageController::browseReport);
    connect(page_, &DeliveryEvidenceWorkspacePage::bindSnapshotRequested, this, &DeliveryEvidencePageController::bindSnapshot);
    connect(page_, &DeliveryEvidenceWorkspacePage::selectReportRequested, this, &DeliveryEvidencePageController::selectReport);
    connect(page_, &DeliveryEvidenceWorkspacePage::refreshRequested, this, &DeliveryEvidencePageController::refresh);
    connect(page_, &DeliveryEvidenceWorkspacePage::moreRequested, this, [this]() { presenter_->loadMoreAsync(); });
    connect(page_, &DeliveryEvidenceWorkspacePage::importOcrRequested, this, &DeliveryEvidencePageController::importOcrOfficialReports);
    connect(page_, &DeliveryEvidenceWorkspacePage::acceptanceRequested, this, &DeliveryEvidencePageController::runOcrAcceptance);
    connect(page_, &DeliveryEvidenceWorkspacePage::diagnosticsRequested, this, &DeliveryEvidencePageController::collectDiagnostics);
    connect(page_, &DeliveryEvidenceWorkspacePage::externalImportRequested, this, &DeliveryEvidencePageController::importAcceptanceEvidence);
    connect(page_, &DeliveryEvidenceWorkspacePage::openSelectedRequested, this, [this]() {
        const auto* item = page_->acceptanceTable->item(page_->acceptanceTable->currentRow(), 0);
        if (item) emit page_->openTaskRequested(item->data(Qt::UserRole).toString());
    });
    refresh();
}

void DeliveryEvidencePageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    projectOpen_ = projectOpen;
    if (projectRoot_ != projectRoot) {
        activeTaskId_.clear(); presenter_->setCatalogFilter({}); presenter_->clear();
        if (page_) {
            for (auto* edit : page_->findChildren<QLineEdit*>()) if (edit->validator() == nullptr) edit->clear();
            for (auto* label : page_->snapshotLabels) label->setText(aitrain_app::workbenchText(QStringLiteral("尚未绑定数据版本")));
            for (auto* label : page_->reportLabels) label->setText(aitrain_app::workbenchText(QStringLiteral("尚未选择报告")));
            page_->ocrStatusLabel->clear(); page_->setMode(DeliveryEvidenceWorkspacePage::Catalog);
        }
    }
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
    if (projectOpen_) presenter_->refreshAsync();
    render();
}

void DeliveryEvidencePageController::render()
{
    if (!page_) return;
    auto* table = page_->acceptanceTable;
    const QString selected = table->currentRow() >= 0 ? table->item(table->currentRow(), 0)->data(Qt::UserRole).toString() : QString();
    table->setRowCount(0);
    for (const auto& evidence : presenter_->viewModel().records) {
        const int row = table->rowCount(); table->insertRow(row);
        const QStringList values = {aitrain_app::artifactDisplayName(evidence.evidenceKind), evidence.runtimeStatus.isEmpty() ? evidence.taskState : evidence.runtimeStatus,
            !evidence.valid ? aitrain_app::workbenchText(QStringLiteral("证据无效")) : evidence.verified ? aitrain_app::workbenchText(QStringLiteral("已校验")) : aitrain_app::workbenchText(QStringLiteral("待核验")),
            evidence.valid ? evidence.limitations.join(QStringLiteral("；")) : evidence.validationFailure.message};
        for (int c = 0; c < values.size(); ++c) table->setItem(row, c, new QTableWidgetItem(values[c]));
        table->item(row, 0)->setData(Qt::UserRole, evidence.taskId.toString());
        if (evidence.taskId.toString() == selected) table->selectRow(row);
    }
    page_->moreButton->setVisible(presenter_->hasMore());
    page_->acceptanceSummaryLabel->setText(!projectOpen_ ? aitrain_app::workbenchText(QStringLiteral("请先打开项目。"))
        : !presenter_->lastError().isEmpty() ? presenter_->lastError()
        : table->rowCount() == 0 ? aitrain_app::workbenchText(QStringLiteral("尚无验收证据。可导入官方报告或外部验收结果。"))
        : aitrain_app::workbenchText(QStringLiteral("已载入 %1 条证据。运行状态与证据校验分别显示；精度结论以对应报告为准。")).arg(table->rowCount()));
}

void DeliveryEvidencePageController::bindSnapshot(int index)
{
    if (index < 0 || index > 2) return;
    aitrain_app::DatasetSelection selected;
    if (!aitrain_app::selectProjectDataset(page_, queryService_, &selected)) return;
    QLineEdit* snapshots[] = {page_->ocrDetSnapshotIdEdit, page_->ocrRecSnapshotIdEdit, page_->ocrSystemSnapshotIdEdit};
    QLineEdit* artifacts[] = {page_->ocrDetSnapshotArtifactIdEdit, page_->ocrRecSnapshotArtifactIdEdit, page_->ocrSystemSnapshotArtifactIdEdit};
    snapshots[index]->setText(selected.snapshot.snapshotId.toString()); artifacts[index]->setText(selected.snapshot.artifactId.toString());
    page_->snapshotLabels[index]->setText(selected.displayName + QStringLiteral(" · ") + selected.snapshot.createdAt.toLocalTime().toString(QStringLiteral("MM-dd HH:mm")));
}

void DeliveryEvidencePageController::selectReport(int index)
{
    if (index < 0 || index > 2) return;
    const QStringList kinds = {QStringLiteral("paddleocr_det_official_report"), QStringLiteral("paddleocr_rec_official_report"), QStringLiteral("paddleocr_system_official_report")};
    const QString id = aitrain_app::selectProjectArtifact(page_, queryService_, {kinds[index]}, aitrain_app::workbenchText(QStringLiteral("选择官方报告")));
    if (id.isEmpty()) return;
    QLineEdit* reports[] = {page_->ocrDetReportArtifactIdEdit, page_->ocrRecReportArtifactIdEdit, page_->ocrSystemReportArtifactIdEdit};
    reports[index]->setText(id); page_->reportLabels[index]->setText(aitrain_app::workbenchText(QStringLiteral("已选择已提交官方报告"))); page_->reportLabels[index]->setToolTip(id);
}

void DeliveryEvidencePageController::applyTaskViewState(const TaskViewState& state)
{
    if (!page_ || state.taskId != activeTaskId_) return;
    const QString text = state.terminal ? aitrain_app::workbenchText(QStringLiteral("任务结束：%1 · %2")).arg(state.status, state.terminalMessage)
        : aitrain_app::workbenchText(QStringLiteral("任务运行中 · %1%")).arg(state.progress);
    page_->ocrStatusLabel->setText(text); page_->diagnosticsStatusLabel->setText(text);
    if (!state.terminal) return;
    QLineEdit* reports[] = {page_->ocrDetReportArtifactIdEdit, page_->ocrRecReportArtifactIdEdit, page_->ocrSystemReportArtifactIdEdit};
    const QStringList kinds = {QStringLiteral("paddleocr_det_official_report"), QStringLiteral("paddleocr_rec_official_report"), QStringLiteral("paddleocr_system_official_report")};
    for (const auto& artifact : state.artifacts) {
        const int index = kinds.indexOf(artifact.kind);
        if (index >= 0) { reports[index]->setText(artifact.artifactId); page_->reportLabels[index]->setText(aitrain_app::workbenchText(QStringLiteral("本次导入的官方报告"))); }
    }
    refresh();
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
            "请选择三份官方报告并分别绑定已提交数据版本；验收批次和客户域不能为空。"));
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
    for (const auto* field : {page_->ocrMinDetHmeanEdit, page_->ocrMinAccEdit, page_->ocrMaxCerEdit, page_->ocrMinSystemAccEdit}) {
        if (!field->hasAcceptableInput()) { page_->ocrStatusLabel->setText(aitrain_app::workbenchText(QStringLiteral("验收阈值必须填写 0 到 1 之间的数值。"))); return; }
    }
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
