#include "MainWindow.h"

#include "DatasetConversionUiModel.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
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
#include <QTextStream>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>

using namespace aitrain_app;

namespace {
QString sampleTextField(const QJsonObject& sample, const QStringList& keys)
{
    for (const QString& key : keys) {
        const QJsonValue value = sample.value(key);
        if (value.isString() && !value.toString().trimmed().isEmpty()) {
            return value.toString().trimmed();
        }
        if (value.isDouble()) {
            return QString::number(value.toDouble(), 'g', 8);
        }
        if (value.isObject() || value.isArray()) {
            const QString compact = QString::fromUtf8(QJsonDocument(QJsonArray{value}).toJson(QJsonDocument::Compact));
            return compact.mid(1, qMax(0, compact.size() - 2));
        }
    }
    return QString();
}

QString samplePathField(const QJsonObject& sample, const QStringList& keys)
{
    return QDir::fromNativeSeparators(sampleTextField(sample, keys));
}

QJsonObject normalizedReviewSample(QJsonObject sample, const QString& source)
{
    if (sample.value(QStringLiteral("source")).toString().isEmpty()) {
        sample.insert(QStringLiteral("source"), source);
    }
    if (sample.value(QStringLiteral("reason")).toString().isEmpty()) {
        const QString reason = sampleTextField(sample, QStringList()
            << QStringLiteral("code")
            << QStringLiteral("errorType")
            << QStringLiteral("type")
            << QStringLiteral("category"));
        if (!reason.isEmpty()) {
            sample.insert(QStringLiteral("reason"), reason);
        }
    }
    return sample;
}

void appendReviewSamplesFromArray(QJsonArray* target, const QJsonArray& sourceArray, const QString& source)
{
    if (!target) {
        return;
    }
    for (const QJsonValue& value : sourceArray) {
        if (value.isObject()) {
            target->append(normalizedReviewSample(value.toObject(), source));
        }
    }
}

QJsonArray extractReviewSamples(const QJsonDocument& document)
{
    QJsonArray samples;
    if (document.isArray()) {
        appendReviewSamplesFromArray(&samples, document.array(), QStringLiteral("array"));
        return samples;
    }
    const QJsonObject root = document.object();
    const QList<QPair<QString, QString>> keys = {
        qMakePair(QStringLiteral("problemSamples"), QStringLiteral("problem_samples")),
        qMakePair(QStringLiteral("errorSamples"), QStringLiteral("error_samples")),
        qMakePair(QStringLiteral("lowConfidenceSamples"), QStringLiteral("low_confidence")),
        qMakePair(QStringLiteral("samples"), QStringLiteral("samples")),
        qMakePair(QStringLiteral("reworkSamples"), QStringLiteral("rework_samples")),
        qMakePair(QStringLiteral("items"), QStringLiteral("items"))
    };
    for (const auto& item : keys) {
        appendReviewSamplesFromArray(&samples, root.value(item.first).toArray(), item.second);
    }
    const QJsonObject nested = root.value(QStringLiteral("payload")).toObject();
    if (!nested.isEmpty()) {
        for (const auto& item : keys) {
            appendReviewSamplesFromArray(&samples, nested.value(item.first).toArray(), item.second);
        }
    }
    return samples;
}

QString reviewMetricText(const QJsonObject& sample)
{
    QStringList parts;
    for (const QString& key : {
             QStringLiteral("confidence"),
             QStringLiteral("matchedIou"),
             QStringLiteral("matchedMaskIoU"),
             QStringLiteral("editDistance"),
             QStringLiteral("cer")}) {
        if (sample.contains(key)) {
            parts.append(QStringLiteral("%1=%2").arg(key, sampleTextField(sample, QStringList() << key)));
        }
    }
    const QJsonObject prediction = sample.value(QStringLiteral("prediction")).toObject();
    if (!prediction.isEmpty()) {
        const double confidence = prediction.value(QStringLiteral("confidence")).toDouble(-1.0);
        if (confidence >= 0.0) {
            parts.append(QStringLiteral("pred_conf=%1").arg(confidence, 0, 'f', 4));
        }
    }
    return parts.join(QStringLiteral(" | "));
}

QString reviewClassText(const QJsonObject& sample)
{
    const QString direct = sampleTextField(sample, QStringList()
        << QStringLiteral("className")
        << QStringLiteral("class")
        << QStringLiteral("category")
        << QStringLiteral("label"));
    if (!direct.isEmpty()) {
        return direct;
    }
    const QJsonObject prediction = sample.value(QStringLiteral("prediction")).toObject();
    const QJsonObject box = prediction.value(QStringLiteral("box")).toObject();
    if (box.contains(QStringLiteral("classId"))) {
        return QStringLiteral("class_%1").arg(box.value(QStringLiteral("classId")).toInt());
    }
    const QJsonObject groundTruth = sample.value(QStringLiteral("groundTruth")).toObject();
    if (groundTruth.contains(QStringLiteral("classId"))) {
        return QStringLiteral("class_%1").arg(groundTruth.value(QStringLiteral("classId")).toInt());
    }
    return QString();
}

void setAcceptanceTableRow(QTableWidget* table, const QString& stage, const QString& status, const QString& evidence, const QString& message)
{
    if (!table) {
        return;
    }
    int row = -1;
    for (int index = 0; index < table->rowCount(); ++index) {
        if (table->item(index, 0) && table->item(index, 0)->text() == stage) {
            row = index;
            break;
        }
    }
    if (row < 0) {
        row = table->rowCount();
        table->insertRow(row);
    }
    table->setItem(row, 0, new QTableWidgetItem(stage));
    table->setItem(row, 1, new QTableWidgetItem(status));
    table->setItem(row, 2, new QTableWidgetItem(QDir::toNativeSeparators(evidence)));
    table->setItem(row, 3, new QTableWidgetItem(message));
}

QString defaultDatasetConversionOutputPath(const QString& sourcePath, const QString& projectPath, const QString& targetFormat)
{
    const QString normalizedSourcePath = QDir::fromNativeSeparators(sourcePath.trimmed());
    if (normalizedSourcePath.isEmpty()) {
        return QString();
    }

    const QFileInfo sourceInfo(normalizedSourcePath);
    QString datasetName = sourceInfo.isFile() ? sourceInfo.completeBaseName() : sourceInfo.fileName();
    if (datasetName.isEmpty()) {
        datasetName = QStringLiteral("dataset");
    }
    const QString suffix = targetFormat.trimmed().isEmpty() ? QStringLiteral("converted") : targetFormat.trimmed();
    const QString directoryName = QStringLiteral("%1-%2").arg(datasetName, suffix);
    const QString normalizedProjectPath = QDir::fromNativeSeparators(projectPath.trimmed());
    if (!normalizedProjectPath.isEmpty()) {
        const QString conversionRoot = QDir(normalizedProjectPath).filePath(QStringLiteral("datasets/converted"));
        QDir().mkpath(conversionRoot);
        return QDir::cleanPath(QDir(conversionRoot).filePath(directoryName));
    }
    const QDir sourceDir(sourceInfo.isFile() ? sourceInfo.absolutePath() : normalizedSourcePath);
    const QString outputPath = sourceDir.absoluteFilePath(QStringLiteral("../converted/%1").arg(directoryName));
    return QDir::cleanPath(outputPath);
}

void setFieldErrorLabel(QLabel* label, const QString& text)
{
    if (!label) {
        return;
    }
    label->setText(text);
    label->setVisible(!text.isEmpty());
}
} // namespace

void MainWindow::validateDeploymentModelPackageV2()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("Worker 正在执行任务，稍后再运行交付工作流。"));
        return;
    }
    const QString modelPackageText = deploymentModelPackageCombo_
        ? deploymentModelPackageCombo_->currentData().toString().trimmed()
        : QString();
    const QString sampleImagePath = QDir::fromNativeSeparators(
        deploymentValidationImageEdit_ ? deploymentValidationImageEdit_->text().trimmed() : QString());
    aitrain::v2::ModelPackageId modelPackageId;
    QString error;
    if (!v2Workspace_.isOpen() || currentProjectPath_.isEmpty()
        || !aitrain::v2::ModelPackageId::parse(modelPackageText, &modelPackageId, &error)
        || sampleImagePath.isEmpty() || !QFileInfo(sampleImagePath).isFile()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("请选择已验证 V2 模型包和有效验证图片。"));
        return;
    }

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    QJsonObject options;
    options.insert(QStringLiteral("benchmarkWarmup"), 1);
    options.insert(QStringLiteral("benchmarkIterations"), 3);
    activeV2TaskId_ = taskId.toString();
    if (!worker_.requestRuntimeDeliveryWorkflowV2(workerExecutablePath(), currentProjectPath_,
            modelPackageId.toString(), QStringLiteral("aitrain_onnxruntime"), sampleImagePath,
            options, &error, activeV2TaskId_)) {
        activeV2TaskId_.clear();
        QMessageBox::critical(this, uiText("部署验证"), error);
        return;
    }
    if (deploymentValidationResultLabel_) {
        deploymentValidationResultLabel_->setText(uiText(
            "Runtime Delivery 已派发：等待六步状态与最终 Evidence。\n"
            "底层 ONNX Runtime 单次同步 infer 返回前不能中途抢占。"));
    }
    workerPill_->setStatus(uiText("Runtime Delivery 运行中"), StatusPill::Tone::Info);
}
void MainWindow::importOcrOfficialReportsV2()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("OCR 报告导入"), uiText("Worker 正在执行任务，稍后再导入。"));
        return;
    }
    const auto source = [](QLineEdit* report, QLineEdit* snapshot, QLineEdit* artifact) {
        return QJsonObject{{QStringLiteral("reportPath"), QDir::fromNativeSeparators(
                report ? report->text().trimmed() : QString())},
            {QStringLiteral("snapshotId"), snapshot ? snapshot->text().trimmed() : QString()},
            {QStringLiteral("snapshotArtifactId"), artifact ? artifact->text().trimmed() : QString()}};
    };
    const QJsonObject det = source(customerOcrDetReportEdit_, customerOcrDetSnapshotIdEdit_,
        customerOcrDetSnapshotArtifactIdEdit_);
    const QJsonObject rec = source(customerOcrRecReportEdit_, customerOcrRecSnapshotIdEdit_,
        customerOcrRecSnapshotArtifactIdEdit_);
    const QJsonObject system = source(customerOcrSystemReportEdit_, customerOcrSystemSnapshotIdEdit_,
        customerOcrSystemSnapshotArtifactIdEdit_);
    const auto validSource = [](const QJsonObject& value) {
        return QFileInfo(value.value(QStringLiteral("reportPath")).toString()).isFile()
            && (!value.value(QStringLiteral("snapshotId")).toString().isEmpty()
                || !value.value(QStringLiteral("snapshotArtifactId")).toString().isEmpty());
    };
    if (!v2Workspace_.isOpen() || currentProjectPath_.isEmpty()
        || !validSource(det) || !validSource(rec) || !validSource(system)
        || !customerOcrCohortIdEdit_ || customerOcrCohortIdEdit_->text().trimmed().isEmpty()
        || !customerOcrDomainIdEdit_ || customerOcrDomainIdEdit_->text().trimmed().isEmpty()) {
        QMessageBox::warning(this, uiText("OCR 报告导入"), uiText(
            "请选择三份官方 JSON，并为每份填写 SnapshotId 或 committed Snapshot ArtifactId；验收批次和客户域不能为空。"));
        return;
    }

    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    activeV2TaskId_ = taskId.toString();
    activeV2WorkflowKind_ = QStringLiteral("ocr_report_import_v2");
    QString error;
    if (!worker_.requestOcrOfficialReportImportV2(workerExecutablePath(), currentProjectPath_,
            det, rec, system, customerOcrCohortIdEdit_->text().trimmed(),
            customerOcrDomainIdEdit_->text().trimmed(),
            customerOcrEvidenceClassCombo_ ? customerOcrEvidenceClassCombo_->currentText()
                                           : QStringLiteral("customer_domain"),
            &error, activeV2TaskId_)) {
        activeV2TaskId_.clear();
        activeV2WorkflowKind_.clear();
        QMessageBox::critical(this, uiText("OCR 报告导入"), error);
        return;
    }
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("正在受控导入三份官方报告；原始路径不会进入结果事件。"));
    }
    workerPill_->setStatus(uiText("OCR 报告导入中"), StatusPill::Tone::Info);
}

void MainWindow::runOcrAcceptanceWorkflowV2()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("OCR 验收"), uiText("Worker 正在执行任务，稍后再运行验收。"));
        return;
    }
    aitrain::v2::ArtifactId det;
    aitrain::v2::ArtifactId rec;
    aitrain::v2::ArtifactId system;
    QString error;
    if (!v2Workspace_.isOpen() || currentProjectPath_.isEmpty()
        || !aitrain::v2::ArtifactId::parse(customerOcrDetReportArtifactIdEdit_
                ? customerOcrDetReportArtifactIdEdit_->text().trimmed() : QString(), &det, &error)
        || !aitrain::v2::ArtifactId::parse(customerOcrRecReportArtifactIdEdit_
                ? customerOcrRecReportArtifactIdEdit_->text().trimmed() : QString(), &rec, &error)
        || !aitrain::v2::ArtifactId::parse(customerOcrSystemReportArtifactIdEdit_
                ? customerOcrSystemReportArtifactIdEdit_->text().trimmed() : QString(), &system, &error)) {
        QMessageBox::warning(this, uiText("OCR 验收"), uiText("验收只接受三个已提交官方报告 ArtifactId。"));
        return;
    }
    const QJsonObject thresholds{
        {QStringLiteral("minimumDetSamples"), 1},
        {QStringLiteral("minimumRecSamples"), 1},
        {QStringLiteral("minimumSystemSamples"), 1},
        {QStringLiteral("minimumDetHmean"), customerOcrMinDetHmeanEdit_ ? customerOcrMinDetHmeanEdit_->text().toDouble() : 0.50},
        {QStringLiteral("minimumRecAccuracy"), customerOcrMinAccEdit_ ? customerOcrMinAccEdit_->text().toDouble() : 0.70},
        {QStringLiteral("maximumRecCer"), customerOcrMaxCerEdit_ ? customerOcrMaxCerEdit_->text().toDouble() : 0.30},
        {QStringLiteral("minimumSystemAccuracy"), customerOcrMinSystemAccEdit_ ? customerOcrMinSystemAccEdit_->text().toDouble() : 0.70}};
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    activeV2TaskId_ = taskId.toString();
    activeV2WorkflowKind_ = QStringLiteral("ocr_acceptance_v2");
    if (!worker_.requestOcrAcceptanceWorkflowV2(workerExecutablePath(), currentProjectPath_,
            det.toString(), rec.toString(), system.toString(), thresholds, &error,
            activeV2TaskId_)) {
        activeV2TaskId_.clear();
        activeV2WorkflowKind_.clear();
        QMessageBox::critical(this, uiText("OCR 验收"), error);
        return;
    }
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("OCR Acceptance V2 四步验收运行中。"));
    }
    workerPill_->setStatus(uiText("OCR 验收中"), StatusPill::Tone::Info);
}

void MainWindow::collectDiagnosticsBundle()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("诊断包"), uiText("Worker 正在执行任务，稍后再生成诊断包。"));
        return;
    }

    if (!v2Workspace_.isOpen() || currentProjectPath_.isEmpty()) {
        QMessageBox::warning(this, uiText("诊断包"), uiText("请先打开 V2 项目。"));
        return;
    }
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    activeV2TaskId_ = taskId.toString();
    activeV2WorkflowKind_ = QStringLiteral("diagnostics_v2");
    QString error;
    if (!worker_.requestDiagnosticsWorkflowV2(workerExecutablePath(), currentProjectPath_,
            QJsonObject(), &error, activeV2TaskId_)) {
        activeV2TaskId_.clear();
        activeV2WorkflowKind_.clear();
        QMessageBox::critical(this, uiText("诊断包"), error);
        return;
    }
    if (diagnosticsStatusLabel_) {
        diagnosticsStatusLabel_->setText(uiText("诊断包生成中。"));
    }
    workerPill_->setStatus(uiText("诊断包生成中"), StatusPill::Tone::Info);
}

void MainWindow::importAcceptanceEvidence()
{
    const QString file = QFileDialog::getOpenFileName(
        this,
        uiText("导入验收结果"),
        currentProjectPath_,
        QStringLiteral("Acceptance evidence (*.json *.md *.txt);;All files (*.*)"));
    if (file.isEmpty()) {
        return;
    }
    QString status = QStringLiteral("imported");
    QString stage = QFileInfo(file).completeBaseName();
    QString message = uiText("已导入外部验收结果。");
    if (QFileInfo(file).suffix().compare(QStringLiteral("json"), Qt::CaseInsensitive) == 0) {
        QFile jsonFile(file);
        if (jsonFile.open(QIODevice::ReadOnly)) {
            const QJsonDocument document = QJsonDocument::fromJson(jsonFile.readAll());
            const QJsonObject object = document.object();
            status = object.value(QStringLiteral("status")).toString(object.value(QStringLiteral("ok")).toBool(false) ? QStringLiteral("passed") : QStringLiteral("blocked"));
            stage = object.value(QStringLiteral("kind")).toString(stage);
            message = object.value(QStringLiteral("message")).toString(object.value(QStringLiteral("note")).toString(message));
        }
    } else {
        QFile textFile(file);
        if (textFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            const QString text = QString::fromUtf8(textFile.readAll()).toLower();
            if (text.contains(QStringLiteral("hardware-blocked"))) {
                status = QStringLiteral("hardware-blocked");
            } else if (text.contains(QStringLiteral("blocked")) || text.contains(QStringLiteral("failed"))) {
                status = QStringLiteral("blocked");
            } else if (text.contains(QStringLiteral("passed"))) {
                status = QStringLiteral("passed");
            }
        }
    }
    setAcceptanceTableRow(deliveryAcceptanceTable_, stage, status, file, message);
    updateDeliveryAcceptanceSummary();
}
