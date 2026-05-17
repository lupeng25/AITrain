#include "MainWindow.h"

#include "DatasetConversionUiModel.h"
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
#include <QTextStream>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

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

void MainWindow::validateDeploymentArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("Worker 正在执行任务，稍后再验证部署产物。"));
        return;
    }
    QString modelPath = QDir::fromNativeSeparators(conversionOutputEdit_ ? conversionOutputEdit_->text().trimmed() : QString());
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        modelPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
    }
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("部署验证"), uiText("请选择已导出的模型产物或模型输入。"));
        return;
    }
    const QString sampleImagePath = QDir::fromNativeSeparators(conversionValidationImageEdit_ ? conversionValidationImageEdit_->text().trimmed() : QString());
    const QString format = conversionFormatCombo_ ? conversionFormatCombo_->currentData().toString() : QString();
    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Benchmark,
            QStringLiteral("deployment_validation"),
            QStringLiteral("com.aitrain.plugins.yolo_native"),
            outputPath,
            uiText("部署产物验证中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    } else {
        outputPath = QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("deployment_validation"));
    }

    QJsonObject options;
    QString error;
    if (!worker_.requestDeploymentValidation(workerExecutablePath(), modelPath, outputPath, format, sampleImagePath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("部署验证"), error);
        return;
    }
    if (deploymentValidationResultLabel_) {
        deploymentValidationResultLabel_->setText(uiText("正在验证部署产物：%1").arg(QDir::toNativeSeparators(modelPath)));
    }
    workerPill_->setStatus(uiText("部署验证中"), StatusPill::Tone::Info);
}

void MainWindow::runCustomerOcrAcceptance()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("OCR 验收"), uiText("Worker 正在执行任务，稍后再运行 OCR 验收。"));
        return;
    }
    QJsonObject options;
    options.insert(QStringLiteral("detDatasetPath"), QDir::fromNativeSeparators(customerOcrDetDatasetEdit_ ? customerOcrDetDatasetEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("recDatasetPath"), QDir::fromNativeSeparators(customerOcrRecDatasetEdit_ ? customerOcrRecDatasetEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("systemImagesPath"), QDir::fromNativeSeparators(customerOcrSystemImagesEdit_ ? customerOcrSystemImagesEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("detReportPath"), QDir::fromNativeSeparators(customerOcrDetReportEdit_ ? customerOcrDetReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("recReportPath"), QDir::fromNativeSeparators(customerOcrRecReportEdit_ ? customerOcrRecReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("systemReportPath"), QDir::fromNativeSeparators(customerOcrSystemReportEdit_ ? customerOcrSystemReportEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("detOnnxEvidencePath"), QDir::fromNativeSeparators(customerOcrDetOnnxEvidenceEdit_ ? customerOcrDetOnnxEvidenceEdit_->text().trimmed() : QString()));
    options.insert(QStringLiteral("minRecAccuracy"), customerOcrMinAccEdit_ ? customerOcrMinAccEdit_->text().toDouble() : 0.70);
    options.insert(QStringLiteral("maxRecCer"), customerOcrMaxCerEdit_ ? customerOcrMaxCerEdit_->text().toDouble() : 0.30);
    options.insert(QStringLiteral("requireFullDomainEvidence"), true);
    options.insert(QStringLiteral("allowPublicLikeData"), customerOcrAllowPublicCheck_ && customerOcrAllowPublicCheck_->isChecked());
    options.insert(QStringLiteral("requireDetOnnxEvidence"), customerOcrRequireDetOnnxCheck_ && customerOcrRequireDetOnnxCheck_->isChecked());
    options.insert(QStringLiteral("minDetSamples"), 1);
    options.insert(QStringLiteral("minRecSamples"), 1);
    options.insert(QStringLiteral("minSystemImages"), 1);

    QString taskId;
    QString outputPath = QDir::fromNativeSeparators(customerOcrOutputEdit_ ? customerOcrOutputEdit_->text().trimmed() : QString());
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        if (outputPath.isEmpty()) {
            outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        }
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("customer_ocr_acceptance"),
            QStringLiteral("com.aitrain.plugins.ocr_rec_native"),
            outputPath,
            uiText("客户域 OCR 验收中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QString error;
    if (!worker_.requestCustomerOcrAcceptance(workerExecutablePath(), outputPath, options, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("OCR 验收"), error);
        return;
    }
    if (customerOcrStatusLabel_) {
        customerOcrStatusLabel_->setText(uiText("客户域 OCR 验收运行中。"));
    }
    workerPill_->setStatus(uiText("OCR 验收中"), StatusPill::Tone::Info);
}

void MainWindow::collectDiagnosticsBundle()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("诊断包"), uiText("Worker 正在执行任务，稍后再生成诊断包。"));
        return;
    }

    QJsonArray recentTasks;
    QJsonArray recentFailures;
    QJsonArray artifactIndex;
    if (repository_.isOpen()) {
        QString error;
        const QVector<aitrain::TaskRecord> tasks = repository_.recentTasks(20, &error);
        for (const aitrain::TaskRecord& task : tasks) {
            QJsonObject taskObject;
            taskObject.insert(QStringLiteral("id"), task.id);
            taskObject.insert(QStringLiteral("kind"), aitrain::taskKindToString(task.kind));
            taskObject.insert(QStringLiteral("state"), aitrain::taskStateToString(task.state));
            taskObject.insert(QStringLiteral("taskType"), task.taskType);
            taskObject.insert(QStringLiteral("workDir"), task.workDir);
            taskObject.insert(QStringLiteral("message"), task.message);
            taskObject.insert(QStringLiteral("updatedAt"), task.updatedAt.toUTC().toString(Qt::ISODateWithMs));
            recentTasks.append(taskObject);
            if (task.state == aitrain::TaskState::Failed) {
                recentFailures.append(taskObject);
            }
            const QVector<aitrain::ArtifactRecord> artifacts = repository_.artifactsForTask(task.id, &error);
            for (const aitrain::ArtifactRecord& artifact : artifacts) {
                artifactIndex.append(QJsonObject{
                    {QStringLiteral("taskId"), artifact.taskId},
                    {QStringLiteral("kind"), artifact.kind},
                    {QStringLiteral("path"), artifact.path},
                    {QStringLiteral("message"), artifact.message}
                });
            }
        }
    }

    QJsonObject context;
    context.insert(QStringLiteral("projectName"), currentProjectName_);
    context.insert(QStringLiteral("projectPath"), currentProjectPath_);
    context.insert(QStringLiteral("workerExecutable"), workerExecutablePath());
    context.insert(QStringLiteral("recentTasks"), recentTasks);
    context.insert(QStringLiteral("recentFailures"), recentFailures);
    context.insert(QStringLiteral("artifactIndex"), artifactIndex);
    context.insert(QStringLiteral("licenseSummary"), QJsonObject{
        {QStringLiteral("status"), licenseOwner_.isEmpty() ? QStringLiteral("unknown") : QStringLiteral("registered")},
        {QStringLiteral("owner"), licenseOwner_},
        {QStringLiteral("expiry"), licenseExpiry_}
    });
    context.insert(QStringLiteral("pluginSummary"), QJsonObject{
        {QStringLiteral("count"), pluginManager_.plugins().size()},
        {QStringLiteral("searchPaths"), QJsonArray::fromStringList(pluginSearchPaths())}
    });

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("diagnostic_bundle"),
            QStringLiteral("com.aitrain.system"),
            outputPath,
            uiText("诊断包生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    } else {
        outputPath = QDir(QDir::tempPath()).filePath(QStringLiteral("aitrain-diagnostics"));
    }

    QString error;
    if (!worker_.requestDiagnosticsBundle(workerExecutablePath(), outputPath, context, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
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

void MainWindow::generateDeliveryReportFromSelectedArtifact()
{
    if (worker_.isRunning()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("Worker 正在执行任务，稍后再生成交付报告。"));
        return;
    }
    const QString modelPath = selectedArtifactPath();
    if (modelPath.isEmpty()) {
        QMessageBox::warning(this, uiText("交付报告"), uiText("请先选择一个模型或报告产物。"));
        return;
    }

    QString taskId;
    QString outputPath;
    if (repository_.isOpen()) {
        taskId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        outputPath = QDir(currentProjectPath_).filePath(QStringLiteral("runs/%1").arg(taskId));
        taskId = createRepositoryTask(
            aitrain::TaskKind::Report,
            QStringLiteral("delivery_report"),
            QStringLiteral("com.aitrain.workflow"),
            outputPath,
            uiText("训练交付报告生成中。"),
            taskId);
        if (taskId.isEmpty()) {
            return;
        }
    }

    QJsonObject context;
    context.insert(QStringLiteral("projectName"), currentProjectName_);
    context.insert(QStringLiteral("projectPath"), currentProjectPath_);
    context.insert(QStringLiteral("modelPath"), modelPath);
    context.insert(QStringLiteral("datasetPath"), currentDatasetPath_);
    context.insert(QStringLiteral("datasetFormat"), currentDatasetFormat_);
    context.insert(QStringLiteral("taskType"), currentTaskType());
    context.insert(QStringLiteral("trainingBackend"), trainingBackendCombo_ ? trainingBackendCombo_->currentData().toString() : QString());
    context.insert(QStringLiteral("modelPreset"), modelPresetCombo_ ? modelPresetCombo_->currentText().trimmed() : QString());
    context.insert(QStringLiteral("sourceTaskId"), selectedTaskId());
    if (repository_.isOpen() && !currentDatasetPath_.isEmpty()) {
        QString snapshotError;
        const aitrain::DatasetRecord dataset = repository_.datasetByRootPath(currentDatasetPath_, &snapshotError);
        const aitrain::DatasetSnapshotRecord snapshot = repository_.latestDatasetSnapshot(dataset.id, &snapshotError);
        if (snapshot.id > 0) {
            context.insert(QStringLiteral("datasetSnapshotId"), snapshot.id);
            context.insert(QStringLiteral("datasetSnapshotHash"), snapshot.contentHash);
            context.insert(QStringLiteral("datasetSnapshotManifest"), snapshot.manifestPath);
        }
    }
    QString error;
    if (!worker_.requestDeliveryReport(workerExecutablePath(), outputPath, context, &error, taskId)) {
        if (!taskId.isEmpty() && repository_.isOpen()) {
            QString taskError;
            repository_.updateTaskState(taskId, aitrain::TaskState::Failed, error, &taskError);
            currentTaskId_.clear();
            updateRecentTasks();
        }
        QMessageBox::critical(this, uiText("交付报告"), error);
        return;
    }
    workerPill_->setStatus(uiText("交付报告生成中"), StatusPill::Tone::Info);
}
