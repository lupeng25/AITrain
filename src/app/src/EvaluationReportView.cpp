#include "EvaluationReportView.h"

#include "InfoPanel.h"
#include "LanguageSupport.h"

#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QDir>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QSizePolicy>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

namespace {

QString uiText(const char* source)
{
    return aitrain_app::translateText("MainWindow", QString::fromUtf8(source));
}

QString formatNumber(double value, int precision = 4)
{
    return QString::number(value, 'f', precision);
}

QJsonObject readJsonObjectFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        return {};
    }
    return document.object();
}

QStringList readCsvRows(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return {};
    }
    QStringList rows;
    while (!file.atEnd()) {
        rows.append(QString::fromUtf8(file.readLine()).trimmed());
    }
    return rows;
}

QString taskTypeLabel(const QString& taskType)
{
    if (taskType == QStringLiteral("detection")) {
        return uiText("检测");
    }
    if (taskType == QStringLiteral("segmentation")) {
        return uiText("分割");
    }
    if (taskType == QStringLiteral("ocr_recognition")) {
        return uiText("OCR 识别");
    }
    if (taskType == QStringLiteral("ocr_detection")) {
        return uiText("OCR 检测");
    }
    if (taskType == QStringLiteral("ocr")) {
        return uiText("OCR 端到端");
    }
    return taskType.isEmpty() ? uiText("未选择") : taskType;
}

QString jsonValueSummary(const QJsonValue& value)
{
    if (value.isDouble()) {
        return formatNumber(value.toDouble());
    }
    if (value.isBool()) {
        return value.toBool() ? uiText("是") : uiText("否");
    }
    if (value.isString()) {
        return value.toString();
    }
    if (value.isObject() || value.isArray()) {
        return QString::fromUtf8(QJsonDocument(value.isObject() ? QJsonDocument(value.toObject()) : QJsonDocument(value.toArray()))
                                     .toJson(QJsonDocument::Compact));
    }
    return {};
}

} // namespace

EvaluationReportView::EvaluationReportView(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(900);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(12);

    statusLabel_ = new QLabel(uiText("请选择一个评估报告。"));
    statusLabel_->setObjectName(QStringLiteral("InlineStatus"));
    statusLabel_->setWordWrap(true);
    root->addWidget(statusLabel_);

    auto* summaryPanel = new InfoPanel(uiText("评估摘要"));
    summaryLabel_ = new QLabel(uiText("暂无评估数据。"));
    summaryLabel_->setObjectName(QStringLiteral("MutedText"));
    summaryLabel_->setWordWrap(true);
    summaryPanel->bodyLayout()->addWidget(summaryLabel_);
    root->addWidget(summaryPanel);

    auto* upperSplitter = new QSplitter(Qt::Horizontal);

    auto* metricsPanel = new InfoPanel(uiText("关键指标"));
    metricsTable_ = new QTableWidget(0, 2);
    metricsTable_->setHorizontalHeaderLabels(QStringList() << uiText("指标") << uiText("值"));
    configureTable(metricsTable_);
    metricsTable_->setMinimumHeight(180);
    metricsTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    metricsPanel->bodyLayout()->addWidget(metricsTable_);
    upperSplitter->addWidget(metricsPanel);

    auto* perClassPanel = new InfoPanel(uiText("分类别指标"));
    perClassTable_ = new QTableWidget(0, 8);
    perClassTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("类别")
        << QStringLiteral("GT")
        << QStringLiteral("TP")
        << QStringLiteral("FP")
        << QStringLiteral("FN")
        << uiText("Precision")
        << uiText("Recall")
        << uiText("质量"));
    configureTable(perClassTable_);
    perClassTable_->setMinimumHeight(180);
    perClassTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    perClassPanel->bodyLayout()->addWidget(perClassTable_);
    upperSplitter->addWidget(perClassPanel);
    upperSplitter->setStretchFactor(0, 1);
    upperSplitter->setStretchFactor(1, 2);
    upperSplitter->setChildrenCollapsible(false);
    upperSplitter->setSizes(QList<int>() << 320 << 560);
    root->addWidget(upperSplitter, 1);

    auto* lowerSplitter = new QSplitter(Qt::Horizontal);

    auto* confusionPanel = new InfoPanel(uiText("混淆矩阵"));
    confusionTable_ = new QTableWidget(0, 0);
    configureTable(confusionTable_, false);
    confusionTable_->setMinimumHeight(240);
    confusionTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    confusionPanel->bodyLayout()->addWidget(confusionTable_);
    lowerSplitter->addWidget(confusionPanel);

    auto* errorPanel = new InfoPanel(uiText("错误样本"));
    errorTable_ = new QTableWidget(0, 5);
    errorTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("原因")
        << uiText("样本")
        << uiText("目标")
        << uiText("预测")
        << uiText("补充信息"));
    configureTable(errorTable_);
    errorTable_->setMinimumHeight(180);
    errorTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(errorTable_, &QTableWidget::itemSelectionChanged, this, &EvaluationReportView::updateErrorPreview);

    auto* previewSplitter = new QSplitter(Qt::Vertical);
    overlayLabel_ = new QLabel(uiText("暂无 overlay 预览"));
    overlayLabel_->setObjectName(QStringLiteral("MutedText"));
    overlayLabel_->setAlignment(Qt::AlignCenter);
    overlayLabel_->setMinimumHeight(180);
    overlayLabel_->setFrameShape(QFrame::StyledPanel);
    detailText_ = new QPlainTextEdit;
    detailText_->setReadOnly(true);
    detailText_->setPlainText(uiText("选择一个错误样本后显示详情。"));
    previewSplitter->addWidget(overlayLabel_);
    previewSplitter->addWidget(detailText_);
    previewSplitter->setStretchFactor(0, 2);
    previewSplitter->setStretchFactor(1, 1);
    previewSplitter->setChildrenCollapsible(false);
    previewSplitter->setSizes(QList<int>() << 220 << 120);

    errorPanel->bodyLayout()->addWidget(errorTable_, 2);
    errorPanel->bodyLayout()->addWidget(previewSplitter, 2);
    lowerSplitter->addWidget(errorPanel);
    lowerSplitter->setStretchFactor(0, 1);
    lowerSplitter->setStretchFactor(1, 2);
    lowerSplitter->setChildrenCollapsible(false);
    lowerSplitter->setSizes(QList<int>() << 360 << 620);
    root->addWidget(lowerSplitter, 2);

    clear();
}

void EvaluationReportView::clear()
{
    currentReportPath_.clear();
    rowOverlayPaths_.clear();
    rowDetailTexts_.clear();
    statusLabel_->setText(uiText("请选择一个评估报告。"));
    summaryLabel_->setText(uiText("暂无评估数据。"));
    metricsTable_->setRowCount(0);
    perClassTable_->setRowCount(0);
    confusionTable_->clear();
    confusionTable_->setRowCount(0);
    confusionTable_->setColumnCount(0);
    errorTable_->setRowCount(0);
    overlayLabel_->clear();
    overlayLabel_->setText(uiText("暂无 overlay 预览"));
    detailText_->setPlainText(uiText("选择一个错误样本后显示详情。"));
}

bool EvaluationReportView::loadReport(const QString& reportPath)
{
    clear();
    currentReportPath_ = reportPath;
    const QJsonObject report = readJsonObjectFile(reportPath);
    if (report.isEmpty()) {
        showEmptyState(uiText("评估报告无法读取或 JSON 无法解析。"));
        return false;
    }

    const QString taskType = taskTypeLabel(report.value(QStringLiteral("taskType")).toString());
    const QString runtime = report.value(QStringLiteral("runtime")).toString();
    const bool scaffold = report.value(QStringLiteral("scaffold")).toBool();
    QString status = report.value(QStringLiteral("status")).toString();
    if (status.isEmpty()) {
        status = report.value(QStringLiteral("ok")).toBool(true) ? QStringLiteral("ok") : QStringLiteral("failed");
    }
    statusLabel_->setText(uiText("任务类型：%1 | 运行时：%2 | 状态：%3 | 真实评估：%4")
        .arg(taskType)
        .arg(runtime.isEmpty() ? uiText("未记录") : runtime)
        .arg(status)
        .arg(scaffold ? uiText("否，scaffold") : uiText("是")));

    QStringList summaryLines;
    summaryLines << uiText("报告路径：%1").arg(QDir::toNativeSeparators(reportPath));
    if (report.contains(QStringLiteral("sampleCount"))) {
        summaryLines << uiText("样本数：%1").arg(report.value(QStringLiteral("sampleCount")).toInt());
    }
    if (report.contains(QStringLiteral("split"))) {
        summaryLines << uiText("数据划分：%1").arg(report.value(QStringLiteral("split")).toString());
    }
    const QJsonValue limitations = report.value(QStringLiteral("limitations"));
    if (!limitations.isUndefined() && !limitations.isNull()) {
        summaryLines << uiText("限制说明：%1").arg(jsonValueSummary(limitations));
    }
    if (report.contains(QStringLiteral("message"))) {
        summaryLines << uiText("附加说明：%1").arg(report.value(QStringLiteral("message")).toString());
    }
    summaryLabel_->setText(summaryLines.join(QStringLiteral("\n")));

    populateMetrics(report);
    populatePerClass(report);
    populateConfusion(report);
    populateErrors(report);
    return true;
}

void EvaluationReportView::updateErrorPreview()
{
    if (errorTable_->selectedItems().isEmpty()) {
        overlayLabel_->clear();
        overlayLabel_->setText(uiText("暂无 overlay 预览"));
        detailText_->setPlainText(uiText("选择一个错误样本后显示详情。"));
        return;
    }

    const int row = errorTable_->selectedItems().first()->row();
    showOverlayImage(rowOverlayPaths_.value(row));
    detailText_->setPlainText(rowDetailTexts_.value(row, uiText("该样本没有更多详情。")));
}

void EvaluationReportView::configureTable(QTableWidget* table, bool stretchLast) const
{
    table->setAlternatingRowColors(true);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setWordWrap(false);
    table->verticalHeader()->setVisible(false);
    table->verticalHeader()->setDefaultSectionSize(28);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->horizontalHeader()->setStretchLastSection(stretchLast);
    table->horizontalHeader()->setMinimumSectionSize(72);
    table->setShowGrid(false);
}

void EvaluationReportView::populateMetrics(const QJsonObject& report)
{
    const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
    metricsTable_->setRowCount(0);
    if (metrics.isEmpty()) {
        metricsTable_->insertRow(0);
        metricsTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无指标")));
        metricsTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        return;
    }

    for (auto it = metrics.constBegin(); it != metrics.constEnd(); ++it) {
        const int row = metricsTable_->rowCount();
        metricsTable_->insertRow(row);
        metricsTable_->setItem(row, 0, new QTableWidgetItem(it.key()));
        metricsTable_->setItem(row, 1, new QTableWidgetItem(jsonValueSummary(it.value())));
    }
    metricsTable_->resizeColumnsToContents();
}

void EvaluationReportView::populatePerClass(const QJsonObject& report)
{
    perClassTable_->setRowCount(0);
    const QJsonArray perClass = report.value(QStringLiteral("perClass")).toArray();
    if (perClass.isEmpty()) {
        perClassTable_->insertRow(0);
        perClassTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无分类别指标")));
        for (int column = 1; column < perClassTable_->columnCount(); ++column) {
            perClassTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        return;
    }

    for (const QJsonValue& value : perClass) {
        const QJsonObject item = value.toObject();
        const int row = perClassTable_->rowCount();
        perClassTable_->insertRow(row);
        perClassTable_->setItem(row, 0, new QTableWidgetItem(item.value(QStringLiteral("className")).toString(
            QStringLiteral("class_%1").arg(item.value(QStringLiteral("classId")).toInt()))));
        perClassTable_->setItem(row, 1, new QTableWidgetItem(QString::number(item.value(QStringLiteral("gt")).toInt())));
        perClassTable_->setItem(row, 2, new QTableWidgetItem(QString::number(item.value(QStringLiteral("tp")).toInt())));
        perClassTable_->setItem(row, 3, new QTableWidgetItem(QString::number(item.value(QStringLiteral("fp")).toInt())));
        perClassTable_->setItem(row, 4, new QTableWidgetItem(QString::number(item.value(QStringLiteral("fn")).toInt())));
        perClassTable_->setItem(row, 5, new QTableWidgetItem(formatNumber(item.value(QStringLiteral("precision")).toDouble())));
        perClassTable_->setItem(row, 6, new QTableWidgetItem(formatNumber(item.value(QStringLiteral("recall")).toDouble())));
        QString quality = item.contains(QStringLiteral("ap50"))
            ? QStringLiteral("AP50=%1").arg(formatNumber(item.value(QStringLiteral("ap50")).toDouble()))
            : QStringLiteral("maskAP50=%1").arg(formatNumber(item.value(QStringLiteral("maskAP50")).toDouble()));
        if (item.contains(QStringLiteral("maskIoU"))) {
            quality.append(QStringLiteral(" | maskIoU=%1").arg(formatNumber(item.value(QStringLiteral("maskIoU")).toDouble())));
        }
        perClassTable_->setItem(row, 7, new QTableWidgetItem(quality));
    }
    perClassTable_->resizeColumnsToContents();
}

void EvaluationReportView::populateConfusion(const QJsonObject& report)
{
    confusionTable_->clear();
    const QString csvPath = report.value(QStringLiteral("confusionMatrixPath")).toString();
    const QStringList rows = readCsvRows(csvPath);
    if (rows.isEmpty()) {
        confusionTable_->setRowCount(1);
        confusionTable_->setColumnCount(1);
        confusionTable_->setHorizontalHeaderLabels(QStringList() << uiText("矩阵"));
        confusionTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无混淆矩阵")));
        return;
    }

    const QStringList header = rows.first().split(QLatin1Char(','));
    confusionTable_->setColumnCount(header.size());
    confusionTable_->setHorizontalHeaderLabels(header);
    confusionTable_->setRowCount(qMax(0, rows.size() - 1));

    QStringList verticalLabels;
    for (int row = 1; row < rows.size(); ++row) {
        const QStringList columns = rows.at(row).split(QLatin1Char(','));
        verticalLabels.append(columns.isEmpty() ? QString::number(row) : columns.first());
        for (int column = 0; column < header.size() && column < columns.size(); ++column) {
            confusionTable_->setItem(row - 1, column, new QTableWidgetItem(columns.at(column)));
        }
    }
    confusionTable_->setVerticalHeaderLabels(verticalLabels);
    confusionTable_->resizeColumnsToContents();
}

void EvaluationReportView::populateErrors(const QJsonObject& report)
{
    errorTable_->setRowCount(0);
    rowOverlayPaths_.clear();
    rowDetailTexts_.clear();

    const QJsonArray samples = report.value(QStringLiteral("samples")).toArray();
    QHash<QString, QString> overlayByImage;
    for (const QJsonValue& value : samples) {
        const QJsonObject sample = value.toObject();
        const QString imagePath = sample.value(QStringLiteral("imagePath")).toString();
        const QString overlayPath = sample.value(QStringLiteral("overlayPath")).toString();
        if (!imagePath.isEmpty() && !overlayPath.isEmpty()) {
            overlayByImage.insert(imagePath, overlayPath);
        }
    }

    auto appendErrorRow = [this, &overlayByImage](const QJsonObject& item) {
        const int row = errorTable_->rowCount();
        errorTable_->insertRow(row);

        const QString imagePath = item.value(QStringLiteral("imagePath")).toString();
        const QString predictionText = item.value(QStringLiteral("prediction")).isObject()
            ? QString::fromUtf8(QJsonDocument(item.value(QStringLiteral("prediction")).toObject()).toJson(QJsonDocument::Compact))
            : item.value(QStringLiteral("prediction")).toString();
        const QString groundTruthText = item.value(QStringLiteral("groundTruth")).isObject()
            ? QString::fromUtf8(QJsonDocument(item.value(QStringLiteral("groundTruth")).toObject()).toJson(QJsonDocument::Compact))
            : item.value(QStringLiteral("groundTruth")).toString();
        QString detail = uiText("图片：%1").arg(QDir::toNativeSeparators(imagePath));
        const QString labelPath = item.value(QStringLiteral("labelPath")).toString();
        if (!labelPath.isEmpty()) {
            detail.append(uiText("\n标签：%1").arg(QDir::toNativeSeparators(labelPath)));
        }
        for (auto it = item.constBegin(); it != item.constEnd(); ++it) {
            if (it.key() == QStringLiteral("imagePath") || it.key() == QStringLiteral("labelPath")) {
                continue;
            }
            detail.append(QStringLiteral("\n%1: %2").arg(it.key(), jsonValueSummary(it.value())));
        }

        const QString reason = item.value(QStringLiteral("reason")).toString(
            item.contains(QStringLiteral("matched")) ? uiText("识别错误") : uiText("错误样本"));
        const QString extra = item.contains(QStringLiteral("matchedIou"))
            ? QStringLiteral("IoU=%1").arg(formatNumber(item.value(QStringLiteral("matchedIou")).toDouble()))
            : item.contains(QStringLiteral("confidence"))
                ? QStringLiteral("conf=%1").arg(formatNumber(item.value(QStringLiteral("confidence")).toDouble()))
                : item.contains(QStringLiteral("editDistance"))
                    ? QStringLiteral("edit=%1").arg(item.value(QStringLiteral("editDistance")).toInt())
                    : QString();

        errorTable_->setItem(row, 0, new QTableWidgetItem(reason));
        errorTable_->setItem(row, 1, new QTableWidgetItem(QFileInfo(imagePath).fileName()));
        errorTable_->setItem(row, 2, new QTableWidgetItem(groundTruthText));
        errorTable_->setItem(row, 3, new QTableWidgetItem(predictionText));
        errorTable_->setItem(row, 4, new QTableWidgetItem(extra));

        rowOverlayPaths_.insert(row, item.value(QStringLiteral("overlayPath")).toString(overlayByImage.value(imagePath)));
        rowDetailTexts_.insert(row, detail);
    };

    const QJsonArray errors = report.value(QStringLiteral("errorSamples")).toArray();
    for (const QJsonValue& value : errors) {
        appendErrorRow(value.toObject());
    }
    const QJsonArray lowConfidence = report.value(QStringLiteral("lowConfidenceSamples")).toArray();
    for (const QJsonValue& value : lowConfidence) {
        QJsonObject item = value.toObject();
        if (!item.contains(QStringLiteral("reason"))) {
            item.insert(QStringLiteral("reason"), uiText("低置信样本"));
        }
        appendErrorRow(item);
    }

    if (errorTable_->rowCount() == 0) {
        errorTable_->insertRow(0);
        errorTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无错误样本")));
        for (int column = 1; column < errorTable_->columnCount(); ++column) {
            errorTable_->setItem(0, column, new QTableWidgetItem(QString()));
        }
        overlayLabel_->setText(uiText("暂无 overlay 预览"));
        detailText_->setPlainText(uiText("该报告没有记录错误样本。"));
        return;
    }

    errorTable_->resizeColumnsToContents();
    errorTable_->selectRow(0);
}

void EvaluationReportView::showOverlayImage(const QString& overlayPath)
{
    overlayLabel_->clear();
    if (overlayPath.isEmpty()) {
        overlayLabel_->setText(uiText("该样本没有 overlay。"));
        return;
    }
    const QPixmap image(overlayPath);
    if (image.isNull()) {
        overlayLabel_->setText(uiText("overlay 无法读取：%1").arg(QDir::toNativeSeparators(overlayPath)));
        return;
    }
    overlayLabel_->setPixmap(image.scaled(
        overlayLabel_->size().boundedTo(QSize(520, 320)),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation));
}

void EvaluationReportView::showEmptyState(const QString& text)
{
    summaryLabel_->setText(text);
    metricsTable_->setRowCount(0);
    perClassTable_->setRowCount(0);
    confusionTable_->clear();
    confusionTable_->setRowCount(0);
    confusionTable_->setColumnCount(0);
    errorTable_->setRowCount(0);
    overlayLabel_->setText(uiText("暂无 overlay 预览"));
    detailText_->setPlainText(text);
}
