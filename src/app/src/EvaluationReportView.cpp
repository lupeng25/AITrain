#include "EvaluationReportView.h"

#include "InfoPanel.h"
#include "LanguageSupport.h"

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
#include <QSignalBlocker>
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

bool isPreviewImagePath(const QString& path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    return suffix == QStringLiteral("png")
        || suffix == QStringLiteral("jpg")
        || suffix == QStringLiteral("jpeg")
        || suffix == QStringLiteral("bmp")
        || suffix == QStringLiteral("webp");
}

QString artifactKindLabel(const QString& kind)
{
    if (kind == QStringLiteral("official_metrics")) {
        return uiText("官方指标");
    }
    if (kind == QStringLiteral("official_log")) {
        return uiText("官方日志");
    }
    if (kind == QStringLiteral("official_plot")) {
        return uiText("官方图表");
    }
    if (kind == QStringLiteral("official_predictions")) {
        return uiText("官方预测");
    }
    return kind.isEmpty() ? uiText("官方产物") : kind;
}

bool isUltralyticsOfficialYoloReport(const QJsonObject& report)
{
    return report.value(QStringLiteral("evaluationSource")).toString() == QStringLiteral("ultralytics_official_val")
        || report.value(QStringLiteral("runtime")).toString() == QStringLiteral("ultralytics_official_val");
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
    metricsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    metricsTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    metricsTable_->setMinimumHeight(180);
    metricsTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    metricsPanel->bodyLayout()->addWidget(metricsTable_);
    upperSplitter->addWidget(metricsPanel);

    auto* perClassPanel = new InfoPanel(uiText("分类别指标"));
    perClassTable_ = new QTableWidget(0, 8);
    perClassTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("类别")
        << uiText("来源")
        << QStringLiteral("GT")
        << QStringLiteral("TP/FP")
        << QStringLiteral("FN")
        << uiText("Precision")
        << uiText("Recall")
        << uiText("质量"));
    configureTable(perClassTable_);
    perClassTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    perClassTable_->horizontalHeader()->setStretchLastSection(false);
    perClassTable_->horizontalHeader()->setMinimumSectionSize(52);
    perClassTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < perClassTable_->columnCount(); ++column) {
        perClassTable_->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    }
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

    auto* officialPanel = new InfoPanel(uiText("官方产物"));
    officialArtifactsTable_ = new QTableWidget(0, 3);
    officialArtifactsTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("类型")
        << uiText("名称")
        << uiText("Artifact 相对项"));
    configureTable(officialArtifactsTable_);
    officialArtifactsTable_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    officialArtifactsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    officialArtifactsTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    officialArtifactsTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    officialArtifactsTable_->setMinimumHeight(240);
    officialArtifactsTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(officialArtifactsTable_, &QTableWidget::itemSelectionChanged, this, &EvaluationReportView::updateArtifactPreview);
    officialPanel->bodyLayout()->addWidget(officialArtifactsTable_);
    lowerSplitter->addWidget(officialPanel);

    auto* samplePanel = new InfoPanel(uiText("样本与预览"));
    sampleTable_ = new QTableWidget(0, 5);
    sampleTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("类型")
        << uiText("样本")
        << uiText("目标")
        << uiText("预测")
        << uiText("补充信息"));
    configureTable(sampleTable_);
    sampleTable_->setWordWrap(true);
    sampleTable_->verticalHeader()->setDefaultSectionSize(40);
    sampleTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    sampleTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    sampleTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    sampleTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    sampleTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    sampleTable_->setMinimumHeight(180);
    sampleTable_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    connect(sampleTable_, &QTableWidget::itemSelectionChanged, this, &EvaluationReportView::updateSamplePreview);

    auto* previewSplitter = new QSplitter(Qt::Vertical);
    previewLabel_ = new QLabel(uiText("选择官方图表或样本 overlay 后显示预览。"));
    previewLabel_->setObjectName(QStringLiteral("MutedText"));
    previewLabel_->setAlignment(Qt::AlignCenter);
    previewLabel_->setMinimumHeight(180);
    previewLabel_->setFrameShape(QFrame::StyledPanel);
    detailText_ = new QPlainTextEdit;
    detailText_->setReadOnly(true);
    detailText_->setPlainText(uiText("选择官方产物或样本后显示详情。"));
    previewSplitter->addWidget(previewLabel_);
    previewSplitter->addWidget(detailText_);
    previewSplitter->setStretchFactor(0, 2);
    previewSplitter->setStretchFactor(1, 1);
    previewSplitter->setChildrenCollapsible(false);
    previewSplitter->setSizes(QList<int>() << 220 << 120);

    samplePanel->bodyLayout()->addWidget(sampleTable_, 2);
    samplePanel->bodyLayout()->addWidget(previewSplitter, 2);
    lowerSplitter->addWidget(samplePanel);
    lowerSplitter->setStretchFactor(0, 1);
    lowerSplitter->setStretchFactor(1, 2);
    lowerSplitter->setChildrenCollapsible(false);
    lowerSplitter->setSizes(QList<int>() << 420 << 620);
    root->addWidget(lowerSplitter, 2);

    clear();
}

void EvaluationReportView::clear()
{
    const QSignalBlocker metricsBlocker(metricsTable_);
    const QSignalBlocker perClassBlocker(perClassTable_);
    const QSignalBlocker officialArtifactsBlocker(officialArtifactsTable_);
    const QSignalBlocker samplesBlocker(sampleTable_);
    currentReportPath_.clear();
    artifactPreviewPaths_.clear();
    artifactDetailTexts_.clear();
    samplePreviewPaths_.clear();
    sampleDetailTexts_.clear();
    statusLabel_->setText(uiText("请选择一个评估报告。"));
    summaryLabel_->setText(uiText("暂无评估数据。"));
    metricsTable_->setRowCount(0);
    perClassTable_->setRowCount(0);
    officialArtifactsTable_->setRowCount(0);
    sampleTable_->setRowCount(0);
    previewLabel_->clear();
    previewLabel_->setText(uiText("选择官方图表或样本 overlay 后显示预览。"));
    detailText_->setPlainText(uiText("选择官方产物或样本后显示详情。"));
}

bool EvaluationReportView::loadReportData(const QByteArray& data, const QString& relativePath)
{
    clear();
    currentReportPath_.clear();
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        showEmptyState(uiText("评估报告无法读取或 JSON 无法解析。"));
        return false;
    }
    const bool loaded = loadReportObject(document.object());
    if (loaded && !relativePath.isEmpty()) {
        statusLabel_->setText(statusLabel_->text() + uiText(" | Artifact：%1").arg(relativePath));
    }
    return loaded;
}

bool EvaluationReportView::loadReportObject(const QJsonObject& report)
{

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
    if (report.contains(QStringLiteral("sampleCount"))) {
        summaryLines << uiText("样本数：%1").arg(report.value(QStringLiteral("sampleCount")).toInt());
    }
    if (report.contains(QStringLiteral("split"))) {
        summaryLines << uiText("数据划分：%1").arg(report.value(QStringLiteral("split")).toString());
    }
    const QString evaluationSource = report.value(QStringLiteral("evaluationSource")).toString();
    if (!evaluationSource.isEmpty()) {
        summaryLines << uiText("评估来源：%1").arg(evaluationSource);
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
    populateSamples(report);
    populateOfficialArtifacts(report);
    return true;
}

QString EvaluationReportView::resolveArtifactPath(const QString& declaredPath) const
{
    if (declaredPath.trimmed().isEmpty() || currentReportPath_.isEmpty()) {
        return QString();
    }

    const QString reportRoot = QFileInfo(currentReportPath_).canonicalPath();
    if (reportRoot.isEmpty()) {
        return QString();
    }

    const QFileInfo declaredInfo(declaredPath);
    const QString candidatePath = declaredInfo.isAbsolute()
        ? declaredInfo.absoluteFilePath()
        : QDir(reportRoot).absoluteFilePath(QDir::cleanPath(declaredPath));
    const QFileInfo candidateInfo(candidatePath);
    if (!candidateInfo.exists() || !candidateInfo.isFile()) {
        return QString();
    }

    const QString canonicalPath = candidateInfo.canonicalFilePath();
    const QString rootPrefix = reportRoot.endsWith(QDir::separator())
        ? reportRoot
        : reportRoot + QDir::separator();
    if (canonicalPath.compare(reportRoot, Qt::CaseInsensitive) != 0
        && !canonicalPath.startsWith(rootPrefix, Qt::CaseInsensitive)) {
        return QString();
    }
    return canonicalPath;
}

void EvaluationReportView::updateArtifactPreview()
{
    if (officialArtifactsTable_->selectedItems().isEmpty()) {
        previewLabel_->clear();
        previewLabel_->setText(uiText("选择官方图表或样本 overlay 后显示预览。"));
        detailText_->setPlainText(uiText("选择官方产物或样本后显示详情。"));
        return;
    }

    const int row = officialArtifactsTable_->selectedItems().first()->row();
    QSignalBlocker sampleBlocker(sampleTable_);
    sampleTable_->clearSelection();
    showPreviewImage(artifactPreviewPaths_.value(row));
    detailText_->setPlainText(artifactDetailTexts_.value(row, uiText("该官方产物没有更多详情。")));
}

void EvaluationReportView::updateSamplePreview()
{
    if (sampleTable_->selectedItems().isEmpty()) {
        previewLabel_->clear();
        previewLabel_->setText(uiText("选择官方图表或样本 overlay 后显示预览。"));
        detailText_->setPlainText(uiText("选择官方产物或样本后显示详情。"));
        return;
    }

    const int row = sampleTable_->selectedItems().first()->row();
    QSignalBlocker artifactBlocker(officialArtifactsTable_);
    officialArtifactsTable_->clearSelection();
    showPreviewImage(samplePreviewPaths_.value(row));
    detailText_->setPlainText(sampleDetailTexts_.value(row, uiText("该样本没有更多详情。")));
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
    metricsTable_->resizeColumnToContents(1);
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
        const bool official = item.value(QStringLiteral("official")).toBool(false);
        perClassTable_->setItem(row, 1, new QTableWidgetItem(official ? QStringLiteral("Ultralytics") : QStringLiteral("AITrain")));
        perClassTable_->setItem(row, 2, new QTableWidgetItem(item.contains(QStringLiteral("gt")) ? QString::number(item.value(QStringLiteral("gt")).toInt()) : QStringLiteral("-")));
        const QString tpFpText = item.contains(QStringLiteral("tp")) || item.contains(QStringLiteral("fp"))
            ? QStringLiteral("%1/%2").arg(item.value(QStringLiteral("tp")).toInt()).arg(item.value(QStringLiteral("fp")).toInt())
            : QStringLiteral("-");
        perClassTable_->setItem(row, 3, new QTableWidgetItem(tpFpText));
        perClassTable_->setItem(row, 4, new QTableWidgetItem(item.contains(QStringLiteral("fn")) ? QString::number(item.value(QStringLiteral("fn")).toInt()) : QStringLiteral("-")));
        perClassTable_->setItem(row, 5, new QTableWidgetItem(item.contains(QStringLiteral("precision")) ? formatNumber(item.value(QStringLiteral("precision")).toDouble()) : QStringLiteral("-")));
        perClassTable_->setItem(row, 6, new QTableWidgetItem(item.contains(QStringLiteral("recall")) ? formatNumber(item.value(QStringLiteral("recall")).toDouble()) : QStringLiteral("-")));
        QString quality = item.contains(QStringLiteral("mAP50_95"))
            ? QStringLiteral("mAP50-95=%1").arg(formatNumber(item.value(QStringLiteral("mAP50_95")).toDouble()))
            : item.contains(QStringLiteral("ap50"))
            ? QStringLiteral("AP50=%1").arg(formatNumber(item.value(QStringLiteral("ap50")).toDouble()))
            : QStringLiteral("maskAP50=%1").arg(formatNumber(item.value(QStringLiteral("maskAP50")).toDouble()));
        if (item.contains(QStringLiteral("maskMap50_95"))) {
            quality.append(QStringLiteral(" | mask mAP50-95=%1").arg(formatNumber(item.value(QStringLiteral("maskMap50_95")).toDouble())));
        }
        if (item.contains(QStringLiteral("maskIoU"))) {
            quality.append(QStringLiteral(" | maskIoU=%1").arg(formatNumber(item.value(QStringLiteral("maskIoU")).toDouble())));
        }
        perClassTable_->setItem(row, 7, new QTableWidgetItem(quality));
    }
    perClassTable_->resizeColumnToContents(1);
    perClassTable_->resizeColumnToContents(2);
    perClassTable_->resizeColumnToContents(3);
    perClassTable_->resizeColumnToContents(4);
    perClassTable_->resizeColumnToContents(5);
    perClassTable_->resizeColumnToContents(6);
    perClassTable_->resizeColumnToContents(7);
}

void EvaluationReportView::populateOfficialArtifacts(const QJsonObject& report)
{
    officialArtifactsTable_->setRowCount(0);
    artifactPreviewPaths_.clear();
    artifactDetailTexts_.clear();
    QStringList seenPaths;

    auto appendArtifact = [this, &seenPaths](const QString& kind, const QString& name, const QString& path) {
        const QString declaredPath = path.trimmed();
        if (declaredPath.isEmpty() || seenPaths.contains(declaredPath)) {
            return;
        }
        seenPaths.append(declaredPath);
        const QFileInfo info(declaredPath);
        const QString safePath = resolveArtifactPath(declaredPath);
        const QString artifactName = name.isEmpty() ? info.fileName() : name;
        const QString displayPath = safePath.isEmpty()
            ? uiText("外部路径已隐藏")
            : QDir::cleanPath(QDir(QFileInfo(currentReportPath_).canonicalPath()).relativeFilePath(safePath));
        const int row = officialArtifactsTable_->rowCount();
        officialArtifactsTable_->insertRow(row);
        officialArtifactsTable_->setItem(row, 0, new QTableWidgetItem(artifactKindLabel(kind)));
        officialArtifactsTable_->setItem(row, 1, new QTableWidgetItem(artifactName));
        officialArtifactsTable_->setItem(row, 2, new QTableWidgetItem(displayPath));
        if (!safePath.isEmpty() && isPreviewImagePath(safePath)) {
            artifactPreviewPaths_.insert(row, safePath);
        }
        artifactDetailTexts_.insert(row,
            safePath.isEmpty()
                ? uiText("类型：%1\n名称：%2\n该记录不属于当前 committed Artifact，已禁止外部路径预览。请从任务产物列表查看已提交文件。")
                    .arg(artifactKindLabel(kind), artifactName)
                : uiText("类型：%1\n名称：%2\nArtifact 相对项：%3")
                    .arg(artifactKindLabel(kind), artifactName, displayPath));
    };

    appendArtifact(QStringLiteral("official_metrics"), QStringLiteral("ultralytics_official_metrics.json"), report.value(QStringLiteral("officialMetricsPath")).toString());
    appendArtifact(QStringLiteral("official_log"), QStringLiteral("ultralytics_official_val.log"), report.value(QStringLiteral("officialLogPath")).toString());
    appendArtifact(QStringLiteral("evaluation_summary"), QStringLiteral("evaluation_summary.md"), report.value(QStringLiteral("evaluationSummaryPath")).toString());

    const QJsonArray officialArtifacts = report.value(QStringLiteral("officialArtifacts")).toArray();
    for (const QJsonValue& value : officialArtifacts) {
        const QJsonObject artifact = value.toObject();
        appendArtifact(
            artifact.value(QStringLiteral("kind")).toString(QStringLiteral("official_artifact")),
            artifact.value(QStringLiteral("name")).toString(),
            artifact.value(QStringLiteral("path")).toString());
    }

    if (officialArtifactsTable_->rowCount() == 0) {
        officialArtifactsTable_->insertRow(0);
        officialArtifactsTable_->setItem(0, 0, new QTableWidgetItem(uiText("暂无官方产物")));
        officialArtifactsTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        officialArtifactsTable_->setItem(0, 2, new QTableWidgetItem(uiText("该报告没有记录 officialArtifacts 或官方路径字段。")));
        detailText_->setPlainText(uiText("该报告没有记录官方产物。旧历史报告仍可通过任务产物列表查看原始文件。"));
        return;
    }

    officialArtifactsTable_->resizeColumnToContents(0);
    officialArtifactsTable_->resizeColumnToContents(1);
    officialArtifactsTable_->selectRow(0);
}

void EvaluationReportView::populateSamples(const QJsonObject& report)
{
    sampleTable_->setRowCount(0);
    samplePreviewPaths_.clear();
    sampleDetailTexts_.clear();

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

    auto appendSampleRow = [this, &overlayByImage](const QJsonObject& item, const QString& fallbackType) {
        const int row = sampleTable_->rowCount();
        sampleTable_->insertRow(row);

        const QString imagePath = item.value(QStringLiteral("imagePath")).toString();
        const QString safeImagePath = resolveArtifactPath(imagePath);
        const QString predictionText = item.value(QStringLiteral("prediction")).isObject()
            ? QString::fromUtf8(QJsonDocument(item.value(QStringLiteral("prediction")).toObject()).toJson(QJsonDocument::Compact))
            : item.value(QStringLiteral("prediction")).toString();
        const QString groundTruthText = item.value(QStringLiteral("groundTruth")).isObject()
            ? QString::fromUtf8(QJsonDocument(item.value(QStringLiteral("groundTruth")).toObject()).toJson(QJsonDocument::Compact))
            : item.value(QStringLiteral("groundTruth")).toString();
        QString detail = imagePath.isEmpty()
            ? uiText("未记录图片路径")
            : safeImagePath.isEmpty()
                ? uiText("图片路径不属于当前 committed Artifact，已隐藏。")
                : uiText("图片（Artifact 内）：%1").arg(QDir::cleanPath(QDir(QFileInfo(currentReportPath_).canonicalPath()).relativeFilePath(safeImagePath)));
        const QString labelPath = item.value(QStringLiteral("labelPath")).toString();
        if (!labelPath.isEmpty()) {
            const QString safeLabelPath = resolveArtifactPath(labelPath);
            detail.append(safeLabelPath.isEmpty()
                ? uiText("\n标签路径不属于当前 committed Artifact，已隐藏。")
                : uiText("\n标签（Artifact 内）：%1").arg(QDir::cleanPath(QDir(QFileInfo(currentReportPath_).canonicalPath()).relativeFilePath(safeLabelPath))));
        }
        for (auto it = item.constBegin(); it != item.constEnd(); ++it) {
            if (it.key() == QStringLiteral("imagePath") || it.key() == QStringLiteral("labelPath")) {
                continue;
            }
            const QString key = it.key().toLower();
            if (key.contains(QStringLiteral("path")) || key.endsWith(QStringLiteral("dir"))) {
                continue;
            }
            detail.append(QStringLiteral("\n%1: %2").arg(it.key(), jsonValueSummary(it.value())));
        }

        const QString reason = item.value(QStringLiteral("reason")).toString(fallbackType);
        const QString extra = item.contains(QStringLiteral("matchedIou"))
            ? QStringLiteral("IoU=%1").arg(formatNumber(item.value(QStringLiteral("matchedIou")).toDouble()))
            : item.contains(QStringLiteral("confidence"))
                ? QStringLiteral("conf=%1").arg(formatNumber(item.value(QStringLiteral("confidence")).toDouble()))
                : item.contains(QStringLiteral("editDistance"))
                    ? QStringLiteral("edit=%1").arg(item.value(QStringLiteral("editDistance")).toInt())
                    : QString();

        sampleTable_->setItem(row, 0, new QTableWidgetItem(reason));
        sampleTable_->setItem(row, 1, new QTableWidgetItem(imagePath.isEmpty()
            ? QStringLiteral("-")
            : safeImagePath.isEmpty() ? uiText("外部样本（已隐藏）") : QFileInfo(safeImagePath).fileName()));
        sampleTable_->setItem(row, 2, new QTableWidgetItem(groundTruthText));
        sampleTable_->setItem(row, 3, new QTableWidgetItem(predictionText));
        sampleTable_->setItem(row, 4, new QTableWidgetItem(extra));

        const QString declaredOverlayPath = item.value(QStringLiteral("overlayPath")).toString(overlayByImage.value(imagePath));
        samplePreviewPaths_.insert(row, resolveArtifactPath(declaredOverlayPath));
        sampleDetailTexts_.insert(row, detail);
    };

    const QJsonArray errors = report.value(QStringLiteral("errorSamples")).toArray();
    for (const QJsonValue& value : errors) {
        appendSampleRow(value.toObject(), uiText("错误样本"));
    }
    const QJsonArray lowConfidence = report.value(QStringLiteral("lowConfidenceSamples")).toArray();
    for (const QJsonValue& value : lowConfidence) {
        appendSampleRow(value.toObject(), uiText("低置信样本"));
    }
    if (sampleTable_->rowCount() == 0) {
        for (const QJsonValue& value : samples) {
            appendSampleRow(value.toObject(), uiText("历史样本"));
        }
    }

    if (sampleTable_->rowCount() == 0) {
        const bool officialYolo = isUltralyticsOfficialYoloReport(report);
        sampleTable_->insertRow(0);
        sampleTable_->setItem(0, 0, new QTableWidgetItem(officialYolo ? uiText("官方评估") : uiText("无样本预览")));
        sampleTable_->setItem(0, 1, new QTableWidgetItem(QString()));
        sampleTable_->setItem(0, 2, new QTableWidgetItem(QString()));
        sampleTable_->setItem(0, 3, new QTableWidgetItem(QString()));
        sampleTable_->setItem(0, 4, new QTableWidgetItem(officialYolo
                ? uiText("Ultralytics val() 不生成 AITrain 本地错误样本。请查看官方图表和 predictions。")
                : uiText("该报告没有记录本地错误样本。请查看官方产物或报告详情。")));
        previewLabel_->setText(officialYolo
                ? uiText("官方 YOLO 评估不生成本地 overlay。")
                : uiText("该报告没有记录本地 overlay。"));
        detailText_->setPlainText(officialYolo
                ? uiText("YOLO 检测/分割/OBB 指标来自 Ultralytics official val()。AITrain 不再生成本地 AP/mAP、mask IoU、混淆矩阵、错误样本或 overlay。")
                : uiText("该报告没有本地样本预览；请查看上方指标、官方产物和报告详情。"));
        return;
    }

    sampleTable_->resizeColumnToContents(0);
    sampleTable_->resizeColumnToContents(2);
    sampleTable_->resizeColumnToContents(3);
    sampleTable_->selectRow(0);
}

void EvaluationReportView::showPreviewImage(const QString& imagePath)
{
    previewLabel_->clear();
    if (imagePath.isEmpty()) {
        previewLabel_->setText(uiText("该条目没有可预览图片。"));
        return;
    }
    const QPixmap image(imagePath);
    if (image.isNull()) {
        previewLabel_->setText(uiText("图片无法读取。"));
        return;
    }
    previewLabel_->setPixmap(image.scaled(
        previewLabel_->size().boundedTo(QSize(520, 320)),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation));
}

void EvaluationReportView::showEmptyState(const QString& text)
{
    summaryLabel_->setText(text);
    metricsTable_->setRowCount(0);
    perClassTable_->setRowCount(0);
    officialArtifactsTable_->setRowCount(0);
    sampleTable_->setRowCount(0);
    previewLabel_->setText(uiText("选择官方图表或样本 overlay 后显示预览。"));
    detailText_->setPlainText(text);
}
