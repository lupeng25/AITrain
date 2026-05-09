#include "TaskArtifactPanel.h"

#include "EvaluationReportView.h"
#include "MainWindowSupport.h"
#include "aitrain/core/DetectionTrainer.h"

#include <QAbstractItemView>
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSize>
#include <QSizePolicy>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

TaskArtifactPanel::TaskArtifactPanel(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    selectedTaskSummaryLabel_ = inlineStatusLabel(QStringLiteral("请选择一个任务查看产物、指标和导出记录。"));
    selectedTaskSummaryLabel_->setObjectName(QStringLiteral("TaskDetailSummary"));
    selectedTaskSummaryLabel_->setMinimumHeight(40);
    selectedTaskSummaryLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

    artifactTable_ = new QTableWidget(0, 4);
    artifactTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("类型")
        << QStringLiteral("路径")
        << QStringLiteral("说明")
        << QStringLiteral("时间"));
    configureTable(artifactTable_);
    artifactTable_->setWordWrap(true);
    artifactTable_->setMinimumHeight(220);
    artifactTable_->verticalHeader()->setDefaultSectionSize(42);
    artifactTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    artifactTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    artifactTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    artifactTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    connect(artifactTable_, &QTableWidget::itemSelectionChanged, this, &TaskArtifactPanel::updatePreviewFromSelection);

    metricTable_ = new QTableWidget(0, 4);
    metricTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("指标")
        << QStringLiteral("值")
        << QStringLiteral("Step")
        << QStringLiteral("Epoch"));
    configureTable(metricTable_);
    metricTable_->setMinimumHeight(210);
    metricTable_->verticalHeader()->setDefaultSectionSize(38);
    metricTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    metricTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    metricTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);

    exportTable_ = new QTableWidget(0, 3);
    exportTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("路径")
        << QStringLiteral("时间"));
    configureTable(exportTable_);
    exportTable_->setWordWrap(true);
    exportTable_->setMinimumHeight(210);
    exportTable_->verticalHeader()->setDefaultSectionSize(42);
    exportTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    exportTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    exportTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);

    imagePreviewLabel_ = new QLabel(QStringLiteral("暂无产物预览"));
    imagePreviewLabel_->setObjectName(QStringLiteral("ArtifactPreviewCanvas"));
    imagePreviewLabel_->setAlignment(Qt::AlignCenter);
    imagePreviewLabel_->setMinimumHeight(220);
    imagePreviewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    previewText_ = new QPlainTextEdit;
    previewText_->setObjectName(QStringLiteral("ArtifactPreviewText"));
    previewText_->setReadOnly(true);
    previewText_->setMinimumHeight(160);
    previewText_->setPlainText(QStringLiteral("选择一个产物后显示摘要。"));
    auto* defaultPreview = new QWidget;
    auto* defaultLayout = new QVBoxLayout(defaultPreview);
    defaultLayout->setContentsMargins(0, 0, 0, 0);
    defaultLayout->setSpacing(10);
    defaultLayout->addWidget(imagePreviewLabel_, 1);
    defaultLayout->addWidget(previewText_, 2);
    previewStack_ = new QStackedWidget;
    previewStack_->setMinimumHeight(220);
    previewStack_->addWidget(defaultPreview);
    evaluationReportView_ = new EvaluationReportView;
    auto* evaluationScroll = new QScrollArea;
    evaluationScroll->setWidget(evaluationReportView_);
    evaluationScroll->setWidgetResizable(true);
    evaluationScroll->setFrameShape(QFrame::NoFrame);
    evaluationScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    evaluationScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    previewStack_->addWidget(evaluationScroll);

    auto* actionGridFrame = new QFrame;
    actionGridFrame->setObjectName(QStringLiteral("ArtifactActionGrid"));
    actionGridFrame->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);
    auto* actionGrid = new QGridLayout(actionGridFrame);
    actionGrid->setContentsMargins(12, 10, 12, 10);
    actionGrid->setHorizontalSpacing(12);
    actionGrid->setVerticalSpacing(8);
    auto* openDirButton = new QPushButton(QStringLiteral("打开目录"));
    auto* copyPathButton = new QPushButton(QStringLiteral("复制路径"));
    auto* useInferButton = new QPushButton(QStringLiteral("用作推理模型"));
    auto* useExportButton = new QPushButton(QStringLiteral("用作导出输入"));
    auto* registerModelButton = new QPushButton(QStringLiteral("注册模型版本"));
    auto* evaluateButton = new QPushButton(QStringLiteral("评估"));
    auto* benchmarkButton = new QPushButton(QStringLiteral("基准"));
    auto* reportButton = new QPushButton(QStringLiteral("交付报告"));
    connect(openDirButton, &QPushButton::clicked, this, &TaskArtifactPanel::openDirectoryRequested);
    connect(copyPathButton, &QPushButton::clicked, this, &TaskArtifactPanel::copyPathRequested);
    connect(useInferButton, &QPushButton::clicked, this, &TaskArtifactPanel::useForInferenceRequested);
    connect(useExportButton, &QPushButton::clicked, this, &TaskArtifactPanel::useForExportRequested);
    connect(registerModelButton, &QPushButton::clicked, this, &TaskArtifactPanel::registerModelRequested);
    connect(evaluateButton, &QPushButton::clicked, this, &TaskArtifactPanel::evaluateRequested);
    connect(benchmarkButton, &QPushButton::clicked, this, &TaskArtifactPanel::benchmarkRequested);
    connect(reportButton, &QPushButton::clicked, this, &TaskArtifactPanel::deliveryReportRequested);
    actionGrid->addWidget(openDirButton, 0, 0);
    actionGrid->addWidget(copyPathButton, 0, 1);
    actionGrid->addWidget(useInferButton, 0, 2);
    actionGrid->addWidget(useExportButton, 0, 3);
    actionGrid->addWidget(registerModelButton, 1, 0);
    actionGrid->addWidget(evaluateButton, 1, 1);
    actionGrid->addWidget(benchmarkButton, 1, 2);
    actionGrid->addWidget(reportButton, 1, 3);
    for (int column = 0; column < 4; ++column) {
        actionGrid->setColumnStretch(column, 1);
    }

    auto* artifactTab = new QWidget;
    auto* artifactTabLayout = new QVBoxLayout(artifactTab);
    artifactTabLayout->setContentsMargins(0, 0, 0, 0);
    artifactTabLayout->addWidget(artifactTable_);
    auto* metricTab = new QWidget;
    auto* metricTabLayout = new QVBoxLayout(metricTab);
    metricTabLayout->setContentsMargins(0, 0, 0, 0);
    metricTabLayout->addWidget(metricTable_);
    auto* exportTab = new QWidget;
    auto* exportTabLayout = new QVBoxLayout(exportTab);
    exportTabLayout->setContentsMargins(0, 0, 0, 0);
    exportTabLayout->addWidget(exportTable_);
    auto* previewTab = new QWidget;
    auto* previewTabLayout = new QVBoxLayout(previewTab);
    previewTabLayout->setContentsMargins(0, 0, 0, 0);
    previewTabLayout->addWidget(previewStack_);

    auto* detailTabs = new QTabWidget;
    detailTabs->setObjectName(QStringLiteral("TaskDetailTabs"));
    detailTabs->addTab(artifactTab, uiText("产物"));
    detailTabs->addTab(metricTab, uiText("指标"));
    detailTabs->addTab(exportTab, uiText("导出"));
    detailTabs->addTab(previewTab, uiText("预览"));

    detailTabs->setMinimumHeight(420);
    detailTabs->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    previewStack_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    layout->addWidget(selectedTaskSummaryLabel_);
    layout->addWidget(detailTabs, 1);
    layout->addWidget(actionGridFrame);

    clear();
}

void TaskArtifactPanel::clear()
{
    setTaskSummary(uiText("请选择一个任务查看产物、指标和导出记录。"));
    clearTableWithPlaceholder(artifactTable_, uiText("暂无产物"));
    clearTableWithPlaceholder(metricTable_, uiText("暂无指标"));
    clearTableWithPlaceholder(exportTable_, uiText("暂无导出"));
    previewArtifactPath(QString());
}

void TaskArtifactPanel::setTaskSummary(const QString& summary)
{
    if (selectedTaskSummaryLabel_) {
        selectedTaskSummaryLabel_->setText(summary);
    }
}

void TaskArtifactPanel::setArtifacts(const QVector<aitrain::ArtifactRecord>& artifacts)
{
    artifactTable_->setRowCount(0);
    if (artifacts.isEmpty()) {
        clearTableWithPlaceholder(artifactTable_, uiText("暂无产物"));
        previewArtifactPath(QString());
        return;
    }

    for (const aitrain::ArtifactRecord& artifact : artifacts) {
        const int row = artifactTable_->rowCount();
        artifactTable_->insertRow(row);
        artifactTable_->setItem(row, 0, new QTableWidgetItem(artifact.kind));
        auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(artifact.path));
        pathItem->setData(Qt::UserRole, artifact.path);
        artifactTable_->setItem(row, 1, pathItem);
        artifactTable_->setItem(row, 2, new QTableWidgetItem(artifact.message));
        artifactTable_->setItem(row, 3, new QTableWidgetItem(artifact.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
    }

    int preferredRow = 0;
    for (int row = 0; row < artifactTable_->rowCount(); ++row) {
        auto* kindItem = artifactTable_->item(row, 0);
        if (kindItem && kindItem->text() == QStringLiteral("evaluation_report")) {
            preferredRow = row;
            break;
        }
    }
    artifactTable_->selectRow(preferredRow);
    previewArtifactPath(selectedArtifactPath());
}

void TaskArtifactPanel::setMetrics(const QVector<aitrain::MetricPoint>& metrics)
{
    metricTable_->setRowCount(0);
    if (metrics.isEmpty()) {
        clearTableWithPlaceholder(metricTable_, uiText("暂无指标"));
        return;
    }

    for (const aitrain::MetricPoint& metric : metrics) {
        const int row = metricTable_->rowCount();
        metricTable_->insertRow(row);
        metricTable_->setItem(row, 0, new QTableWidgetItem(metric.name));
        metricTable_->setItem(row, 1, new QTableWidgetItem(QString::number(metric.value, 'f', 6)));
        metricTable_->setItem(row, 2, new QTableWidgetItem(QString::number(metric.step)));
        metricTable_->setItem(row, 3, new QTableWidgetItem(QString::number(metric.epoch)));
    }
}

void TaskArtifactPanel::setExports(const QVector<aitrain::ExportRecord>& exports)
{
    exportTable_->setRowCount(0);
    if (exports.isEmpty()) {
        clearTableWithPlaceholder(exportTable_, uiText("暂无导出"));
        return;
    }

    for (const aitrain::ExportRecord& exportRecord : exports) {
        const int row = exportTable_->rowCount();
        exportTable_->insertRow(row);
        exportTable_->setItem(row, 0, new QTableWidgetItem(exportRecord.format));
        auto* pathItem = new QTableWidgetItem(QDir::toNativeSeparators(exportRecord.path));
        pathItem->setData(Qt::UserRole, exportRecord.path);
        exportTable_->setItem(row, 1, pathItem);
        exportTable_->setItem(row, 2, new QTableWidgetItem(exportRecord.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
    }
}

QString TaskArtifactPanel::selectedArtifactPath() const
{
    if (!artifactTable_ || artifactTable_->selectedItems().isEmpty()) {
        return QString();
    }
    const int row = artifactTable_->selectedItems().first()->row();
    auto* item = artifactTable_->item(row, 1);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void TaskArtifactPanel::configureTable(QTableWidget* table) const
{
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setStretchLastSection(true);
}

void TaskArtifactPanel::clearTableWithPlaceholder(QTableWidget* table, const QString& placeholder)
{
    if (!table) {
        return;
    }
    table->clearSelection();
    table->setRowCount(0);
    table->insertRow(0);
    table->setItem(0, 0, new QTableWidgetItem(placeholder));
    for (int column = 1; column < table->columnCount(); ++column) {
        table->setItem(0, column, new QTableWidgetItem(QString()));
    }
}

void TaskArtifactPanel::updatePreviewFromSelection()
{
    previewArtifactPath(selectedArtifactPath());
}

void TaskArtifactPanel::previewArtifactPath(const QString& path)
{
    if (!previewText_ || !imagePreviewLabel_ || !previewStack_) {
        return;
    }

    previewStack_->setCurrentIndex(0);
    if (evaluationReportView_) {
        evaluationReportView_->clear();
    }
    imagePreviewLabel_->clear();
    imagePreviewLabel_->setVisible(false);
    imagePreviewLabel_->setText(uiText("暂无产物预览"));
    previewText_->setVisible(true);
    previewText_->clear();
    if (path.isEmpty()) {
        imagePreviewLabel_->setVisible(true);
        previewText_->setVisible(false);
        return;
    }

    const QFileInfo info(path);
    if (!info.exists()) {
        previewText_->setPlainText(uiText("产物不存在：%1").arg(QDir::toNativeSeparators(path)));
        return;
    }

    if (info.isDir()) {
        previewText_->setPlainText(uiText("目录产物\n路径：%1\n修改时间：%2")
            .arg(QDir::toNativeSeparators(path), info.lastModified().toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
        return;
    }

    const QString suffix = info.suffix().toLower();
    if (QStringList{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("bmp")}.contains(suffix)) {
        QPixmap image(path);
        if (!image.isNull()) {
            imagePreviewLabel_->setVisible(true);
            imagePreviewLabel_->setPixmap(image.scaled(
                imagePreviewLabel_->size().boundedTo(QSize(520, 360)),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation));
        }
        previewText_->setPlainText(uiText("图片产物\n路径：%1\n尺寸：%2 x %3\n大小：%4 bytes")
            .arg(QDir::toNativeSeparators(path))
            .arg(image.width())
            .arg(image.height())
            .arg(info.size()));
        return;
    }

    if (suffix == QStringLiteral("onnx")) {
        previewText_->setPlainText(uiText("ONNX 模型\n路径：%1\n模型族：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), aitrain::inferOnnxModelFamily(path))
            .arg(info.size()));
        return;
    }
    if (suffix == QStringLiteral("engine") || suffix == QStringLiteral("plan") || suffix == QStringLiteral("pdparams")
        || suffix == QStringLiteral("aitrain") || suffix == QStringLiteral("param") || suffix == QStringLiteral("bin")) {
        previewText_->setPlainText(uiText("模型产物\n路径：%1\n类型：%2\n大小：%3 bytes")
            .arg(QDir::toNativeSeparators(path), suffix)
            .arg(info.size()));
        return;
    }

    if (QStringList{QStringLiteral("json"), QStringLiteral("yaml"), QStringLiteral("yml"), QStringLiteral("txt"), QStringLiteral("csv"), QStringLiteral("log")}.contains(suffix)) {
        if (suffix == QStringLiteral("json") && info.fileName() == QStringLiteral("evaluation_report.json") && evaluationReportView_) {
            evaluationReportView_->loadReport(path);
            previewStack_->setCurrentIndex(1);
            return;
        }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            previewText_->setPlainText(uiText("无法读取文本产物：%1").arg(QDir::toNativeSeparators(path)));
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
        previewText_->setPlainText(text);
        return;
    }

    previewText_->setPlainText(uiText("不支持内联预览的产物\n路径：%1\n大小：%2 bytes")
        .arg(QDir::toNativeSeparators(path))
        .arg(info.size()));
}
