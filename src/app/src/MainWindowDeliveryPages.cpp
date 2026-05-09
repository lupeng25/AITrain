#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "PluginMarketplaceWidget.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildConversionPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerExportButton = primaryButton(QStringLiteral("开始导出"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("ExperimentHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(14, 12, 14, 12);
    headerRoot->setSpacing(10);
    auto* headerTop = new QHBoxLayout;
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);
    auto* kicker = new QLabel(QStringLiteral("MODEL DELIVERY WORKBENCH"));
    kicker->setObjectName(QStringLiteral("ExperimentKicker"));
    auto* title = new QLabel(QStringLiteral("模型导出"));
    title->setObjectName(QStringLiteral("ExperimentTitle"));
    auto* subtitle = new QLabel(QStringLiteral("从训练产物生成 ONNX、NCNN、TensorRT 或诊断 JSON，并把报告写入任务与产物。"));
    subtitle->setObjectName(QStringLiteral("ExperimentMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerExportButton);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* sourceLabel = new QLabel(QStringLiteral("输入"));
    sourceLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* policyLabel = new QLabel(QStringLiteral("策略"));
    policyLabel->setObjectName(QStringLiteral("ExperimentMeta"));
    auto* sourceStatus = inlineStatusLabel(QStringLiteral("优先从“任务与产物”选择 checkpoint、ONNX 或官方训练产物。"));
    sourceStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* policyStatus = inlineStatusLabel(QStringLiteral("导出只通过 Worker 执行；TensorRT 仍需 RTX / SM 75+ 外部验收。"));
    policyStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(sourceStatus);
    allowLabelToShrink(policyStatus);
    headerGrid->addWidget(sourceLabel, 0, 0);
    headerGrid->addWidget(sourceStatus, 0, 1);
    headerGrid->addWidget(policyLabel, 1, 0);
    headerGrid->addWidget(policyStatus, 1, 1);
    headerRoot->addLayout(headerGrid);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* setupPanel = new InfoPanel(QStringLiteral("导出设置"));
    conversionCheckpointEdit_ = new QLineEdit;
    conversionCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 checkpoint / ONNX / AITrain export"));
    auto* chooseCheckpointButton = new QPushButton(QStringLiteral("选择模型产物"));
    connect(chooseCheckpointButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型产物"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.pt *.pdparams *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            conversionCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });

    conversionFormatCombo_ = new QComboBox;
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("onnx")), QStringLiteral("onnx"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("ncnn")), QStringLiteral("ncnn"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tiny_detector_json")), QStringLiteral("tiny_detector_json"));
    conversionFormatCombo_->addItem(exportComboLabel(QStringLiteral("tensorrt")), QStringLiteral("tensorrt"));

    conversionOutputEdit_ = new QLineEdit;
    conversionOutputEdit_->setPlaceholderText(QStringLiteral("留空则输出到项目 models/exported；未打开项目时使用输入同目录"));
    auto* chooseOutputButton = new QPushButton(QStringLiteral("选择输出"));
    connect(chooseOutputButton, &QPushButton::clicked, this, [this]() {
        const QString format = conversionFormatCombo_
            ? conversionFormatCombo_->currentData().toString()
            : QStringLiteral("onnx");
        const QString inputPath = QDir::fromNativeSeparators(conversionCheckpointEdit_ ? conversionCheckpointEdit_->text().trimmed() : QString());
        const QString defaultDir = !currentProjectPath_.isEmpty()
            ? QDir(currentProjectPath_).filePath(QStringLiteral("models/exported"))
            : QFileInfo(inputPath).absolutePath();
        const QString selected = QFileDialog::getSaveFileName(
            this,
            uiText("选择导出路径"),
            QDir(defaultDir).filePath(defaultExportFileName(format)),
            exportFileFilter(format));
        if (!selected.isEmpty()) {
            conversionOutputEdit_->setText(QDir::toNativeSeparators(selected));
        }
    });
    auto* exportButton = primaryButton(QStringLiteral("开始导出"));
    connect(headerExportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::startModelExport);

    auto* inputRow = new QWidget;
    auto* inputLayout = new QHBoxLayout(inputRow);
    inputLayout->setContentsMargins(0, 0, 0, 0);
    inputLayout->setSpacing(8);
    inputLayout->addWidget(conversionCheckpointEdit_, 1);
    inputLayout->addWidget(chooseCheckpointButton);

    auto* outputRow = new QWidget;
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->setSpacing(8);
    outputLayout->addWidget(conversionOutputEdit_, 1);
    outputLayout->addWidget(chooseOutputButton);

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(QStringLiteral("模型输入"), inputRow);
    form->addRow(QStringLiteral("目标格式"), conversionFormatCombo_);
    form->addRow(QStringLiteral("输出路径"), outputRow);
    setupPanel->bodyLayout()->addLayout(form);

    auto* sourceHelp = emptyStateLabel(QStringLiteral("从“任务与产物”中选中 best.onnx、checkpoint 或官方导出目录后，可点击“用作导出输入”自动带入这里。"));
    allowLabelToShrink(sourceHelp);
    setupPanel->bodyLayout()->addWidget(sourceHelp);

    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    auto* exportActionHint = mutedLabel(QStringLiteral("导出任务会记录到任务历史，完成后可直接作为推理输入。"));
    allowLabelToShrink(exportActionHint);
    actionLayout->addWidget(exportActionHint, 1);
    actionLayout->addWidget(exportButton);
    setupPanel->bodyLayout()->addWidget(actionStrip);
    setupPanel->bodyLayout()->addStretch();

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* matrixPanel = new InfoPanel(QStringLiteral("格式矩阵"));
    auto* matrixTable = new QTableWidget(4, 4);
    matrixTable->setWordWrap(true);
    matrixTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("格式")
        << QStringLiteral("输入")
        << QStringLiteral("产物")
        << QStringLiteral("状态"));
    configureTable(matrixTable);
    matrixTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    matrixTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    matrixTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    matrixTable->setMinimumHeight(170);
    matrixTable->verticalHeader()->setDefaultSectionSize(44);
    const QStringList formats = {
        QStringLiteral("onnx"),
        QStringLiteral("ncnn"),
        QStringLiteral("tiny_detector_json"),
        QStringLiteral("tensorrt")
    };
    for (int row = 0; row < formats.size(); ++row) {
        const QString format = formats.at(row);
        matrixTable->setItem(row, 0, new QTableWidgetItem(exportFormatLabel(format)));
        matrixTable->setItem(row, 1, new QTableWidgetItem(
            format == QStringLiteral("tiny_detector_json")
                ? QStringLiteral("tiny detector checkpoint")
                : uiText("checkpoint 或 ONNX")));
        matrixTable->setItem(row, 2, new QTableWidgetItem(defaultExportFileName(format)));
        matrixTable->setItem(row, 3, new QTableWidgetItem(exportFormatNote(format)));
    }
    matrixPanel->bodyLayout()->addWidget(matrixTable);

    auto* resultPanel = new InfoPanel(QStringLiteral("运行状态"));
    exportResultLabel_ = inlineStatusLabel(QStringLiteral("暂无导出任务。"));
    resultPanel->bodyLayout()->addWidget(exportResultLabel_);
    resultPanel->bodyLayout()->addWidget(mutedLabel(QStringLiteral("ONNX 会写入 AITrain sidecar；NCNN 依赖本机 onnx2ncnn；TensorRT 在当前 GTX 1060 / SM 61 上保持 hardware-blocked。")));
    resultPanel->bodyLayout()->addStretch();

    rightLayout->addWidget(matrixPanel);
    rightLayout->addWidget(resultPanel, 1);

    mainSplitter->addWidget(setupPanel);
    mainSplitter->addWidget(rightStack);
    mainSplitter->setChildrenCollapsible(false);
    mainSplitter->setStretchFactor(0, 3);
    mainSplitter->setStretchFactor(1, 4);
    mainSplitter->setSizes(QList<int>() << 520 << 680);

    layout->addWidget(headerPanel);
    layout->addWidget(mainSplitter, 1);
    return page;
}

QWidget* MainWindow::buildInferencePage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerInferButton = primaryButton(QStringLiteral("开始推理"));

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("InferenceHeader"));
    auto* headerRoot = new QVBoxLayout(headerPanel);
    headerRoot->setContentsMargins(16, 14, 16, 14);
    headerRoot->setSpacing(11);

    auto* headerTop = new QHBoxLayout;
    headerTop->setSpacing(14);
    auto* titleBlock = new QWidget;
    auto* titleLayout = new QVBoxLayout(titleBlock);
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(4);
    auto* kicker = new QLabel(QStringLiteral("INFERENCE VALIDATION"));
    kicker->setObjectName(QStringLiteral("InferenceKicker"));
    auto* title = new QLabel(QStringLiteral("推理验证工作台"));
    title->setObjectName(QStringLiteral("InferenceTitle"));
    auto* subtitle = new QLabel(QStringLiteral("选择模型和图片，运行 Worker 推理并查看 prediction JSON 与 overlay。这里不直接承载模型后处理逻辑。"));
    subtitle->setObjectName(QStringLiteral("InferenceMeta"));
    subtitle->setWordWrap(true);
    allowLabelToShrink(subtitle);
    auto* badgeRow = new QWidget;
    auto* badgeLayout = new QHBoxLayout(badgeRow);
    badgeLayout->setContentsMargins(0, 4, 0, 0);
    badgeLayout->setSpacing(7);
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("ONNX Runtime")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Worker")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Prediction JSON")));
    badgeLayout->addWidget(inferenceBadge(QStringLiteral("Overlay")));
    badgeLayout->addStretch();
    titleLayout->addWidget(kicker);
    titleLayout->addWidget(title);
    titleLayout->addWidget(subtitle);
    titleLayout->addWidget(badgeRow);
    headerTop->addWidget(titleBlock, 1);
    headerTop->addWidget(headerInferButton, 0, Qt::AlignTop);
    headerRoot->addLayout(headerTop);

    auto* headerGrid = new QGridLayout;
    headerGrid->setHorizontalSpacing(12);
    headerGrid->setVerticalSpacing(8);
    auto* executionLabel = new QLabel(QStringLiteral("执行"));
    executionLabel->setObjectName(QStringLiteral("InferenceMeta"));
    auto* boundaryLabel = new QLabel(QStringLiteral("边界"));
    boundaryLabel->setObjectName(QStringLiteral("InferenceMeta"));
    auto* executionStatus = inlineStatusLabel(QStringLiteral("Worker 隔离执行推理；任务、产物和失败原因写入任务历史。"));
    executionStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    auto* boundaryStatus = inlineStatusLabel(QStringLiteral("C++ ONNX 支持 detection / segmentation / OCR Rec；PaddleOCR System 仍查看官方工具链产物。"));
    boundaryStatus->setObjectName(QStringLiteral("DarkInlineStatus"));
    allowLabelToShrink(executionStatus);
    allowLabelToShrink(boundaryStatus);
    headerGrid->addWidget(executionLabel, 0, 0);
    headerGrid->addWidget(executionStatus, 0, 1);
    headerGrid->addWidget(boundaryLabel, 1, 0);
    headerGrid->addWidget(boundaryStatus, 1, 1);
    headerGrid->setColumnStretch(1, 1);
    headerRoot->addLayout(headerGrid);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* leftStack = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftStack);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("验证输入"));
    auto* inferForm = new QFormLayout;
    inferenceCheckpointEdit_ = new QLineEdit;
    inferenceImageEdit_ = new QLineEdit;
    inferenceOutputEdit_ = new QLineEdit;
    for (QLineEdit* edit : {inferenceCheckpointEdit_, inferenceImageEdit_, inferenceOutputEdit_}) {
        edit->setMinimumWidth(0);
        edit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    }
    inferenceCheckpointEdit_->setPlaceholderText(QStringLiteral("从任务产物带入，或选择 ONNX / AITrain export / TensorRT engine"));
    inferenceImageEdit_->setPlaceholderText(QStringLiteral("选择验证图片"));
    inferenceOutputEdit_->setPlaceholderText(QStringLiteral("输出目录；留空则输出到模型同目录 inference"));
    auto* chooseModelButton = new QPushButton(QStringLiteral("选择模型文件"));
    auto* chooseImageButton = new QPushButton(QStringLiteral("选择图片"));
    auto* chooseOutputButton = new QPushButton(QStringLiteral("选择输出目录"));
    auto* inferButton = primaryButton(QStringLiteral("开始推理"));
    connect(chooseModelButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择模型文件"), currentProjectPath_, QStringLiteral("AITrain model (*.aitrain *.json *.onnx *.engine *.plan);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceCheckpointEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseImageButton, &QPushButton::clicked, this, [this]() {
        const QString file = QFileDialog::getOpenFileName(this, uiText("选择图片"), currentProjectPath_, QStringLiteral("Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)"));
        if (!file.isEmpty()) {
            inferenceImageEdit_->setText(QDir::toNativeSeparators(file));
        }
    });
    connect(chooseOutputButton, &QPushButton::clicked, this, [this]() {
        const QString modelPath = QDir::fromNativeSeparators(inferenceCheckpointEdit_ ? inferenceCheckpointEdit_->text().trimmed() : QString());
        const QString currentOutput = QDir::fromNativeSeparators(inferenceOutputEdit_ ? inferenceOutputEdit_->text().trimmed() : QString());
        const QString defaultDir = !currentOutput.isEmpty()
            ? currentOutput
            : (!currentProjectPath_.isEmpty()
                ? QDir(currentProjectPath_).filePath(QStringLiteral("inference"))
                : QFileInfo(modelPath).absoluteDir().filePath(QStringLiteral("inference")));
        const QString dir = QFileDialog::getExistingDirectory(this, uiText("选择推理输出目录"), defaultDir);
        if (!dir.isEmpty() && inferenceOutputEdit_) {
            inferenceOutputEdit_->setText(QDir::toNativeSeparators(dir));
        }
    });
    connect(inferButton, &QPushButton::clicked, this, &MainWindow::startInference);
    auto* modelRow = new QWidget;
    auto* modelLayout = new QHBoxLayout(modelRow);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    modelLayout->addWidget(inferenceCheckpointEdit_);
    modelLayout->addWidget(chooseModelButton);
    auto* imageRow = new QWidget;
    auto* imageLayout = new QHBoxLayout(imageRow);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->setSpacing(8);
    imageLayout->addWidget(inferenceImageEdit_);
    imageLayout->addWidget(chooseImageButton);
    modelLayout->setSpacing(8);
    auto* outputRow = new QWidget;
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->setSpacing(8);
    outputLayout->addWidget(inferenceOutputEdit_);
    outputLayout->addWidget(chooseOutputButton);
    inferForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inferForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    inferForm->setHorizontalSpacing(14);
    inferForm->setVerticalSpacing(10);
    inferForm->addRow(QStringLiteral("模型路径"), modelRow);
    inferForm->addRow(QStringLiteral("图片路径"), imageRow);
    inferForm->addRow(QStringLiteral("推理输出"), outputRow);
    toolbar->bodyLayout()->addLayout(inferForm);
    auto* sourceHelp = emptyStateLabel(QStringLiteral("从“任务与产物”选中 ONNX、AITrain export 或 engine 后，可点击“用作推理模型”自动带入这里。输出目录留空会写到模型同目录 inference。"));
    allowLabelToShrink(sourceHelp);
    toolbar->bodyLayout()->addWidget(sourceHelp);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(10);
    auto* inferenceActionHint = mutedLabel(QStringLiteral("推理任务会记录到任务历史，完成后可在产物详情中复查 JSON、overlay 和耗时。"));
    allowLabelToShrink(inferenceActionHint);
    actionLayout->addWidget(inferenceActionHint, 1);
    actionLayout->addWidget(inferButton);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addStretch();

    auto* capabilityPanel = new InfoPanel(QStringLiteral("可解析结果"));
    auto* capabilityHint = mutedLabel(QStringLiteral("推理验证会根据 ONNX 模型族或 AITrain 导出信息选择后处理；scaffold / 官方工具链边界保持显式。"));
    allowLabelToShrink(capabilityHint);
    capabilityPanel->bodyLayout()->addWidget(capabilityHint);
    auto* capabilityGrid = new QGridLayout;
    capabilityGrid->setHorizontalSpacing(10);
    capabilityGrid->setVerticalSpacing(10);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 检测"), QStringLiteral("box、类别、置信度、NMS 与 overlay。")), 0, 0);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 分割"), QStringLiteral("box、mask、mask area 与半透明 overlay。")), 0, 1);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("OCR Rec"), QStringLiteral("CTC greedy decode、文本与置信度摘要。")), 1, 0);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("PaddleOCR System"), QStringLiteral("端到端结果仍通过官方工具链任务产物查看。")), 1, 1);
    capabilityGrid->setColumnStretch(0, 1);
    capabilityGrid->setColumnStretch(1, 1);
    capabilityPanel->bodyLayout()->addLayout(capabilityGrid);
    capabilityPanel->bodyLayout()->addStretch();

    leftLayout->addWidget(toolbar, 3);
    leftLayout->addWidget(capabilityPanel, 2);

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* flowPanel = new InfoPanel(QStringLiteral("验证链路"));
    auto* flowGrid = new QGridLayout;
    flowGrid->setHorizontalSpacing(10);
    flowGrid->setVerticalSpacing(10);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("1"), QStringLiteral("模型产物"), QStringLiteral("ONNX / AITrain export / TensorRT engine")), 0, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("2"), QStringLiteral("样本图片"), QStringLiteral("单张验证图进入预处理")), 0, 1);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("3"), QStringLiteral("Worker 推理"), QStringLiteral("隔离执行，不阻塞 GUI")), 1, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("4"), QStringLiteral("结果归档"), QStringLiteral("prediction JSON + overlay")), 1, 1);
    flowGrid->setColumnStretch(0, 1);
    flowGrid->setColumnStretch(1, 1);
    flowPanel->bodyLayout()->addLayout(flowGrid);

    auto* preview = new QSplitter(Qt::Vertical);
    auto* summaryPanel = new InfoPanel(QStringLiteral("结果摘要"));
    auto* summaryHint = mutedLabel(QStringLiteral("Worker 返回的 prediction JSON 会压缩显示任务类型、结果数量、首个类别 / 文本、耗时和结果文件路径。"));
    allowLabelToShrink(summaryHint);
    summaryPanel->bodyLayout()->addWidget(summaryHint);
    inferenceResultLabel_ = inlineStatusLabel(QStringLiteral("尚未推理。"));
    inferenceResultLabel_->setObjectName(QStringLiteral("InferenceResultSummary"));
    allowLabelToShrink(inferenceResultLabel_);
    summaryPanel->bodyLayout()->addWidget(inferenceResultLabel_);
    auto* summaryFootnote = mutedLabel(QStringLiteral("完整原始 JSON 可在“任务与产物”的产物详情中查看。"));
    allowLabelToShrink(summaryFootnote);
    summaryPanel->bodyLayout()->addWidget(summaryFootnote);
    summaryPanel->bodyLayout()->addStretch();

    auto* resultPanel = new InfoPanel(QStringLiteral("Overlay 预览"));
    auto* overlayHint = mutedLabel(QStringLiteral("完成后显示检测框、分割 mask 或 OCR 可视化图；失败时保留明确状态文本。"));
    allowLabelToShrink(overlayHint);
    resultPanel->bodyLayout()->addWidget(overlayHint);
    inferenceOverlayLabel_ = new QLabel(QStringLiteral("暂无 overlay\n运行推理后显示可视化产物。"));
    inferenceOverlayLabel_->setObjectName(QStringLiteral("InferenceOverlayCanvas"));
    inferenceOverlayLabel_->setAlignment(Qt::AlignCenter);
    inferenceOverlayLabel_->setMinimumHeight(260);
    inferenceOverlayLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    inferenceOverlayLabel_->setFrameShape(QFrame::NoFrame);
    resultPanel->bodyLayout()->addWidget(inferenceOverlayLabel_);
    preview->addWidget(summaryPanel);
    preview->addWidget(resultPanel);
    preview->setStretchFactor(0, 1);
    preview->setStretchFactor(1, 3);
    preview->setChildrenCollapsible(false);
    preview->setSizes(QList<int>() << 180 << 420);

    rightLayout->addWidget(flowPanel);
    rightLayout->addWidget(preview, 1);

    mainSplitter->addWidget(leftStack);
    mainSplitter->addWidget(rightStack);
    mainSplitter->setChildrenCollapsible(false);
    mainSplitter->setStretchFactor(0, 3);
    mainSplitter->setStretchFactor(1, 5);
    mainSplitter->setSizes(QList<int>() << 460 << 760);

    connect(headerInferButton, &QPushButton::clicked, this, &MainWindow::startInference);

    layout->addWidget(headerPanel);
    layout->addWidget(mainSplitter);
    page->setWidget(content);
    return page;
}
