#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"

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

QWidget* MainWindow::buildDeploymentPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("DEPLOYMENT VALIDATION"),
        uiText("部署验证"),
        uiText("基于已登记且校验通过的模型包运行推理与部署验证。"),
        nullptr,
        QStringList()
            << QStringLiteral("ONNX")
            << QStringLiteral("NCNN")
            << QStringLiteral("TensorRT")
            << uiText("推理")));

    deploymentTabs_ = new QTabWidget;
    deploymentTabs_->setObjectName(QStringLiteral("DeploymentTabs"));
    deploymentTabs_->addTab(buildDeploymentValidationPanel(), uiText("部署验证"));
    deploymentTabs_->addTab(buildInferenceValidationPanel(), uiText("推理验证"));
    layout->addWidget(deploymentTabs_, 1);
    return page;
}

QWidget* MainWindow::buildDeploymentValidationPanel()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    page->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* setupPanel = new InfoPanel(QStringLiteral(" 模型包部署验证"));
    deploymentModelPackageCombo_ = new QComboBox;
    deploymentModelPackageCombo_->setObjectName(QStringLiteral("DeploymentModelPackageCombo"));
    deploymentModelPackageCombo_->setMinimumWidth(0);
    deploymentModelPackageCombo_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    deploymentModelPackageCombo_->addItem(uiText("请先打开项目并导入已验证模型包"), QString());

    deploymentSampleDatasetIdEdit_ = new QLineEdit;
    deploymentSampleDatasetVersionIdEdit_ = new QLineEdit;
    deploymentSampleSnapshotIdEdit_ = new QLineEdit;
    deploymentSampleSnapshotArtifactIdEdit_ = new QLineEdit;
    deploymentSampleRelativePathEdit_ = new QLineEdit;
    for (QLineEdit* edit : {deploymentSampleDatasetIdEdit_, deploymentSampleDatasetVersionIdEdit_,
             deploymentSampleSnapshotIdEdit_, deploymentSampleSnapshotArtifactIdEdit_,
             deploymentSampleRelativePathEdit_}) {
        edit->setMinimumWidth(0);
        edit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    }
    deploymentSampleDatasetIdEdit_->setPlaceholderText(uiText("DatasetId"));
    deploymentSampleDatasetVersionIdEdit_->setPlaceholderText(uiText("DatasetVersionId"));
    deploymentSampleSnapshotIdEdit_->setPlaceholderText(uiText("SnapshotId"));
    deploymentSampleSnapshotArtifactIdEdit_->setPlaceholderText(uiText("Snapshot ArtifactId"));
    deploymentSampleRelativePathEdit_->setPlaceholderText(uiText("样本在 Snapshot Artifact 内的相对路径，例如 images/0001.png"));

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setRowWrapPolicy(QFormLayout::WrapLongRows);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    form->addRow(uiText("已验证模型包"), deploymentModelPackageCombo_);
    form->addRow(uiText("样本 DatasetId"), deploymentSampleDatasetIdEdit_);
    form->addRow(uiText("样本 VersionId"), deploymentSampleDatasetVersionIdEdit_);
    form->addRow(uiText("样本 SnapshotId"), deploymentSampleSnapshotIdEdit_);
    form->addRow(uiText("样本 ArtifactId"), deploymentSampleSnapshotArtifactIdEdit_);
    form->addRow(uiText("样本相对路径"), deploymentSampleRelativePathEdit_);
    setupPanel->bodyLayout()->addLayout(form);

    auto* boundary = emptyStateLabel(uiText(
        "此入口只接受由 ModelPackageId 解析的模型包，以及已提交 Dataset Snapshot Artifact 内的样本；"
        "不再接受 checkpoint、ONNX、engine 或样本图片裸路径。"));
    allowLabelToShrink(boundary);
    setupPanel->bodyLayout()->addWidget(boundary);

    auto* validateButton = primaryButton(uiText("运行完整 Runtime Delivery"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDeploymentModelPackage);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->addStretch();
    actionLayout->addWidget(validateButton);
    setupPanel->bodyLayout()->addWidget(actionStrip);

    auto* resultPanel = new InfoPanel(QStringLiteral("运行状态"));
    deploymentValidationResultLabel_ = inlineStatusLabel(uiText("尚未运行 Runtime Delivery 六步工作流。"));
    resultPanel->bodyLayout()->addWidget(deploymentValidationResultLabel_);
    resultPanel->bodyLayout()->addWidget(mutedLabel(uiText(
        "实际 runtime 路由由模型包 Manifest 与 Runtime 能力矩阵共同决定；"
        "缺少 SDK、依赖、硬件或 decoder 时会返回精确状态。")));

    layout->addWidget(setupPanel);
    layout->addWidget(resultPanel);
    layout->addStretch();
    page->setWidget(content);
    return page;
}

QWidget* MainWindow::buildInferenceValidationPanel()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* mainSplitter = new QSplitter(Qt::Horizontal);

    auto* leftStack = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftStack);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(16);

    auto* toolbar = new InfoPanel(QStringLiteral("验证输入"));
    auto* inferForm = new QFormLayout;
    inferenceModelPackageCombo_ = new QComboBox;
    inferenceSampleDatasetIdEdit_ = new QLineEdit;
    inferenceSampleDatasetVersionIdEdit_ = new QLineEdit;
    inferenceSampleSnapshotIdEdit_ = new QLineEdit;
    inferenceSampleSnapshotArtifactIdEdit_ = new QLineEdit;
    inferenceSampleRelativePathEdit_ = new QLineEdit;
    for (QLineEdit* edit : {inferenceSampleDatasetIdEdit_, inferenceSampleDatasetVersionIdEdit_,
             inferenceSampleSnapshotIdEdit_, inferenceSampleSnapshotArtifactIdEdit_,
             inferenceSampleRelativePathEdit_}) {
        edit->setMinimumWidth(0);
        edit->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    }
    inferenceModelPackageCombo_->setMinimumWidth(0);
    inferenceModelPackageCombo_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    inferenceModelPackageCombo_->addItem(uiText("请先打开项目并导入已验证模型包"), QString());
    inferenceSampleDatasetIdEdit_->setPlaceholderText(QStringLiteral("DatasetId"));
    inferenceSampleDatasetVersionIdEdit_->setPlaceholderText(QStringLiteral("DatasetVersionId"));
    inferenceSampleSnapshotIdEdit_->setPlaceholderText(QStringLiteral("SnapshotId"));
    inferenceSampleSnapshotArtifactIdEdit_->setPlaceholderText(QStringLiteral("Snapshot ArtifactId"));
    inferenceSampleRelativePathEdit_->setPlaceholderText(QStringLiteral("样本在 Snapshot Artifact 内的相对路径，例如 images/0001.png"));
    auto* inferButton = primaryButton(QStringLiteral("运行完整 Runtime Delivery"));
    connect(inferButton, &QPushButton::clicked, this, &MainWindow::startInference);
    auto* modelRow = new QWidget;
    auto* modelLayout = new QHBoxLayout(modelRow);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    modelLayout->addWidget(inferenceModelPackageCombo_);
    auto* imageRow = new QWidget;
    auto* imageLayout = new QHBoxLayout(imageRow);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    imageLayout->setSpacing(8);
    imageLayout->addWidget(inferenceSampleRelativePathEdit_);
    modelLayout->setSpacing(8);
    auto* outputHint = mutedLabel(uiText("输出由 Artifact Store 托管，完成后在任务与产物中按 ArtifactId 预览。"));
    allowLabelToShrink(outputHint);
    inferForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    inferForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    inferForm->setHorizontalSpacing(14);
    inferForm->setVerticalSpacing(10);
    inferForm->addRow(QStringLiteral("已验证模型包"), modelRow);
    inferForm->addRow(QStringLiteral("样本 DatasetId"), inferenceSampleDatasetIdEdit_);
    inferForm->addRow(QStringLiteral("样本 VersionId"), inferenceSampleDatasetVersionIdEdit_);
    inferForm->addRow(QStringLiteral("样本 SnapshotId"), inferenceSampleSnapshotIdEdit_);
    inferForm->addRow(QStringLiteral("样本 ArtifactId"), inferenceSampleSnapshotArtifactIdEdit_);
    inferForm->addRow(QStringLiteral("样本相对路径"), imageRow);
    inferForm->addRow(QStringLiteral("推理输出"), outputHint);
    toolbar->bodyLayout()->addLayout(inferForm);
    auto* sourceHelp = emptyStateLabel(QStringLiteral("推理只能使用模型库中已登记、已校验哈希且声明 ONNX Runtime 路由的模型包，以及已提交 Snapshot Artifact 内的样本。模型文件、样本图片、NCNN 和 TensorRT engine 裸路径不会进入此推理链路。"));
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
    auto* capabilityHint = mutedLabel(QStringLiteral("当前推理仅执行已验证 ONNX Runtime 路由；具体检测、分割、OBB 或语义分割解码由 Model Manifest 声明。TensorRT、NCNN、异常检测和 OCR 不进入此运行时。"));
    allowLabelToShrink(capabilityHint);
    capabilityPanel->bodyLayout()->addWidget(capabilityHint);
    auto* capabilityGrid = new QGridLayout;
    capabilityGrid->setHorizontalSpacing(10);
    capabilityGrid->setVerticalSpacing(10);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 检测"), QStringLiteral("box、类别、置信度、NMS 与 overlay。")), 0, 0);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO 分割"), QStringLiteral("box、mask、mask area 与半透明 overlay。")), 0, 1);
    capabilityGrid->addWidget(createInferenceCapability(QStringLiteral("YOLO OBB"), QStringLiteral("旋转四边形、xywhr、外接 bbox 与 overlay。")), 1, 0);
    capabilityGrid->addWidget(createInferenceCapability(uiText("PaddleOCR 官方"), uiText("Det / Rec / System 结果通过官方报告和可视化产物查看。")), 1, 1);
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

    auto* flowPanel = new InfoPanel(QStringLiteral("Runtime Delivery 六步链路"));
    auto* flowGrid = new QGridLayout;
    flowGrid->setHorizontalSpacing(10);
    flowGrid->setVerticalSpacing(10);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("1"), QStringLiteral("解析模型包"), QStringLiteral("只接受 ModelPackageId 与已提交 Artifact")), 0, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("2"), QStringLiteral("校验 Manifest"), QStringLiteral("校验 runtime、decoder、哈希与依赖")), 0, 1);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("3"), QStringLiteral("推理 Smoke"), QStringLiteral("Worker 内同步 Runtime 推理")), 1, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("4"), QStringLiteral("Benchmark"), QStringLiteral("固定样本 smoke timing，非性能验收")), 1, 1);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("5"), QStringLiteral("部署验证"), QStringLiteral("提交预测、overlay 与验证报告")), 2, 0);
    flowGrid->addWidget(createInferenceStep(QStringLiteral("6"), QStringLiteral("交付报告"), QStringLiteral("生成终态 Evidence 与 Model Card")), 2, 1);
    flowGrid->setColumnStretch(0, 1);
    flowGrid->setColumnStretch(1, 1);
    flowPanel->bodyLayout()->addLayout(flowGrid);

    auto* preview = new QSplitter(Qt::Vertical);
    auto* summaryPanel = new InfoPanel(QStringLiteral("结果摘要"));
    auto* summaryHint = mutedLabel(QStringLiteral("Worker 返回的 prediction JSON 会压缩显示任务类型、结果数量、首个类别 / 文本、耗时和结果文件路径。"));
    allowLabelToShrink(summaryHint);
    summaryPanel->bodyLayout()->addWidget(summaryHint);
    inferenceResultLabel_ = inlineStatusLabel(QStringLiteral("尚未运行 Runtime Delivery 六步工作流。"));
    inferenceResultLabel_->setObjectName(QStringLiteral("InferenceResultSummary"));
    allowLabelToShrink(inferenceResultLabel_);
    summaryPanel->bodyLayout()->addWidget(inferenceResultLabel_);
    auto* summaryFootnote = mutedLabel(QStringLiteral("完整原始 JSON 可在“任务与产物”的产物详情中查看。"));
    allowLabelToShrink(summaryFootnote);
    summaryPanel->bodyLayout()->addWidget(summaryFootnote);
    summaryPanel->bodyLayout()->addStretch();

    auto* resultPanel = new InfoPanel(QStringLiteral("Overlay 预览"));
    auto* overlayHint = mutedLabel(uiText("完成后显示检测框或分割 mask；OCR 可视化图来自 PaddleOCR 官方任务产物。"));
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

    layout->addWidget(mainSplitter);
    page->setWidget(content);
    return page;
}

QWidget* MainWindow::buildDeliveryEvidencePanel()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* splitter = new QSplitter(Qt::Horizontal);

    auto* leftStack = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftStack);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(16);

    auto* summaryPanel = new InfoPanel(uiText("验收证据"));
    deliveryAcceptanceSummaryLabel_ = inlineStatusLabel(uiText("等待导入或运行验收证据。"));
    deliveryAcceptanceSummaryLabel_->setObjectName(QStringLiteral("DeliveryAcceptanceSummary"));
    connect(deliveryAcceptanceSummaryLabel_, &QObject::destroyed, this, [this]() {
        deliveryAcceptanceSummaryLabel_ = nullptr;
    });
    summaryPanel->bodyLayout()->addWidget(deliveryAcceptanceSummaryLabel_);
    deliveryAcceptanceTable_ = new QTableWidget(0, 4);
    deliveryAcceptanceTable_->setObjectName(QStringLiteral("DeliveryAcceptanceTable"));
    connect(deliveryAcceptanceTable_, &QObject::destroyed, this, [this]() {
        deliveryAcceptanceTable_ = nullptr;
    });
    deliveryAcceptanceTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("项目")
        << uiText("状态")
        << uiText("证据")
        << uiText("说明"));
    configureTable(deliveryAcceptanceTable_);
    deliveryAcceptanceTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    deliveryAcceptanceTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    deliveryAcceptanceTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    deliveryAcceptanceTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::Stretch);
    deliveryAcceptanceTable_->setMinimumHeight(240);
    summaryPanel->bodyLayout()->addWidget(deliveryAcceptanceTable_);
    auto* importButton = new QPushButton(uiText("导入外部验收结果"));
    connect(importButton, &QPushButton::clicked, this, &MainWindow::importAcceptanceEvidence);
    summaryPanel->bodyLayout()->addWidget(importButton, 0, Qt::AlignRight);

    leftLayout->addWidget(summaryPanel, 1);

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);

    auto* ocrPanel = new InfoPanel(uiText("客户域 OCR 官方报告受控验收 "));
    const auto makePathRow = [this](QLineEdit** target, const QString& placeholder) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(8);
        *target = new QLineEdit;
        (*target)->setPlaceholderText(placeholder);
        auto* button = new QPushButton(uiText("选择文件"));
        connect(button, &QPushButton::clicked, this, [this, target]() {
            const QString selected = QFileDialog::getOpenFileName(this, uiText("选择文件"),
                currentProjectPath_, QStringLiteral("PaddleOCR official reports (*.json);;All files (*.*)"));
            if (!selected.isEmpty() && *target) {
                (*target)->setText(QDir::toNativeSeparators(selected));
            }
        });
        rowLayout->addWidget(*target, 1);
        rowLayout->addWidget(button);
        return row;
    };

    const auto makeIdPair = [](QLineEdit** snapshotId, QLineEdit** snapshotArtifactId,
                                const QString& objectPrefix) {
        auto* row = new QWidget;
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(8);
        *snapshotId = new QLineEdit;
        (*snapshotId)->setObjectName(objectPrefix + QStringLiteral("SnapshotId"));
        (*snapshotId)->setPlaceholderText(QStringLiteral("SnapshotId（可选）"));
        *snapshotArtifactId = new QLineEdit;
        (*snapshotArtifactId)->setObjectName(objectPrefix + QStringLiteral("SnapshotArtifactId"));
        (*snapshotArtifactId)->setPlaceholderText(QStringLiteral("Snapshot ArtifactId（可选）"));
        layout->addWidget(*snapshotId, 1);
        layout->addWidget(*snapshotArtifactId, 1);
        return row;
    };

    auto* ocrForm = new QFormLayout;
    ocrForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    ocrForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    ocrForm->setHorizontalSpacing(12);
    ocrForm->setVerticalSpacing(9);
    auto* importTitle = new QLabel(uiText("步骤 1：受控导入（裸路径仅允许停留在此导入边界）"));
    importTitle->setObjectName(QStringLiteral("OcrImportSectionTitle"));
    ocrPanel->bodyLayout()->addWidget(importTitle);
    ocrForm->addRow(uiText("Det 原始报告"), makePathRow(&customerOcrDetReportEdit_, uiText("PaddleOCR Det 官方 JSON")));
    customerOcrDetReportEdit_->setObjectName(QStringLiteral("OcrDetRawReportPath"));
    ocrForm->addRow(uiText("Det Snapshot"), makeIdPair(&customerOcrDetSnapshotIdEdit_,
        &customerOcrDetSnapshotArtifactIdEdit_, QStringLiteral("OcrDet")));
    ocrForm->addRow(uiText("Rec 原始报告"), makePathRow(&customerOcrRecReportEdit_, uiText("PaddleOCR Rec 官方 JSON（含 accuracy/CER）")));
    customerOcrRecReportEdit_->setObjectName(QStringLiteral("OcrRecRawReportPath"));
    ocrForm->addRow(uiText("Rec Snapshot"), makeIdPair(&customerOcrRecSnapshotIdEdit_,
        &customerOcrRecSnapshotArtifactIdEdit_, QStringLiteral("OcrRec")));
    ocrForm->addRow(uiText("System 原始报告"), makePathRow(&customerOcrSystemReportEdit_, uiText("PaddleOCR System 官方 JSON（必须含真实 accuracy）")));
    customerOcrSystemReportEdit_->setObjectName(QStringLiteral("OcrSystemRawReportPath"));
    ocrForm->addRow(uiText("System Snapshot"), makeIdPair(&customerOcrSystemSnapshotIdEdit_,
        &customerOcrSystemSnapshotArtifactIdEdit_, QStringLiteral("OcrSystem")));
    customerOcrCohortIdEdit_ = new QLineEdit;
    customerOcrCohortIdEdit_->setObjectName(QStringLiteral("OcrAcceptanceCohortId"));
    customerOcrCohortIdEdit_->setPlaceholderText(uiText("同一验收批次标识"));
    ocrForm->addRow(uiText("验收批次"), customerOcrCohortIdEdit_);
    customerOcrDomainIdEdit_ = new QLineEdit;
    customerOcrDomainIdEdit_->setObjectName(QStringLiteral("OcrCustomerDomainId"));
    customerOcrDomainIdEdit_->setPlaceholderText(uiText("客户域标识"));
    ocrForm->addRow(uiText("客户域"), customerOcrDomainIdEdit_);
    customerOcrEvidenceClassCombo_ = new QComboBox;
    customerOcrEvidenceClassCombo_->setObjectName(QStringLiteral("OcrEvidenceClass"));
    customerOcrEvidenceClassCombo_->addItems({QStringLiteral("customer_domain"),
        QStringLiteral("public"), QStringLiteral("generated"), QStringLiteral("smoke")});
    ocrForm->addRow(uiText("证据分类"), customerOcrEvidenceClassCombo_);
    auto* importOcrButton = primaryButton(uiText("受控导入官方报告"));
    importOcrButton->setObjectName(QStringLiteral("ImportOcrOfficialReportsButton"));
    connect(importOcrButton, &QPushButton::clicked, this, &MainWindow::importOcrOfficialReports);
    ocrForm->addRow(QString(), importOcrButton);

    auto* acceptanceTitle = new QLabel(uiText("步骤 2：仅使用已提交报告 ArtifactId 运行验收"));
    acceptanceTitle->setObjectName(QStringLiteral("OcrAcceptanceSectionTitle"));
    ocrForm->addRow(acceptanceTitle);
    customerOcrDetReportArtifactIdEdit_ = new QLineEdit;
    customerOcrDetReportArtifactIdEdit_->setObjectName(QStringLiteral("OcrDetReportArtifactId"));
    customerOcrRecReportArtifactIdEdit_ = new QLineEdit;
    customerOcrRecReportArtifactIdEdit_->setObjectName(QStringLiteral("OcrRecReportArtifactId"));
    customerOcrSystemReportArtifactIdEdit_ = new QLineEdit;
    customerOcrSystemReportArtifactIdEdit_->setObjectName(QStringLiteral("OcrSystemReportArtifactId"));
    ocrForm->addRow(uiText("Det 报告 ArtifactId"), customerOcrDetReportArtifactIdEdit_);
    ocrForm->addRow(uiText("Rec 报告 ArtifactId"), customerOcrRecReportArtifactIdEdit_);
    ocrForm->addRow(uiText("System 报告 ArtifactId"), customerOcrSystemReportArtifactIdEdit_);
    auto* thresholdRow = new QWidget;
    auto* thresholdLayout = new QHBoxLayout(thresholdRow);
    thresholdLayout->setContentsMargins(0, 0, 0, 0);
    thresholdLayout->setSpacing(8);
    customerOcrMinDetHmeanEdit_ = new QLineEdit(QStringLiteral("0.50"));
    customerOcrMinAccEdit_ = new QLineEdit(QStringLiteral("0.70"));
    customerOcrMaxCerEdit_ = new QLineEdit(QStringLiteral("0.30"));
    customerOcrMinSystemAccEdit_ = new QLineEdit(QStringLiteral("0.70"));
    thresholdLayout->addWidget(new QLabel(uiText("Det hmean ≥")));
    thresholdLayout->addWidget(customerOcrMinDetHmeanEdit_);
    thresholdLayout->addWidget(new QLabel(uiText("Rec accuracy >=")));
    thresholdLayout->addWidget(customerOcrMinAccEdit_);
    thresholdLayout->addWidget(new QLabel(uiText("CER <=")));
    thresholdLayout->addWidget(customerOcrMaxCerEdit_);
    thresholdLayout->addWidget(new QLabel(uiText("System accuracy ≥")));
    thresholdLayout->addWidget(customerOcrMinSystemAccEdit_);
    ocrForm->addRow(uiText("门槛"), thresholdRow);
    ocrPanel->bodyLayout()->addLayout(ocrForm);
    customerOcrStatusLabel_ = inlineStatusLabel(uiText("尚未导入官方报告或运行 OCR Acceptance 。"));
    customerOcrStatusLabel_->setObjectName(QStringLiteral("OcrAcceptanceStatus"));
    ocrPanel->bodyLayout()->addWidget(customerOcrStatusLabel_);
    auto* runOcrButton = primaryButton(uiText("运行 OCR Acceptance "));
    runOcrButton->setObjectName(QStringLiteral("RunOcrAcceptanceWorkflowButton"));
    connect(runOcrButton, &QPushButton::clicked, this, &MainWindow::runOcrAcceptanceWorkflow);
    ocrPanel->bodyLayout()->addWidget(runOcrButton, 0, Qt::AlignRight);
    ocrPanel->bodyLayout()->addWidget(mutedLabel(uiText("Total-Text、generated smoke 和 .deps 示例只能证明流程可跑，不能作为客户域生产 OCR 精度证明。")));

    auto* diagnosticsPanel = new InfoPanel(uiText("诊断包"));
    diagnosticsStatusLabel_ = inlineStatusLabel(uiText("诊断包尚未生成。"));
    diagnosticsPanel->bodyLayout()->addWidget(diagnosticsStatusLabel_);
    diagnosticsPanel->bodyLayout()->addWidget(mutedLabel(uiText("诊断包包含 Worker self-check、环境 profile、GPU/驱动、最近任务日志、失败请求、artifact index、内置能力状态和授权摘要。")));
    auto* diagnosticsPanelButton = primaryButton(uiText("一键诊断包"));
    connect(diagnosticsPanelButton, &QPushButton::clicked, this, &MainWindow::collectDiagnosticsBundle);
    diagnosticsPanel->bodyLayout()->addWidget(diagnosticsPanelButton, 0, Qt::AlignRight);

    rightLayout->addWidget(ocrPanel, 4);
    rightLayout->addWidget(diagnosticsPanel, 1);

    splitter->addWidget(leftStack);
    splitter->addWidget(rightStack);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 5);
    splitter->setSizes(QList<int>() << 560 << 680);

    layout->addWidget(splitter, 1);
    page->setWidget(content);
    return page;
}
