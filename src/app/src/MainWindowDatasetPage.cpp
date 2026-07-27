#include "MainWindow.h"

#include "DatasetPage.h"
#include "DatasetPageController.h"
#include "DatasetConversionUiModel.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
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
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildDatasetPage()
{
    datasetPage_ = new DatasetWorkspacePage;
    auto* page = datasetPage_;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerValidateButton = primaryButton(QStringLiteral("运行质量工作流"));
    connect(headerValidateButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::runDataQuality);

    auto* inputPanel = new InfoPanel(QStringLiteral("数据集操作"));
    auto* form = new QFormLayout;
    datasetPage_->datasetPathEdit = new QLineEdit;
    datasetPage_->datasetPathEdit->setObjectName(QStringLiteral("DatasetPathEdit"));
    auto* browseButton = new QPushButton(QStringLiteral("选择数据集"));
    connect(browseButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::browseDataset);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(datasetPage_->datasetPathEdit);
    pathLayout->addWidget(browseButton);

    datasetPage_->datasetProbeStatusLabel = mutedLabel(QStringLiteral(
        "格式探测在后台执行；最终格式由 Worker/Core 完整校验。"));
    datasetPage_->datasetProbeStatusLabel->setObjectName(QStringLiteral("DatasetProbeStatus"));
    allowLabelToShrink(datasetPage_->datasetProbeStatusLabel);

    datasetPage_->datasetFormatCombo = new QComboBox;
    {
        const QSignalBlocker blocker(datasetPage_->datasetFormatCombo);
        QStringList formats;
        for (const aitrain::CapabilityDescriptor& capability
            : aitrain::BuiltinCapabilityRegistry::instance().capabilities()) {
            for (const QString& format : capability.datasetFormats) {
                if (!formats.contains(format)) {
                    formats.append(format);
                }
            }
        }
        for (const QString& format : formats) {
            datasetPage_->datasetFormatCombo->addItem(datasetFormatLabel(format), format);
        }
    }
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::runDataQuality);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QString(), datasetPage_->datasetProbeStatusLabel);
    form->addRow(QStringLiteral("格式"), datasetPage_->datasetFormatCombo);

    datasetPage_->dataQualityDatasetIdEdit = new QLineEdit;
    datasetPage_->dataQualityDatasetIdEdit->setObjectName(QStringLiteral("DataQualityDatasetId"));
    datasetPage_->dataQualityDatasetIdEdit->setPlaceholderText(QStringLiteral("已登记 DatasetId"));
    datasetPage_->dataQualityDatasetVersionIdEdit = new QLineEdit;
    datasetPage_->dataQualityDatasetVersionIdEdit->setObjectName(QStringLiteral("DataQualityDatasetVersionId"));
    datasetPage_->dataQualityDatasetVersionIdEdit->setPlaceholderText(QStringLiteral("已登记 DatasetVersionId"));
    datasetPage_->dataQualitySnapshotIdEdit = new QLineEdit;
    datasetPage_->dataQualitySnapshotIdEdit->setObjectName(QStringLiteral("DataQualitySnapshotId"));
    datasetPage_->dataQualitySnapshotIdEdit->setPlaceholderText(QStringLiteral("已登记 SnapshotId"));
    datasetPage_->dataQualitySnapshotArtifactIdEdit = new QLineEdit;
    datasetPage_->dataQualitySnapshotArtifactIdEdit->setObjectName(QStringLiteral("DataQualitySnapshotArtifactId"));
    datasetPage_->dataQualitySnapshotArtifactIdEdit->setPlaceholderText(QStringLiteral("committed Snapshot ArtifactId"));
    form->addRow(QStringLiteral("质量 DatasetId"), datasetPage_->dataQualityDatasetIdEdit);
    form->addRow(QStringLiteral("质量 VersionId"), datasetPage_->dataQualityDatasetVersionIdEdit);
    form->addRow(QStringLiteral("质量 SnapshotId"), datasetPage_->dataQualitySnapshotIdEdit);
    form->addRow(QStringLiteral("质量 Snapshot ArtifactId"), datasetPage_->dataQualitySnapshotArtifactIdEdit);

    datasetPage_->splitSourceDatasetIdEdit = new QLineEdit;
    datasetPage_->splitSourceDatasetIdEdit->setObjectName(QStringLiteral("SplitSourceDatasetId"));
    datasetPage_->splitSourceDatasetIdEdit->setPlaceholderText(QStringLiteral("源 DatasetId"));
    datasetPage_->splitSourceDatasetVersionIdEdit = new QLineEdit;
    datasetPage_->splitSourceDatasetVersionIdEdit->setObjectName(QStringLiteral("SplitSourceDatasetVersionId"));
    datasetPage_->splitSourceDatasetVersionIdEdit->setPlaceholderText(QStringLiteral("源 DatasetVersionId"));
    datasetPage_->splitSourceSnapshotIdEdit = new QLineEdit;
    datasetPage_->splitSourceSnapshotIdEdit->setObjectName(QStringLiteral("SplitSourceSnapshotId"));
    datasetPage_->splitSourceSnapshotIdEdit->setPlaceholderText(QStringLiteral("源 SnapshotId"));
    datasetPage_->splitSourceSnapshotArtifactIdEdit = new QLineEdit;
    datasetPage_->splitSourceSnapshotArtifactIdEdit->setObjectName(QStringLiteral("SplitSourceSnapshotArtifactId"));
    datasetPage_->splitSourceSnapshotArtifactIdEdit->setPlaceholderText(QStringLiteral("committed Snapshot ArtifactId"));
    datasetPage_->splitTargetDatasetIdEdit = new QLineEdit(aitrain::DatasetId::create().toString());
    datasetPage_->splitTargetDatasetIdEdit->setObjectName(QStringLiteral("SplitTargetDatasetId"));
    datasetPage_->splitTargetDatasetNameEdit = new QLineEdit;
    datasetPage_->splitTargetDatasetNameEdit->setObjectName(QStringLiteral("SplitTargetDatasetName"));
    datasetPage_->splitTargetDatasetNameEdit->setPlaceholderText(QStringLiteral("审计名称（当前 schema 不持久化名称）"));
    datasetPage_->splitTrainRatioEdit = new QLineEdit(QStringLiteral("0.8"));
    datasetPage_->splitValRatioEdit = new QLineEdit(QStringLiteral("0.2"));
    datasetPage_->splitTestRatioEdit = new QLineEdit(QStringLiteral("0.0"));
    datasetPage_->splitSeedEdit = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::runSplit);
    auto* curateButton = new QPushButton(QStringLiteral("运行 Data Quality "));
    curateButton->setObjectName(QStringLiteral("RunDataQualityWorkflowButton"));
    connect(curateButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::runDataQuality);
    auto* snapshotButton = new QPushButton(QStringLiteral("创建数据快照"));
    snapshotButton->setObjectName(QStringLiteral("RunDatasetSnapshotImportWorkflowButton"));
    connect(snapshotButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::runSnapshotImport);
    datasetPage_->datasetSnapshotTargetDatasetIdEdit = new QLineEdit(aitrain::DatasetId::create().toString());
    datasetPage_->datasetSnapshotTargetDatasetIdEdit->setObjectName(QStringLiteral("DatasetSnapshotTargetDatasetId"));
    datasetPage_->datasetSnapshotTargetDatasetIdEdit->setPlaceholderText(QStringLiteral("新 DatasetId，或已有同格式 DatasetId"));
    datasetPage_->datasetSnapshotTargetDatasetNameEdit = new QLineEdit;
    datasetPage_->datasetSnapshotTargetDatasetNameEdit->setObjectName(QStringLiteral("DatasetSnapshotTargetDatasetName"));
    datasetPage_->datasetSnapshotTargetDatasetNameEdit->setPlaceholderText(QStringLiteral("审计名称（当前 schema 不持久化名称）"));
    auto* openQualityReportButton = new QPushButton(QStringLiteral("打开质量报告"));
    connect(openQualityReportButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityReport);
    auto* openFixListButton = new QPushButton(QStringLiteral("打开问题清单"));
    connect(openFixListButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityFixList);
    auto* fixWithXAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling 修复"));
    connect(fixWithXAnyButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::createAnnotationSession);
    auto* ratioRow = new QWidget;
    auto* ratioLayout = new QHBoxLayout(ratioRow);
    ratioLayout->setContentsMargins(0, 0, 0, 0);
    ratioLayout->addWidget(datasetPage_->splitTrainRatioEdit);
    ratioLayout->addWidget(datasetPage_->splitValRatioEdit);
    ratioLayout->addWidget(datasetPage_->splitTestRatioEdit);
    ratioLayout->addWidget(datasetPage_->splitSeedEdit);
    form->addRow(QStringLiteral("划分源 DatasetId"), datasetPage_->splitSourceDatasetIdEdit);
    form->addRow(QStringLiteral("划分源 VersionId"), datasetPage_->splitSourceDatasetVersionIdEdit);
    form->addRow(QStringLiteral("划分源 SnapshotId"), datasetPage_->splitSourceSnapshotIdEdit);
    form->addRow(QStringLiteral("划分源 ArtifactId"), datasetPage_->splitSourceSnapshotArtifactIdEdit);
    form->addRow(QStringLiteral("划分目标 DatasetId"), datasetPage_->splitTargetDatasetIdEdit);
    form->addRow(QStringLiteral("划分审计名称"), datasetPage_->splitTargetDatasetNameEdit);
    form->addRow(QStringLiteral("train / val / test / seed"), ratioRow);
    form->addRow(QStringLiteral("快照目标 DatasetId"), datasetPage_->datasetSnapshotTargetDatasetIdEdit);
    form->addRow(QStringLiteral("快照审计名称"), datasetPage_->datasetSnapshotTargetDatasetNameEdit);
    inputPanel->bodyLayout()->addLayout(form);
    auto* datasetActionStrip = new QFrame;
    datasetActionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* datasetActionGrid = new QGridLayout(datasetActionStrip);
    datasetActionGrid->setContentsMargins(10, 8, 10, 8);
    datasetActionGrid->setHorizontalSpacing(10);
    datasetActionGrid->setVerticalSpacing(8);
    datasetActionGrid->addWidget(validateButton, 0, 0);
    datasetActionGrid->addWidget(splitButton, 0, 1);
    datasetActionGrid->addWidget(snapshotButton, 0, 2);
    datasetActionGrid->addWidget(curateButton, 1, 0);
    datasetActionGrid->addWidget(openQualityReportButton, 1, 1);
    datasetActionGrid->addWidget(openFixListButton, 1, 2);
    datasetActionGrid->addWidget(fixWithXAnyButton, 2, 0, 1, 3);
    for (int column = 0; column < 3; ++column) {
        datasetActionGrid->setColumnStretch(column, 1);
    }
    inputPanel->bodyLayout()->addWidget(datasetActionStrip);

    auto* conversionStrip = new QFrame;
    conversionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* conversionLayout = new QVBoxLayout(conversionStrip);
    conversionLayout->setContentsMargins(10, 10, 10, 10);
    conversionLayout->setSpacing(8);

    auto* conversionTitle = new QLabel(QStringLiteral("格式转换"));
    conversionTitle->setObjectName(QStringLiteral("SectionTitle"));
    datasetPage_->datasetConversionStatusLabel = mutedLabel(QStringLiteral("选择外部源、目标格式和目标 Dataset 身份后开始受控转换。"));
    allowLabelToShrink(datasetPage_->datasetConversionStatusLabel);
    conversionLayout->addWidget(conversionTitle);
    conversionLayout->addWidget(datasetPage_->datasetConversionStatusLabel);

    datasetPage_->datasetConversionSourceFormatCombo = new QComboBox;
    for (const QString& format : supportedDatasetConversionSourceFormats()) {
        addComboItem(datasetPage_->datasetConversionSourceFormatCombo, datasetConversionFormatLabel(format), format);
    }
    datasetPage_->datasetConversionTargetFormatCombo = new QComboBox;
    for (const QString& target : supportedDatasetConversionTargets(comboCurrentDataOrText(datasetPage_->datasetConversionSourceFormatCombo))) {
        addComboItem(datasetPage_->datasetConversionTargetFormatCombo, datasetConversionFormatLabel(target), target);
    }
    if (datasetPage_->datasetConversionTargetFormatCombo->count() > 0) {
        datasetPage_->datasetConversionTargetFormatCombo->setCurrentIndex(0);
    }
    connect(datasetPage_->datasetConversionSourceFormatCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        datasetPageController_, &DatasetPageController::updateConversionTargets);

    datasetPage_->datasetConversionInputEdit = new QLineEdit;
    datasetPage_->datasetConversionBrowseInputButton = new QPushButton(QStringLiteral("选择输入"));
    connect(datasetPage_->datasetConversionBrowseInputButton,
        &QPushButton::clicked, datasetPageController_,
        &DatasetPageController::browseConversionInput);
    auto* conversionInputRow = new QWidget;
    auto* conversionInputLayout = new QHBoxLayout(conversionInputRow);
    conversionInputLayout->setContentsMargins(0, 0, 0, 0);
    conversionInputLayout->setSpacing(8);
    conversionInputLayout->addWidget(datasetPage_->datasetConversionInputEdit, 1);
    conversionInputLayout->addWidget(datasetPage_->datasetConversionBrowseInputButton);
    datasetPage_->datasetConversionProbeStatusLabel = mutedLabel(QStringLiteral(
        "选择目录后将在后台探测源格式；Worker 转换前会重新校验。"));
    datasetPage_->datasetConversionProbeStatusLabel->setObjectName(QStringLiteral("DatasetConversionProbeStatus"));
    allowLabelToShrink(datasetPage_->datasetConversionProbeStatusLabel);

    datasetPage_->datasetConversionTargetDatasetIdEdit = new QLineEdit(
        aitrain::DatasetId::create().toString());
    datasetPage_->datasetConversionTargetDatasetIdEdit->setObjectName(QStringLiteral("DatasetConversionTargetDatasetId"));
    datasetPage_->datasetConversionTargetDatasetNameEdit = new QLineEdit;
    datasetPage_->datasetConversionTargetDatasetNameEdit->setObjectName(QStringLiteral("DatasetConversionTargetDatasetName"));
    datasetPage_->datasetConversionTargetDatasetNameEdit->setPlaceholderText(QStringLiteral("仅作审计显示；名称尚无独立 Storage 字段"));

    auto* conversionForm = new QFormLayout;
    conversionForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    conversionForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    conversionForm->setHorizontalSpacing(12);
    conversionForm->setVerticalSpacing(6);
    conversionForm->addRow(QStringLiteral("源格式"), datasetPage_->datasetConversionSourceFormatCombo);
    datasetPage_->datasetConversionSourceErrorLabel = new QLabel;
    datasetPage_->datasetConversionSourceErrorLabel->setObjectName(QStringLiteral("FieldErrorText"));
    datasetPage_->datasetConversionSourceErrorLabel->hide();
    conversionForm->addRow(QString(), datasetPage_->datasetConversionSourceErrorLabel);
    conversionForm->addRow(QStringLiteral("目标格式"), datasetPage_->datasetConversionTargetFormatCombo);
    datasetPage_->datasetConversionTargetErrorLabel = new QLabel;
    datasetPage_->datasetConversionTargetErrorLabel->setObjectName(QStringLiteral("FieldErrorText"));
    datasetPage_->datasetConversionTargetErrorLabel->hide();
    conversionForm->addRow(QString(), datasetPage_->datasetConversionTargetErrorLabel);
    conversionForm->addRow(QStringLiteral("输入路径"), conversionInputRow);
    conversionForm->addRow(QString(), datasetPage_->datasetConversionProbeStatusLabel);
    datasetPage_->datasetConversionInputErrorLabel = new QLabel;
    datasetPage_->datasetConversionInputErrorLabel->setObjectName(QStringLiteral("FieldErrorText"));
    datasetPage_->datasetConversionInputErrorLabel->hide();
    conversionForm->addRow(QString(), datasetPage_->datasetConversionInputErrorLabel);
    conversionForm->addRow(QStringLiteral("目标 DatasetId"), datasetPage_->datasetConversionTargetDatasetIdEdit);
    conversionForm->addRow(QStringLiteral("目标名称（审计）"), datasetPage_->datasetConversionTargetDatasetNameEdit);
    conversionLayout->addLayout(conversionForm);

    auto* conversionActionRow = new QWidget;
    auto* conversionActionLayout = new QHBoxLayout(conversionActionRow);
    conversionActionLayout->setContentsMargins(0, 0, 0, 0);
    conversionActionLayout->setSpacing(8);
    datasetPage_->datasetConversionStartButton = primaryButton(QStringLiteral("转换数据集"));
    datasetPage_->datasetConversionStartButton->setObjectName(QStringLiteral("RunDatasetConversionWorkflowButton"));
    datasetPage_->datasetConversionCancelButton = dangerButton(QStringLiteral("取消转换"));
    datasetPage_->datasetConversionCancelButton->setEnabled(false);
    connect(datasetPage_->datasetConversionStartButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::startConversion);
    connect(datasetPage_->datasetConversionCancelButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::cancelConversion);
    conversionActionLayout->addStretch();
    conversionActionLayout->addWidget(datasetPage_->datasetConversionStartButton);
    conversionActionLayout->addWidget(datasetPage_->datasetConversionCancelButton);
    conversionLayout->addWidget(conversionActionRow);

    datasetPage_->datasetConversionProgressBar = new QProgressBar;
    datasetPage_->datasetConversionProgressBar->setRange(0, 100);
    datasetPage_->datasetConversionProgressBar->setValue(0);
    datasetPage_->datasetConversionResultLabel = inlineStatusLabel(QStringLiteral("转换结果会显示在这里。"));
    allowLabelToShrink(datasetPage_->datasetConversionResultLabel);
    datasetPage_->datasetConversionLog = new QPlainTextEdit;
    datasetPage_->datasetConversionLog->setReadOnly(true);
    datasetPage_->datasetConversionLog->setMinimumHeight(96);
    datasetPage_->datasetConversionLog->setPlainText(QStringLiteral("等待转换。"));
    conversionLayout->addWidget(datasetPage_->datasetConversionProgressBar);
    conversionLayout->addWidget(datasetPage_->datasetConversionResultLabel);
    conversionLayout->addWidget(datasetPage_->datasetConversionLog);
    inputPanel->bodyLayout()->addWidget(conversionStrip);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("所选数据集详情"));
    datasetPage_->datasetDetailLabel = inlineStatusLabel(QStringLiteral("选择或导入数据集后显示格式、样本数、校验状态和最近报告。"));
    datasetPage_->datasetDetailLabel->setObjectName(QStringLiteral("DatasetDetailLabel"));
    datasetPage_->validationSummaryLabel = mutedLabel(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    allowLabelToShrink(datasetPage_->datasetDetailLabel);
    allowLabelToShrink(datasetPage_->validationSummaryLabel);
    datasetPage_->datasetRepairLoopLabel = inlineStatusLabel(QStringLiteral("修复闭环：等待质量报告。"));
    allowLabelToShrink(datasetPage_->datasetRepairLoopLabel);
    datasetPage_->datasetRepairLoopTable = new QTableWidget(0, 3);
    datasetPage_->datasetRepairLoopTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("环节")
        << QStringLiteral("状态")
        << QStringLiteral("下一步"));
    configureTable(datasetPage_->datasetRepairLoopTable);
    datasetPage_->datasetRepairLoopTable->setWordWrap(true);
    datasetPage_->datasetRepairLoopTable->verticalHeader()->setDefaultSectionSize(34);
    datasetPage_->datasetRepairLoopTable->setMaximumHeight(156);
    datasetPage_->datasetRepairLoopTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPage_->datasetRepairLoopTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetPage_->datasetRepairLoopTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    datasetPage_->validationIssuesTable = new QTableWidget(0, 5);
    datasetPage_->validationIssuesTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("级别")
        << QStringLiteral("代码")
        << QStringLiteral("文件")
        << QStringLiteral("行号")
        << QStringLiteral("说明"));
    configureTable(datasetPage_->validationIssuesTable);
    datasetPage_->validationIssuesTable->verticalHeader()->setDefaultSectionSize(38);
    datasetPage_->validationIssuesTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPage_->validationIssuesTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetPage_->validationIssuesTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    datasetPage_->validationIssuesTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    datasetPage_->validationIssuesTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    datasetPage_->validationOutput = new QPlainTextEdit;
    datasetPage_->validationOutput->setReadOnly(true);
    datasetPage_->validationOutput->setMinimumHeight(130);
    datasetPage_->validationOutput->setPlainText(QStringLiteral("校验报告 JSON 会显示在这里。"));
    resultPanel->bodyLayout()->addWidget(datasetPage_->datasetDetailLabel);
    resultPanel->bodyLayout()->addWidget(datasetPage_->validationSummaryLabel);
    resultPanel->bodyLayout()->addWidget(datasetPage_->datasetRepairLoopLabel);
    resultPanel->bodyLayout()->addWidget(datasetPage_->datasetRepairLoopTable);
    resultPanel->bodyLayout()->addWidget(datasetPage_->validationIssuesTable, 2);
    resultPanel->bodyLayout()->addWidget(datasetPage_->validationOutput);

    auto* toolsPanel = new InfoPanel(QStringLiteral("数据集库与样本预览"));
    datasetPage_->datasetListTable = new QTableWidget(0, 5);
    datasetPage_->datasetListTable->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("数据集")
        << QStringLiteral("格式")
        << QStringLiteral("状态")
        << QStringLiteral("样本")
        << QStringLiteral("快照身份"));
    configureTable(datasetPage_->datasetListTable);
    datasetPage_->datasetListTable->setWordWrap(true);
    datasetPage_->datasetListTable->verticalHeader()->setDefaultSectionSize(40);
    datasetPage_->datasetListTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPage_->datasetListTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetPage_->datasetListTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    datasetPage_->datasetListTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    datasetPage_->datasetListTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    datasetPage_->datasetPreviewTable = new QTableWidget(0, 2);
    datasetPage_->datasetPreviewTable->setHorizontalHeaderLabels(QStringList() << QStringLiteral("样本") << QStringLiteral("标签 / 说明"));
    configureTable(datasetPage_->datasetPreviewTable);
    datasetPage_->datasetPreviewTable->setWordWrap(true);
    datasetPage_->datasetPreviewTable->verticalHeader()->setDefaultSectionSize(40);
    datasetPage_->datasetPreviewTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPage_->datasetPreviewTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    auto* annotationPanel = new QGroupBox(QStringLiteral("外部标注工具"));
    annotationPanel->setMinimumHeight(136);
    annotationPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    auto* annotationLayout = new QVBoxLayout(annotationPanel);
    annotationLayout->setContentsMargins(10, 12, 10, 8);
    annotationLayout->setSpacing(6);
    auto* annotationSummary = mutedLabel(uiText("X-AnyLabeling：检测导出 YOLO bbox，分割导出 YOLO polygon；PaddleOCR 使用 det_gt / rec_gt + dict。"));
    datasetPage_->annotationToolStatusLabel = inlineStatusLabel(xAnyLabelingStatusText());
    allowLabelToShrink(annotationSummary);
    allowLabelToShrink(datasetPage_->annotationToolStatusLabel);
    annotationSummary->setMinimumHeight(28);
    datasetPage_->annotationToolStatusLabel->setMinimumHeight(28);
    datasetPage_->annotationToolStatusLabel->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    auto* createAnnotationSessionButton = new QPushButton(QStringLiteral("准备修复会话"));
    auto* syncAnnotationSessionButton = new QPushButton(QStringLiteral("同步标注会话"));
    createAnnotationSessionButton->setObjectName(QStringLiteral("CreateAnnotationSessionButton"));
    syncAnnotationSessionButton->setObjectName(QStringLiteral("SyncAnnotationSessionButton"));
    auto* refreshAnnotationStatusButton = new QPushButton(QStringLiteral("检测状态"));
    connect(refreshAnnotationStatusButton, &QPushButton::clicked, this, &MainWindow::updateAnnotationToolStatus);
    connect(createAnnotationSessionButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::createAnnotationSession);
    connect(syncAnnotationSessionButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::syncAnnotationSession);
    auto* annotationBoundaryHint = mutedLabel(uiText(
        "原始数据目录不会由 GUI 直接打开。修复必须先创建 Annotation Session，完成后由 Worker 按 Session ArtifactId 和受控工作目录同步。"));
    allowLabelToShrink(annotationBoundaryHint);
    auto* annotationActionRow = new QWidget;
    auto* annotationActionGrid = new QGridLayout(annotationActionRow);
    annotationActionGrid->setContentsMargins(0, 0, 0, 0);
    annotationActionGrid->setHorizontalSpacing(10);
    annotationActionGrid->setVerticalSpacing(8);
    annotationActionGrid->addWidget(createAnnotationSessionButton, 0, 0);
    annotationActionGrid->addWidget(syncAnnotationSessionButton, 0, 1);
    annotationActionGrid->addWidget(refreshAnnotationStatusButton, 0, 2);
    for (int column = 0; column < 3; ++column) {
        annotationActionGrid->setColumnStretch(column, 1);
    }
    annotationLayout->addWidget(annotationSummary);
    annotationLayout->addWidget(datasetPage_->annotationToolStatusLabel);
    annotationLayout->addWidget(annotationBoundaryHint);
    annotationLayout->addWidget(annotationActionRow);
    auto* datasetLibraryTab = new QWidget;
    auto* datasetLibraryLayout = new QVBoxLayout(datasetLibraryTab);
    datasetLibraryLayout->setContentsMargins(0, 0, 0, 0);
    datasetLibraryLayout->setSpacing(10);
    auto* datasetLibraryCaption = mutedLabel(QStringLiteral("已登记数据集"));
    allowLabelToShrink(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetPage_->datasetListTable, 1);
    datasetLibraryLayout->addWidget(annotationPanel);

    auto* samplePreviewTab = new QWidget;
    auto* samplePreviewLayout = new QVBoxLayout(samplePreviewTab);
    samplePreviewLayout->setContentsMargins(0, 0, 0, 0);
    samplePreviewLayout->setSpacing(10);
    auto* samplePreviewHint = mutedLabel(uiText("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、YOLO OBB、语义分割 Mask PNG、PaddleOCR Det 和 PaddleOCR Rec。"));
    allowLabelToShrink(samplePreviewHint);
    samplePreviewLayout->addWidget(datasetPage_->datasetPreviewTable, 1);
    samplePreviewLayout->addWidget(samplePreviewHint);

    auto* datasetToolsTabs = new QTabWidget;
    datasetToolsTabs->setObjectName(QStringLiteral("DatasetToolsTabs"));
    datasetToolsTabs->addTab(datasetLibraryTab, uiText("数据集库"));
    datasetToolsTabs->addTab(samplePreviewTab, uiText("样本预览"));
    toolsPanel->bodyLayout()->addWidget(datasetToolsTabs, 1);

    splitter->addWidget(toolsPanel);
    splitter->addWidget(resultPanel);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 3);
    splitter->setSizes(QList<int>() << 480 << 620);

    auto* preparationTab = new QWidget;
    auto* preparationLayout = new QVBoxLayout(preparationTab);
    preparationLayout->setContentsMargins(0, 0, 0, 0);
    preparationLayout->setSpacing(16);
    preparationLayout->addWidget(inputPanel);
    preparationLayout->addWidget(splitter, 1);

    datasetPage_->tabs = new QTabWidget;
    datasetPage_->tabs->setObjectName(QStringLiteral("DatasetTabs"));
    datasetPage_->tabs->addTab(preparationTab, uiText("数据集准备"));
    datasetPage_->tabs->addTab(buildSampleReviewPanel(), uiText("质量与复核"));

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("DATASET VALIDATION"),
        uiText("数据集工作台"),
        uiText("导入、校验、转换、快照，并处理质量复核样本。"),
        headerValidateButton,
        QStringList()
            << QStringLiteral("YOLO BBox")
            << QStringLiteral("YOLO Polygon")
            << QStringLiteral("Mask PNG")
            << QStringLiteral("PaddleOCR Det")
            << QStringLiteral("PaddleOCR Rec")));
    layout->addWidget(datasetPage_->tabs, 1);
    page->setWidget(content);
    datasetPageController_->attach(datasetPage_);
    return page;
}

QWidget* MainWindow::buildSampleReviewPanel()
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

    auto* setupPanel = new InfoPanel(uiText("输入与过滤"));
    setupPanel->setMinimumWidth(280);
    setupPanel->setMaximumWidth(340);
    datasetPage_->reviewSamplePathEdit = new QLineEdit;
    datasetPage_->reviewSamplePathEdit->setPlaceholderText(uiText("输入已提交的 dataset_quality_analysis / dataset_repair_manifest ArtifactId"));
    auto* browseButton = new QPushButton(uiText("选择 Artifact"));
    connect(browseButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::browseSampleReview);
    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->setSpacing(8);
    pathLayout->addWidget(datasetPage_->reviewSamplePathEdit, 1);
    pathLayout->addWidget(browseButton);

    datasetPage_->reviewSourceFilterCombo = new QComboBox;
    datasetPage_->reviewSourceFilterCombo->addItem(uiText("全部来源"), QString());
    datasetPage_->reviewReasonFilterCombo = new QComboBox;
    datasetPage_->reviewReasonFilterCombo->addItem(uiText("全部问题"), QString());
    datasetPage_->reviewSearchEdit = new QLineEdit;
    datasetPage_->reviewSearchEdit->setPlaceholderText(uiText("按图片、标签、类别、说明搜索"));
    connect(datasetPage_->reviewSourceFilterCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        datasetPageController_, &DatasetPageController::refreshSampleReview);
    connect(datasetPage_->reviewReasonFilterCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        datasetPageController_, &DatasetPageController::refreshSampleReview);
    connect(datasetPage_->reviewSearchEdit, &QLineEdit::textChanged,
        datasetPageController_, &DatasetPageController::refreshSampleReview);

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(10);
    form->addRow(uiText("复核 Artifact"), pathRow);
    form->addRow(uiText("来源"), datasetPage_->reviewSourceFilterCombo);
    form->addRow(uiText("问题类型"), datasetPage_->reviewReasonFilterCombo);
    form->addRow(uiText("搜索"), datasetPage_->reviewSearchEdit);
    setupPanel->bodyLayout()->addLayout(form);

    auto* reviewWorkflowHint = mutedLabel(uiText("标注完成后返回数据集准备页重新校验并创建快照。"));
    allowLabelToShrink(reviewWorkflowHint);
    setupPanel->bodyLayout()->addWidget(reviewWorkflowHint);

    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(8);
    auto* loadButton = primaryButton(uiText("加载复核样本"));
    auto* openButton = new QPushButton(uiText("查看受控信息"));
    connect(loadButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::loadSampleReview);
    connect(openButton, &QPushButton::clicked,
        datasetPageController_, &DatasetPageController::openSelectedReviewSample);
    actionLayout->addWidget(loadButton, 0, 0);
    actionLayout->addWidget(openButton, 0, 1);
    actionLayout->setColumnStretch(0, 1);
    actionLayout->setColumnStretch(1, 1);
    setupPanel->bodyLayout()->addWidget(actionStrip);

    datasetPage_->sampleReviewSummaryLabel = inlineStatusLabel(uiText("尚未加载复核样本。"));
    setupPanel->bodyLayout()->addWidget(datasetPage_->sampleReviewSummaryLabel);
    setupPanel->bodyLayout()->addWidget(emptyStateLabel(uiText("样本复核页只读展示已加载清单；修复必须通过 Data Quality 与 Annotation Session。")));
    setupPanel->bodyLayout()->addStretch();

    auto* tablePanel = new InfoPanel(uiText("复核队列"));
    datasetPage_->sampleReviewTable = new QTableWidget(0, 7);
    datasetPage_->sampleReviewTable->setHorizontalHeaderLabels(QStringList()
        << uiText("来源")
        << uiText("问题")
        << uiText("类别")
        << uiText("指标")
        << uiText("图片")
        << uiText("标签")
        << uiText("说明"));
    configureTable(datasetPage_->sampleReviewTable);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    datasetPage_->sampleReviewTable->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Stretch);
    datasetPage_->sampleReviewTable->setMinimumHeight(420);
    tablePanel->bodyLayout()->addWidget(datasetPage_->sampleReviewTable);

    splitter->addWidget(setupPanel);
    splitter->addWidget(tablePanel);
    splitter->setChildrenCollapsible(false);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setSizes(QList<int>() << 320 << 820);

    layout->addWidget(splitter, 1);
    page->setWidget(content);
    return page;
}
