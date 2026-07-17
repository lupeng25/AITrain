#include "MainWindow.h"

#include "DatasetConversionUiModel.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"

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
#include <QSignalBlocker>
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

QWidget* MainWindow::buildDatasetPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerValidateButton = primaryButton(QStringLiteral("运行质量工作流"));
    connect(headerValidateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);

    auto* inputPanel = new InfoPanel(QStringLiteral("数据集操作"));
    auto* form = new QFormLayout;
    datasetPathEdit_ = new QLineEdit;
    auto* browseButton = new QPushButton(QStringLiteral("选择数据集"));
    connect(browseButton, &QPushButton::clicked, this, &MainWindow::browseDataset);

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(datasetPathEdit_);
    pathLayout->addWidget(browseButton);

    datasetFormatCombo_ = new QComboBox;
    {
        const QSignalBlocker blocker(datasetFormatCombo_);
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
            datasetFormatCombo_->addItem(datasetFormatLabel(format), format);
        }
    }
    connect(datasetFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        state_.dataset.currentFormat = currentDatasetFormat();
        state_.dataset.currentDatasetId.clear();
        state_.dataset.currentDatasetVersionId.clear();
        state_.dataset.currentSnapshotId.clear();
        state_.dataset.currentSnapshotArtifactId.clear();
        state_.dataset.currentValid = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
        refreshDatasetConversionDefaultsFromCurrentDataset();
    });
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QStringLiteral("格式"), datasetFormatCombo_);

    dataQualityDatasetIdEdit_ = new QLineEdit;
    dataQualityDatasetIdEdit_->setObjectName(QStringLiteral("DataQualityDatasetId"));
    dataQualityDatasetIdEdit_->setPlaceholderText(QStringLiteral("已登记 DatasetId"));
    dataQualityDatasetVersionIdEdit_ = new QLineEdit;
    dataQualityDatasetVersionIdEdit_->setObjectName(QStringLiteral("DataQualityDatasetVersionId"));
    dataQualityDatasetVersionIdEdit_->setPlaceholderText(QStringLiteral("已登记 DatasetVersionId"));
    dataQualitySnapshotIdEdit_ = new QLineEdit;
    dataQualitySnapshotIdEdit_->setObjectName(QStringLiteral("DataQualitySnapshotId"));
    dataQualitySnapshotIdEdit_->setPlaceholderText(QStringLiteral("已登记 SnapshotId"));
    dataQualitySnapshotArtifactIdEdit_ = new QLineEdit;
    dataQualitySnapshotArtifactIdEdit_->setObjectName(QStringLiteral("DataQualitySnapshotArtifactId"));
    dataQualitySnapshotArtifactIdEdit_->setPlaceholderText(QStringLiteral("committed Snapshot ArtifactId"));
    form->addRow(QStringLiteral("质量 DatasetId"), dataQualityDatasetIdEdit_);
    form->addRow(QStringLiteral("质量 VersionId"), dataQualityDatasetVersionIdEdit_);
    form->addRow(QStringLiteral("质量 SnapshotId"), dataQualitySnapshotIdEdit_);
    form->addRow(QStringLiteral("质量 Snapshot ArtifactId"), dataQualitySnapshotArtifactIdEdit_);

    splitSourceDatasetIdEdit_ = new QLineEdit;
    splitSourceDatasetIdEdit_->setObjectName(QStringLiteral("SplitSourceDatasetId"));
    splitSourceDatasetIdEdit_->setPlaceholderText(QStringLiteral("源 DatasetId"));
    splitSourceDatasetVersionIdEdit_ = new QLineEdit;
    splitSourceDatasetVersionIdEdit_->setObjectName(QStringLiteral("SplitSourceDatasetVersionId"));
    splitSourceDatasetVersionIdEdit_->setPlaceholderText(QStringLiteral("源 DatasetVersionId"));
    splitSourceSnapshotIdEdit_ = new QLineEdit;
    splitSourceSnapshotIdEdit_->setObjectName(QStringLiteral("SplitSourceSnapshotId"));
    splitSourceSnapshotIdEdit_->setPlaceholderText(QStringLiteral("源 SnapshotId"));
    splitSourceSnapshotArtifactIdEdit_ = new QLineEdit;
    splitSourceSnapshotArtifactIdEdit_->setObjectName(QStringLiteral("SplitSourceSnapshotArtifactId"));
    splitSourceSnapshotArtifactIdEdit_->setPlaceholderText(QStringLiteral("committed Snapshot ArtifactId"));
    splitTargetDatasetIdEdit_ = new QLineEdit(aitrain::DatasetId::create().toString());
    splitTargetDatasetIdEdit_->setObjectName(QStringLiteral("SplitTargetDatasetId"));
    splitTargetDatasetNameEdit_ = new QLineEdit;
    splitTargetDatasetNameEdit_->setObjectName(QStringLiteral("SplitTargetDatasetName"));
    splitTargetDatasetNameEdit_->setPlaceholderText(QStringLiteral("审计名称（当前 schema 不持久化名称）"));
    splitTrainRatioEdit_ = new QLineEdit(QStringLiteral("0.8"));
    splitValRatioEdit_ = new QLineEdit(QStringLiteral("0.2"));
    splitTestRatioEdit_ = new QLineEdit(QStringLiteral("0.0"));
    splitSeedEdit_ = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked, this, &MainWindow::splitDataset);
    auto* curateButton = new QPushButton(QStringLiteral("运行 Data Quality "));
    curateButton->setObjectName(QStringLiteral("RunDataQualityWorkflowButton"));
    connect(curateButton, &QPushButton::clicked, this, &MainWindow::curateDataset);
    auto* snapshotButton = new QPushButton(QStringLiteral("创建数据快照"));
    snapshotButton->setObjectName(QStringLiteral("RunDatasetSnapshotImportWorkflowButton"));
    connect(snapshotButton, &QPushButton::clicked, this, &MainWindow::createDatasetSnapshot);
    datasetSnapshotTargetDatasetIdEdit_ = new QLineEdit(aitrain::DatasetId::create().toString());
    datasetSnapshotTargetDatasetIdEdit_->setObjectName(QStringLiteral("DatasetSnapshotTargetDatasetId"));
    datasetSnapshotTargetDatasetIdEdit_->setPlaceholderText(QStringLiteral("新 DatasetId，或已有同格式 DatasetId"));
    datasetSnapshotTargetDatasetNameEdit_ = new QLineEdit;
    datasetSnapshotTargetDatasetNameEdit_->setObjectName(QStringLiteral("DatasetSnapshotTargetDatasetName"));
    datasetSnapshotTargetDatasetNameEdit_->setPlaceholderText(QStringLiteral("审计名称（当前 schema 不持久化名称）"));
    auto* openQualityReportButton = new QPushButton(QStringLiteral("打开质量报告"));
    connect(openQualityReportButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityReport);
    auto* openFixListButton = new QPushButton(QStringLiteral("打开问题清单"));
    connect(openFixListButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityFixList);
    auto* fixWithXAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling 修复"));
    connect(fixWithXAnyButton, &QPushButton::clicked, this, &MainWindow::createXAnyLabelingAnnotationSession);
    auto* ratioRow = new QWidget;
    auto* ratioLayout = new QHBoxLayout(ratioRow);
    ratioLayout->setContentsMargins(0, 0, 0, 0);
    ratioLayout->addWidget(splitTrainRatioEdit_);
    ratioLayout->addWidget(splitValRatioEdit_);
    ratioLayout->addWidget(splitTestRatioEdit_);
    ratioLayout->addWidget(splitSeedEdit_);
    form->addRow(QStringLiteral("划分源 DatasetId"), splitSourceDatasetIdEdit_);
    form->addRow(QStringLiteral("划分源 VersionId"), splitSourceDatasetVersionIdEdit_);
    form->addRow(QStringLiteral("划分源 SnapshotId"), splitSourceSnapshotIdEdit_);
    form->addRow(QStringLiteral("划分源 ArtifactId"), splitSourceSnapshotArtifactIdEdit_);
    form->addRow(QStringLiteral("划分目标 DatasetId"), splitTargetDatasetIdEdit_);
    form->addRow(QStringLiteral("划分审计名称"), splitTargetDatasetNameEdit_);
    form->addRow(QStringLiteral("train / val / test / seed"), ratioRow);
    form->addRow(QStringLiteral("快照目标 DatasetId"), datasetSnapshotTargetDatasetIdEdit_);
    form->addRow(QStringLiteral("快照审计名称"), datasetSnapshotTargetDatasetNameEdit_);
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
    datasetConversionStatusLabel_ = mutedLabel(QStringLiteral("选择外部源、目标格式和目标 Dataset 身份后开始受控转换。"));
    allowLabelToShrink(datasetConversionStatusLabel_);
    conversionLayout->addWidget(conversionTitle);
    conversionLayout->addWidget(datasetConversionStatusLabel_);

    datasetConversionSourceFormatCombo_ = new QComboBox;
    for (const QString& format : supportedDatasetConversionSourceFormats()) {
        addComboItem(datasetConversionSourceFormatCombo_, datasetConversionFormatLabel(format), format);
    }
    datasetConversionTargetFormatCombo_ = new QComboBox;
    for (const QString& target : supportedDatasetConversionTargets(comboCurrentDataOrText(datasetConversionSourceFormatCombo_))) {
        addComboItem(datasetConversionTargetFormatCombo_, datasetConversionFormatLabel(target), target);
    }
    if (datasetConversionTargetFormatCombo_->count() > 0) {
        datasetConversionTargetFormatCombo_->setCurrentIndex(0);
    }
    connect(datasetConversionSourceFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::updateDatasetConversionTargetFormats);

    datasetConversionInputEdit_ = new QLineEdit;
    datasetConversionBrowseInputButton_ = new QPushButton(QStringLiteral("选择输入"));
    connect(datasetConversionBrowseInputButton_, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionInput);
    auto* conversionInputRow = new QWidget;
    auto* conversionInputLayout = new QHBoxLayout(conversionInputRow);
    conversionInputLayout->setContentsMargins(0, 0, 0, 0);
    conversionInputLayout->setSpacing(8);
    conversionInputLayout->addWidget(datasetConversionInputEdit_, 1);
    conversionInputLayout->addWidget(datasetConversionBrowseInputButton_);

    datasetConversionTargetDatasetIdEdit_ = new QLineEdit(
        aitrain::DatasetId::create().toString());
    datasetConversionTargetDatasetIdEdit_->setObjectName(QStringLiteral("DatasetConversionTargetDatasetId"));
    datasetConversionTargetDatasetNameEdit_ = new QLineEdit;
    datasetConversionTargetDatasetNameEdit_->setObjectName(QStringLiteral("DatasetConversionTargetDatasetName"));
    datasetConversionTargetDatasetNameEdit_->setPlaceholderText(QStringLiteral("仅作审计显示；名称尚无独立 Storage 字段"));

    auto* conversionForm = new QFormLayout;
    conversionForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    conversionForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    conversionForm->setHorizontalSpacing(12);
    conversionForm->setVerticalSpacing(6);
    conversionForm->addRow(QStringLiteral("源格式"), datasetConversionSourceFormatCombo_);
    datasetConversionSourceErrorLabel_ = new QLabel;
    datasetConversionSourceErrorLabel_->setObjectName(QStringLiteral("FieldErrorText"));
    datasetConversionSourceErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionSourceErrorLabel_);
    conversionForm->addRow(QStringLiteral("目标格式"), datasetConversionTargetFormatCombo_);
    datasetConversionTargetErrorLabel_ = new QLabel;
    datasetConversionTargetErrorLabel_->setObjectName(QStringLiteral("FieldErrorText"));
    datasetConversionTargetErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionTargetErrorLabel_);
    conversionForm->addRow(QStringLiteral("输入路径"), conversionInputRow);
    datasetConversionInputErrorLabel_ = new QLabel;
    datasetConversionInputErrorLabel_->setObjectName(QStringLiteral("FieldErrorText"));
    datasetConversionInputErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionInputErrorLabel_);
    conversionForm->addRow(QStringLiteral("目标 DatasetId"), datasetConversionTargetDatasetIdEdit_);
    conversionForm->addRow(QStringLiteral("目标名称（审计）"), datasetConversionTargetDatasetNameEdit_);
    conversionLayout->addLayout(conversionForm);

    auto* conversionActionRow = new QWidget;
    auto* conversionActionLayout = new QHBoxLayout(conversionActionRow);
    conversionActionLayout->setContentsMargins(0, 0, 0, 0);
    conversionActionLayout->setSpacing(8);
    datasetConversionStartButton_ = primaryButton(QStringLiteral("转换数据集"));
    datasetConversionStartButton_->setObjectName(QStringLiteral("RunDatasetConversionWorkflowButton"));
    datasetConversionCancelButton_ = dangerButton(QStringLiteral("取消转换"));
    datasetConversionCancelButton_->setEnabled(false);
    connect(datasetConversionStartButton_, &QPushButton::clicked, this, &MainWindow::startDatasetConversion);
    connect(datasetConversionCancelButton_, &QPushButton::clicked, this, &MainWindow::cancelDatasetConversion);
    conversionActionLayout->addStretch();
    conversionActionLayout->addWidget(datasetConversionStartButton_);
    conversionActionLayout->addWidget(datasetConversionCancelButton_);
    conversionLayout->addWidget(conversionActionRow);

    datasetConversionProgressBar_ = new QProgressBar;
    datasetConversionProgressBar_->setRange(0, 100);
    datasetConversionProgressBar_->setValue(0);
    datasetConversionResultLabel_ = inlineStatusLabel(QStringLiteral("转换结果会显示在这里。"));
    allowLabelToShrink(datasetConversionResultLabel_);
    datasetConversionLog_ = new QPlainTextEdit;
    datasetConversionLog_->setReadOnly(true);
    datasetConversionLog_->setMinimumHeight(96);
    datasetConversionLog_->setPlainText(QStringLiteral("等待转换。"));
    conversionLayout->addWidget(datasetConversionProgressBar_);
    conversionLayout->addWidget(datasetConversionResultLabel_);
    conversionLayout->addWidget(datasetConversionLog_);
    inputPanel->bodyLayout()->addWidget(conversionStrip);

    auto* splitter = new QSplitter(Qt::Horizontal);
    auto* resultPanel = new InfoPanel(QStringLiteral("所选数据集详情"));
    datasetDetailLabel_ = inlineStatusLabel(QStringLiteral("选择或导入数据集后显示格式、样本数、校验状态和最近报告。"));
    validationSummaryLabel_ = mutedLabel(QStringLiteral("请选择数据集目录和格式，然后执行校验。"));
    allowLabelToShrink(datasetDetailLabel_);
    allowLabelToShrink(validationSummaryLabel_);
    datasetRepairLoopLabel_ = inlineStatusLabel(QStringLiteral("修复闭环：等待质量报告。"));
    allowLabelToShrink(datasetRepairLoopLabel_);
    datasetRepairLoopTable_ = new QTableWidget(0, 3);
    datasetRepairLoopTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("环节")
        << QStringLiteral("状态")
        << QStringLiteral("下一步"));
    configureTable(datasetRepairLoopTable_);
    datasetRepairLoopTable_->setWordWrap(true);
    datasetRepairLoopTable_->verticalHeader()->setDefaultSectionSize(34);
    datasetRepairLoopTable_->setMaximumHeight(156);
    datasetRepairLoopTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetRepairLoopTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetRepairLoopTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    validationIssuesTable_ = new QTableWidget(0, 5);
    validationIssuesTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("级别")
        << QStringLiteral("代码")
        << QStringLiteral("文件")
        << QStringLiteral("行号")
        << QStringLiteral("说明"));
    configureTable(validationIssuesTable_);
    validationIssuesTable_->verticalHeader()->setDefaultSectionSize(38);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    validationIssuesTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    validationOutput_ = new QPlainTextEdit;
    validationOutput_->setReadOnly(true);
    validationOutput_->setMinimumHeight(130);
    validationOutput_->setPlainText(QStringLiteral("校验报告 JSON 会显示在这里。"));
    resultPanel->bodyLayout()->addWidget(datasetDetailLabel_);
    resultPanel->bodyLayout()->addWidget(validationSummaryLabel_);
    resultPanel->bodyLayout()->addWidget(datasetRepairLoopLabel_);
    resultPanel->bodyLayout()->addWidget(datasetRepairLoopTable_);
    resultPanel->bodyLayout()->addWidget(validationIssuesTable_, 2);
    resultPanel->bodyLayout()->addWidget(validationOutput_);

    auto* toolsPanel = new InfoPanel(QStringLiteral("数据集库与样本预览"));
    datasetListTable_ = new QTableWidget(0, 5);
    datasetListTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("数据集")
        << QStringLiteral("格式")
        << QStringLiteral("状态")
        << QStringLiteral("样本")
        << QStringLiteral("快照身份"));
    configureTable(datasetListTable_);
    datasetListTable_->setWordWrap(true);
    datasetListTable_->verticalHeader()->setDefaultSectionSize(40);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    datasetListTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    connect(datasetListTable_, &QTableWidget::itemSelectionChanged, this, [this]() {
        if (!datasetListTable_ || datasetListTable_->selectedItems().isEmpty()) {
            return;
        }
        const int row = datasetListTable_->selectedItems().first()->row();
        const QString datasetId = datasetListTable_->item(row, 0)
            ? datasetListTable_->item(row, 0)->data(Qt::UserRole).toString() : QString();
        const QString format = datasetListTable_->item(row, 1) ? datasetListTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
        const QString snapshotId = datasetListTable_->item(row, 2)
            ? datasetListTable_->item(row, 2)->data(Qt::UserRole).toString() : QString();
        const QString artifactId = datasetListTable_->item(row, 4)
            ? datasetListTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        const QString versionId = datasetListTable_->item(row, 4)
            ? datasetListTable_->item(row, 4)->data(Qt::UserRole + 1).toString() : QString();
        if (!datasetId.isEmpty()) {
            // 目录查询只返回 committed 身份；不能把 ArtifactId 当成本地数据集路径。
            datasetPathEdit_->clear();
            const int formatIndex = datasetFormatCombo_->findData(format);
            if (formatIndex >= 0) {
                datasetFormatCombo_->setCurrentIndex(formatIndex);
            }
            state_.dataset.currentPath.clear();
            state_.dataset.currentFormat = format;
            state_.dataset.currentDatasetId = datasetId;
            state_.dataset.currentDatasetVersionId = versionId;
            state_.dataset.currentSnapshotId = snapshotId;
            state_.dataset.currentSnapshotArtifactId = artifactId;
            state_.dataset.currentValid = !versionId.isEmpty()
                && !snapshotId.isEmpty() && !artifactId.isEmpty();
            if (dataQualityDatasetIdEdit_) dataQualityDatasetIdEdit_->setText(datasetId);
            if (dataQualityDatasetVersionIdEdit_) dataQualityDatasetVersionIdEdit_->setText(versionId);
            if (dataQualitySnapshotIdEdit_) dataQualitySnapshotIdEdit_->setText(snapshotId);
            if (dataQualitySnapshotArtifactIdEdit_) dataQualitySnapshotArtifactIdEdit_->setText(artifactId);
            if (splitSourceDatasetIdEdit_) splitSourceDatasetIdEdit_->setText(datasetId);
            if (splitSourceDatasetVersionIdEdit_) splitSourceDatasetVersionIdEdit_->setText(versionId);
            if (splitSourceSnapshotIdEdit_) splitSourceSnapshotIdEdit_->setText(snapshotId);
            if (splitSourceSnapshotArtifactIdEdit_) splitSourceSnapshotArtifactIdEdit_->setText(artifactId);
            updateTrainingSelectionSummary();
            refreshTrainingDefaults();
            refreshDatasetConversionDefaultsFromCurrentDataset();
        }
    });
    datasetPreviewTable_ = new QTableWidget(0, 2);
    datasetPreviewTable_->setHorizontalHeaderLabels(QStringList() << QStringLiteral("样本") << QStringLiteral("标签 / 说明"));
    configureTable(datasetPreviewTable_);
    datasetPreviewTable_->setWordWrap(true);
    datasetPreviewTable_->verticalHeader()->setDefaultSectionSize(40);
    datasetPreviewTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    datasetPreviewTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    auto* annotationPanel = new QGroupBox(QStringLiteral("外部标注工具"));
    annotationPanel->setMinimumHeight(136);
    annotationPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    auto* annotationLayout = new QVBoxLayout(annotationPanel);
    annotationLayout->setContentsMargins(10, 12, 10, 8);
    annotationLayout->setSpacing(6);
    auto* annotationSummary = mutedLabel(uiText("X-AnyLabeling：检测导出 YOLO bbox，分割导出 YOLO polygon；PaddleOCR 使用 det_gt / rec_gt + dict。"));
    annotationToolStatusLabel_ = inlineStatusLabel(xAnyLabelingStatusText());
    allowLabelToShrink(annotationSummary);
    allowLabelToShrink(annotationToolStatusLabel_);
    annotationSummary->setMinimumHeight(28);
    annotationToolStatusLabel_->setMinimumHeight(28);
    annotationToolStatusLabel_->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    auto* createAnnotationSessionButton = new QPushButton(QStringLiteral("准备修复会话"));
    auto* syncAnnotationSessionButton = new QPushButton(QStringLiteral("同步标注会话"));
    createAnnotationSessionButton->setObjectName(QStringLiteral("CreateAnnotationSessionButton"));
    syncAnnotationSessionButton->setObjectName(QStringLiteral("SyncAnnotationSessionButton"));
    auto* refreshAnnotationStatusButton = new QPushButton(QStringLiteral("检测状态"));
    auto* openDatasetDirButton = new QPushButton(QStringLiteral("打开数据目录"));
    connect(refreshAnnotationStatusButton, &QPushButton::clicked, this, &MainWindow::updateAnnotationToolStatus);
    connect(createAnnotationSessionButton, &QPushButton::clicked, this, &MainWindow::createXAnyLabelingAnnotationSession);
    connect(syncAnnotationSessionButton, &QPushButton::clicked, this, &MainWindow::syncXAnyLabelingAnnotationSession);
    connect(openDatasetDirButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    });
    auto* annotationActionRow = new QWidget;
    auto* annotationActionGrid = new QGridLayout(annotationActionRow);
    annotationActionGrid->setContentsMargins(0, 0, 0, 0);
    annotationActionGrid->setHorizontalSpacing(10);
    annotationActionGrid->setVerticalSpacing(8);
    annotationActionGrid->addWidget(createAnnotationSessionButton, 0, 0);
    annotationActionGrid->addWidget(syncAnnotationSessionButton, 0, 1);
    annotationActionGrid->addWidget(refreshAnnotationStatusButton, 0, 2);
    annotationActionGrid->addWidget(openDatasetDirButton, 1, 0, 1, 3);
    for (int column = 0; column < 3; ++column) {
        annotationActionGrid->setColumnStretch(column, 1);
    }
    annotationLayout->addWidget(annotationSummary);
    annotationLayout->addWidget(annotationToolStatusLabel_);
    annotationLayout->addWidget(annotationActionRow);
    auto* datasetLibraryTab = new QWidget;
    auto* datasetLibraryLayout = new QVBoxLayout(datasetLibraryTab);
    datasetLibraryLayout->setContentsMargins(0, 0, 0, 0);
    datasetLibraryLayout->setSpacing(10);
    auto* datasetLibraryCaption = mutedLabel(QStringLiteral("已登记数据集"));
    allowLabelToShrink(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetLibraryCaption);
    datasetLibraryLayout->addWidget(datasetListTable_, 1);
    datasetLibraryLayout->addWidget(annotationPanel);

    auto* samplePreviewTab = new QWidget;
    auto* samplePreviewLayout = new QVBoxLayout(samplePreviewTab);
    samplePreviewLayout->setContentsMargins(0, 0, 0, 0);
    samplePreviewLayout->setSpacing(10);
    auto* samplePreviewHint = mutedLabel(uiText("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、YOLO OBB、语义分割 Mask PNG、PaddleOCR Det 和 PaddleOCR Rec。"));
    allowLabelToShrink(samplePreviewHint);
    samplePreviewLayout->addWidget(datasetPreviewTable_, 1);
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

    datasetTabs_ = new QTabWidget;
    datasetTabs_->setObjectName(QStringLiteral("DatasetTabs"));
    datasetTabs_->addTab(preparationTab, uiText("数据集准备"));
    datasetTabs_->addTab(buildSampleReviewPanel(), uiText("质量与复核"));

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
    layout->addWidget(datasetTabs_, 1);
    page->setWidget(content);
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
    reviewSamplePathEdit_ = new QLineEdit;
    reviewSamplePathEdit_->setPlaceholderText(uiText("输入已提交的 dataset_quality_analysis / dataset_repair_manifest ArtifactId"));
    auto* browseButton = new QPushButton(uiText("选择 Artifact"));
    connect(browseButton, &QPushButton::clicked, this, &MainWindow::browseSampleReviewFile);
    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->setSpacing(8);
    pathLayout->addWidget(reviewSamplePathEdit_, 1);
    pathLayout->addWidget(browseButton);

    reviewSourceFilterCombo_ = new QComboBox;
    reviewSourceFilterCombo_->addItem(uiText("全部来源"), QString());
    reviewReasonFilterCombo_ = new QComboBox;
    reviewReasonFilterCombo_->addItem(uiText("全部问题"), QString());
    reviewSearchEdit_ = new QLineEdit;
    reviewSearchEdit_->setPlaceholderText(uiText("按图片、标签、类别、说明搜索"));
    connect(reviewSourceFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::refreshSampleReviewTable);
    connect(reviewReasonFilterCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::refreshSampleReviewTable);
    connect(reviewSearchEdit_, &QLineEdit::textChanged, this, &MainWindow::refreshSampleReviewTable);

    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(10);
    form->addRow(uiText("复核 Artifact"), pathRow);
    form->addRow(uiText("来源"), reviewSourceFilterCombo_);
    form->addRow(uiText("问题类型"), reviewReasonFilterCombo_);
    form->addRow(uiText("搜索"), reviewSearchEdit_);
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
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadSampleReviewFile);
    connect(openButton, &QPushButton::clicked, this, &MainWindow::openSelectedReviewSample);
    actionLayout->addWidget(loadButton, 0, 0);
    actionLayout->addWidget(openButton, 0, 1);
    actionLayout->setColumnStretch(0, 1);
    actionLayout->setColumnStretch(1, 1);
    setupPanel->bodyLayout()->addWidget(actionStrip);

    sampleReviewSummaryLabel_ = inlineStatusLabel(uiText("尚未加载复核样本。"));
    setupPanel->bodyLayout()->addWidget(sampleReviewSummaryLabel_);
    setupPanel->bodyLayout()->addWidget(emptyStateLabel(uiText("样本复核页只读展示已加载清单；修复必须通过 Data Quality 与 Annotation Session。")));
    setupPanel->bodyLayout()->addStretch();

    auto* tablePanel = new InfoPanel(uiText("复核队列"));
    sampleReviewTable_ = new QTableWidget(0, 7);
    sampleReviewTable_->setHorizontalHeaderLabels(QStringList()
        << uiText("来源")
        << uiText("问题")
        << uiText("类别")
        << uiText("指标")
        << uiText("图片")
        << uiText("标签")
        << uiText("说明"));
    configureTable(sampleReviewTable_);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Stretch);
    sampleReviewTable_->horizontalHeader()->setSectionResizeMode(6, QHeaderView::Stretch);
    sampleReviewTable_->setMinimumHeight(420);
    tablePanel->bodyLayout()->addWidget(sampleReviewTable_);

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
