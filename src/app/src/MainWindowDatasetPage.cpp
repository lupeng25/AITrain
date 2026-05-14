#include "MainWindow.h"

#include "DatasetConversionUiModel.h"
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

QWidget* MainWindow::buildDatasetPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* headerValidateButton = primaryButton(QStringLiteral("校验数据集"));
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
    connect(datasetFormatCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        currentDatasetFormat_ = currentDatasetFormat();
        currentDatasetValid_ = false;
        updateTrainingSelectionSummary();
        refreshTrainingDefaults();
        refreshDatasetConversionDefaultsFromCurrentDataset();
    });
    auto* validateButton = primaryButton(QStringLiteral("校验数据集"));
    connect(validateButton, &QPushButton::clicked, this, &MainWindow::validateDataset);
    form->addRow(QStringLiteral("数据集目录"), pathRow);
    form->addRow(QStringLiteral("格式"), datasetFormatCombo_);

    splitOutputEdit_ = new QLineEdit;
    splitOutputEdit_->setPlaceholderText(QStringLiteral("默认输出到当前项目 datasets/normalized"));
    splitTrainRatioEdit_ = new QLineEdit(QStringLiteral("0.8"));
    splitValRatioEdit_ = new QLineEdit(QStringLiteral("0.2"));
    splitTestRatioEdit_ = new QLineEdit(QStringLiteral("0.0"));
    splitSeedEdit_ = new QLineEdit(QStringLiteral("42"));
    auto* splitButton = new QPushButton(QStringLiteral("划分数据集"));
    connect(splitButton, &QPushButton::clicked, this, &MainWindow::splitDataset);
    auto* curateButton = new QPushButton(QStringLiteral("生成质量报告"));
    connect(curateButton, &QPushButton::clicked, this, &MainWindow::curateDataset);
    auto* snapshotButton = new QPushButton(QStringLiteral("创建数据快照"));
    connect(snapshotButton, &QPushButton::clicked, this, &MainWindow::createDatasetSnapshot);
    auto* openQualityReportButton = new QPushButton(QStringLiteral("打开质量报告"));
    connect(openQualityReportButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityReport);
    auto* openFixListButton = new QPushButton(QStringLiteral("打开问题清单"));
    connect(openFixListButton, &QPushButton::clicked, this, &MainWindow::openDatasetQualityFixList);
    auto* fixWithXAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling 修复"));
    connect(fixWithXAnyButton, &QPushButton::clicked, this, &MainWindow::launchXAnyLabelingForQualityFix);
    auto* ratioRow = new QWidget;
    auto* ratioLayout = new QHBoxLayout(ratioRow);
    ratioLayout->setContentsMargins(0, 0, 0, 0);
    ratioLayout->addWidget(splitTrainRatioEdit_);
    ratioLayout->addWidget(splitValRatioEdit_);
    ratioLayout->addWidget(splitTestRatioEdit_);
    ratioLayout->addWidget(splitSeedEdit_);
    form->addRow(QStringLiteral("输出目录"), splitOutputEdit_);
    form->addRow(QStringLiteral("train / val / test / seed"), ratioRow);
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
    datasetConversionStatusLabel_ = mutedLabel(QStringLiteral("选择源格式、目标格式和输出目录后开始转换。"));
    allowLabelToShrink(datasetConversionStatusLabel_);
    conversionLayout->addWidget(conversionTitle);
    conversionLayout->addWidget(datasetConversionStatusLabel_);

    datasetConversionSourceFormatCombo_ = new QComboBox;
    for (const QString& format : supportedDatasetConversionSourceFormats()) {
        addComboItem(datasetConversionSourceFormatCombo_, datasetConversionFormatLabel(format), format);
    }
    datasetConversionTargetFormatCombo_ = new QComboBox;
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

    datasetConversionOutputEdit_ = new QLineEdit;
    datasetConversionBrowseOutputButton_ = new QPushButton(QStringLiteral("选择输出"));
    connect(datasetConversionBrowseOutputButton_, &QPushButton::clicked, this, &MainWindow::browseDatasetConversionOutput);
    auto* conversionOutputRow = new QWidget;
    auto* conversionOutputLayout = new QHBoxLayout(conversionOutputRow);
    conversionOutputLayout->setContentsMargins(0, 0, 0, 0);
    conversionOutputLayout->setSpacing(8);
    conversionOutputLayout->addWidget(datasetConversionOutputEdit_, 1);
    conversionOutputLayout->addWidget(datasetConversionBrowseOutputButton_);

    auto* conversionForm = new QFormLayout;
    conversionForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    conversionForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    conversionForm->setHorizontalSpacing(12);
    conversionForm->setVerticalSpacing(6);
    conversionForm->addRow(QStringLiteral("源格式"), datasetConversionSourceFormatCombo_);
    datasetConversionSourceErrorLabel_ = new QLabel;
    datasetConversionSourceErrorLabel_->setStyleSheet(QStringLiteral("color: #DC2626;"));
    datasetConversionSourceErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionSourceErrorLabel_);
    conversionForm->addRow(QStringLiteral("目标格式"), datasetConversionTargetFormatCombo_);
    datasetConversionTargetErrorLabel_ = new QLabel;
    datasetConversionTargetErrorLabel_->setStyleSheet(QStringLiteral("color: #DC2626;"));
    datasetConversionTargetErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionTargetErrorLabel_);
    conversionForm->addRow(QStringLiteral("输入目录"), conversionInputRow);
    datasetConversionInputErrorLabel_ = new QLabel;
    datasetConversionInputErrorLabel_->setStyleSheet(QStringLiteral("color: #DC2626;"));
    datasetConversionInputErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionInputErrorLabel_);
    conversionForm->addRow(QStringLiteral("输出目录"), conversionOutputRow);
    datasetConversionOutputErrorLabel_ = new QLabel;
    datasetConversionOutputErrorLabel_->setStyleSheet(QStringLiteral("color: #DC2626;"));
    datasetConversionOutputErrorLabel_->hide();
    conversionForm->addRow(QString(), datasetConversionOutputErrorLabel_);
    conversionLayout->addLayout(conversionForm);

    auto* conversionActionRow = new QWidget;
    auto* conversionActionLayout = new QHBoxLayout(conversionActionRow);
    conversionActionLayout->setContentsMargins(0, 0, 0, 0);
    conversionActionLayout->setSpacing(8);
    datasetConversionStartButton_ = primaryButton(QStringLiteral("转换数据集"));
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
    updateDatasetConversionTargetFormats();

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
        << QStringLiteral("路径"));
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
        const QString path = datasetListTable_->item(row, 4) ? datasetListTable_->item(row, 4)->data(Qt::UserRole).toString() : QString();
        const QString format = datasetListTable_->item(row, 1) ? datasetListTable_->item(row, 1)->data(Qt::UserRole).toString() : QString();
        if (!path.isEmpty()) {
            datasetPathEdit_->setText(QDir::toNativeSeparators(path));
            const int formatIndex = datasetFormatCombo_->findData(format);
            if (formatIndex >= 0) {
                datasetFormatCombo_->setCurrentIndex(formatIndex);
            }
            currentDatasetPath_ = path;
            currentDatasetFormat_ = format;
            currentDatasetValid_ = datasetListTable_->item(row, 2)
                && datasetListTable_->item(row, 2)->data(Qt::UserRole).toString() == QStringLiteral("valid");
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
    auto* annotationSummary = mutedLabel(QStringLiteral("X-AnyLabeling：检测导出 YOLO bbox，分割导出 YOLO polygon；PaddleOCR 使用 det_gt / rec_gt + dict。"));
    annotationToolStatusLabel_ = inlineStatusLabel(xAnyLabelingStatusText());
    allowLabelToShrink(annotationSummary);
    allowLabelToShrink(annotationToolStatusLabel_);
    annotationSummary->setMinimumHeight(28);
    annotationToolStatusLabel_->setMinimumHeight(28);
    annotationToolStatusLabel_->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    auto* launchAnnotationToolButton = new QPushButton(QStringLiteral("启动 X-AnyLabeling"));
    auto* refreshAnnotationStatusButton = new QPushButton(QStringLiteral("检测状态"));
    auto* refreshDatasetAfterAnnotationButton = new QPushButton(QStringLiteral("标注后刷新 / 重新校验"));
    auto* openDatasetDirButton = new QPushButton(QStringLiteral("打开数据目录"));
    connect(refreshAnnotationStatusButton, &QPushButton::clicked, this, &MainWindow::updateAnnotationToolStatus);
    connect(refreshDatasetAfterAnnotationButton, &QPushButton::clicked, this, &MainWindow::refreshAfterAnnotation);
    connect(openDatasetDirButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(datasetPath));
    });
    connect(launchAnnotationToolButton, &QPushButton::clicked, this, [this]() {
        const QString datasetPath = QDir::fromNativeSeparators(datasetPathEdit_ ? datasetPathEdit_->text().trimmed() : QString());
        if (datasetPath.isEmpty()) {
            QMessageBox::information(this, uiText("标注工具"), uiText("请先选择数据集目录。"));
            return;
        }
        const QString program = resolvedXAnyLabelingProgram();
        if (program.isEmpty()) {
            updateAnnotationToolStatus();
            QMessageBox::warning(this,
                QStringLiteral("X-AnyLabeling"),
                uiText("未找到 X-AnyLabeling。请确保 xanylabeling 在 PATH 中，或将 X-AnyLabeling.exe 放到程序目录 / tools/x-anylabeling / .deps/annotation-tools/X-AnyLabeling。"));
            return;
        }
        const QStringList arguments = {QStringLiteral("--filename"), datasetPath, QStringLiteral("--no-auto-update-check")};
        if (QProcess::startDetached(program, arguments)) {
            updateAnnotationToolStatus();
            statusBar()->showMessage(uiText("已启动 X-AnyLabeling：%1").arg(QDir::toNativeSeparators(datasetPath)), 4000);
            return;
        }
        QMessageBox::warning(this,
            QStringLiteral("X-AnyLabeling"),
            uiText("X-AnyLabeling 启动失败：%1").arg(QDir::toNativeSeparators(program)));
    });
    auto* annotationActionRow = new QWidget;
    auto* annotationActionGrid = new QGridLayout(annotationActionRow);
    annotationActionGrid->setContentsMargins(0, 0, 0, 0);
    annotationActionGrid->setHorizontalSpacing(10);
    annotationActionGrid->setVerticalSpacing(0);
    annotationActionGrid->addWidget(launchAnnotationToolButton, 0, 0);
    annotationActionGrid->addWidget(refreshAnnotationStatusButton, 0, 1);
    annotationActionGrid->addWidget(refreshDatasetAfterAnnotationButton, 0, 2);
    annotationActionGrid->addWidget(openDatasetDirButton, 0, 3);
    for (int column = 0; column < 4; ++column) {
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
    auto* samplePreviewHint = mutedLabel(QStringLiteral("划分会复制到新目录，不修改原始数据；支持 YOLO 检测、YOLO 分割、PaddleOCR Det 和 PaddleOCR Rec。"));
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

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("DATASET VALIDATION"),
        QStringLiteral("数据集工作台"),
        QStringLiteral("导入、校验、划分、快照和标注工具入口统一在这里；训练前必须先完成当前格式校验。"),
        headerValidateButton,
        QStringList()
            << QStringLiteral("YOLO BBox")
            << QStringLiteral("YOLO Polygon")
            << QStringLiteral("PaddleOCR Det")
            << QStringLiteral("PaddleOCR Rec")));
    layout->addWidget(inputPanel);
    layout->addWidget(splitter, 1);
    page->setWidget(content);
    return page;
}

QWidget* MainWindow::buildSampleReviewPage()
{
    auto* page = new QScrollArea;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 18, 18, 18);
    layout->setSpacing(16);

    auto* loadButton = primaryButton(uiText("加载复核样本"));
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadSampleReviewFile);
    auto* header = createWorkbenchHeader(
        QStringLiteral("SAMPLE REVIEW LOOP"),
        uiText("样本复核"),
        uiText("从质量报告、评估错误、低置信预测和 rework 清单中筛出样本，生成 X-AnyLabeling 本地复核队列。"),
        loadButton,
        QStringList() << QStringLiteral("problem_samples.json") << QStringLiteral("error_samples.json") << QStringLiteral("rework_sample_set.json"));
    layout->addWidget(header);

    auto* splitter = new QSplitter(Qt::Horizontal);

    auto* setupPanel = new InfoPanel(uiText("输入与过滤"));
    reviewSamplePathEdit_ = new QLineEdit;
    reviewSamplePathEdit_->setPlaceholderText(uiText("选择 problem_samples.json / error_samples.json / rework_sample_set.json / evaluation_report.json"));
    auto* browseButton = new QPushButton(uiText("选择文件"));
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
    form->addRow(uiText("样本文件"), pathRow);
    form->addRow(uiText("来源"), reviewSourceFilterCombo_);
    form->addRow(uiText("问题类型"), reviewReasonFilterCombo_);
    form->addRow(uiText("搜索"), reviewSearchEdit_);
    setupPanel->bodyLayout()->addLayout(form);

    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    actionLayout->setSpacing(8);
    auto* openButton = new QPushButton(uiText("打开样本"));
    auto* generateButton = primaryButton(uiText("生成复核清单"));
    auto* xAnyButton = new QPushButton(QStringLiteral("X-AnyLabeling"));
    connect(openButton, &QPushButton::clicked, this, &MainWindow::openSelectedReviewSample);
    connect(generateButton, &QPushButton::clicked, this, &MainWindow::generateFilteredReviewList);
    connect(xAnyButton, &QPushButton::clicked, this, &MainWindow::launchXAnyLabelingForReview);
    actionLayout->addWidget(mutedLabel(uiText("标注完成后回到“数据集”页重新校验并创建快照。")), 1);
    actionLayout->addWidget(openButton);
    actionLayout->addWidget(generateButton);
    actionLayout->addWidget(xAnyButton);
    setupPanel->bodyLayout()->addWidget(actionStrip);

    sampleReviewSummaryLabel_ = inlineStatusLabel(uiText("尚未加载复核样本。"));
    setupPanel->bodyLayout()->addWidget(sampleReviewSummaryLabel_);
    setupPanel->bodyLayout()->addWidget(emptyStateLabel(uiText("v1 只生成本地复核队列，不内嵌标注器，也不实现多人协作。")));
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
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 5);
    splitter->setSizes(QList<int>() << 360 << 840);

    layout->addWidget(splitter, 1);
    page->setWidget(content);
    return page;
}
