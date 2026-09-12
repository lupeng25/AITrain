#include "WorkbenchTranslation.h"
#include "DatasetPage.h"

#include "DatasetConversionUiModel.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"

#include <QComboBox>
#include <QFormLayout>
#include <QLineEdit>
#include <QMenu>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QSplitter>

using namespace aitrain_app;

namespace {
QLineEdit* edit(const QString& name, const QString& value = {})
{
    auto* result = new QLineEdit(value);
    result->setObjectName(name);
    result->setMinimumWidth(0);
    return result;
}

QFormLayout* form(InfoPanel* panel)
{
    auto* result = new QFormLayout;
    result->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    result->setHorizontalSpacing(18);
    result->setVerticalSpacing(12);
    panel->bodyLayout()->addLayout(result);
    panel->setMaximumWidth(850);
    return result;
}

QWidget* pathRow(QLineEdit* field, QPushButton* button)
{
    auto* row = new QWidget;
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(field, 1);
    layout->addWidget(button);
    return row;
}
} // namespace

DatasetWorkspacePage::DatasetWorkspacePage(QWidget* parent)
    : WorkspaceViewHost(parent)
{
    setObjectName(QStringLiteral("DatasetWorkspacePage"));
    auto* importButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入数据")), QStringLiteral("OpenDatasetImportButton"), true);
    auto* refreshButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("刷新")), QStringLiteral("DatasetRefreshButton"));
    auto* conversionButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("格式转换")), QStringLiteral("OpenDatasetConversionButton"));
    toolbar->addWidget(conversionButton);
    connect(conversionButton, &QPushButton::clicked, this, [this]() { showView(Conversion); });
    toolbar->addWidget(refreshButton);
    toolbar->addWidget(importButton);
    connect(importButton, &QPushButton::clicked, this, [this]() { showView(Import); });
    connect(refreshButton, &QPushButton::clicked, this, &DatasetWorkspacePage::refreshRequested);
    connect(views, &QStackedWidget::currentChanged, this, [importButton, refreshButton, conversionButton](int index) {
        importButton->setVisible(index == Catalog);
        refreshButton->setVisible(index == Catalog);
        conversionButton->setVisible(index == Catalog);
    });

    auto* hidden = new QWidget(this);
    hidden->hide();
    const auto internal = [hidden](const QString& name) {
        auto* value = new QLineEdit(hidden);
        value->setObjectName(name);
        value->setReadOnly(true);
        return value;
    };
    splitSourceDatasetIdEdit = internal(QStringLiteral("SplitSourceDatasetId"));
    splitSourceDatasetVersionIdEdit = internal(QStringLiteral("SplitSourceDatasetVersionId"));
    splitSourceSnapshotIdEdit = internal(QStringLiteral("SplitSourceSnapshotId"));
    splitSourceSnapshotArtifactIdEdit = internal(QStringLiteral("SplitSourceSnapshotArtifactId"));
    splitTargetDatasetIdEdit = internal(QStringLiteral("SplitTargetDatasetId"));
    datasetSnapshotTargetDatasetIdEdit = internal(QStringLiteral("DatasetSnapshotTargetDatasetId"));
    datasetConversionTargetDatasetIdEdit = internal(QStringLiteral("DatasetConversionTargetDatasetId"));
    reviewSamplePathEdit = internal(QStringLiteral("ReviewArtifactId"));

    auto* catalog = addMode(aitrain_app::workbenchText(QStringLiteral("已登记数据集")));
    catalog->addWidget(catalogSearchField(QStringLiteral("DatasetCatalogSearch"), aitrain_app::workbenchText(QStringLiteral("搜索整个项目：数据集名称或格式"))));
    datasetListTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("名称")), aitrain_app::workbenchText(QStringLiteral("任务 / 格式")),
        aitrain_app::workbenchText(QStringLiteral("质量")), aitrain_app::workbenchText(QStringLiteral("样本数")), aitrain_app::workbenchText(QStringLiteral("版本"))});
    datasetListTable->setObjectName(QStringLiteral("DatasetCatalogTable"));
    catalog->addWidget(datasetListTable, 1);
    catalogStatusLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("打开项目后显示数据集；使用“导入数据”建立第一份快照。")));
    catalog->addWidget(catalogStatusLabel);
    auto* actions = new QHBoxLayout;
    auto* detail = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看样本")), QStringLiteral("DatasetDetailButton"));
    auto* train = workbenchButton(aitrain_app::workbenchText(QStringLiteral("用于训练")), QStringLiteral("DatasetTrainButton"), true);
    auto* more = workbenchButton(aitrain_app::workbenchText(QStringLiteral("更多操作")), QStringLiteral("DatasetMoreButton"));
    auto* moreMenu = new QMenu(more);
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("检查质量")), this, [this]() { emit qualityRequested(); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("导入新版本")), this, [this]() { emit importVersionRequested(); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("划分数据")), this, [this]() { showView(Split); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("格式转换")), this, [this]() { showView(Conversion); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("修复与标注")), this, [this]() { showView(Annotation); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("样本复核")), this, [this]() { showView(Review); });
    moreMenu->addAction(aitrain_app::workbenchText(QStringLiteral("技术详情")), this, [this]() { showView(Technical); });
    more->setMenu(moreMenu);
    connect(detail, &QPushButton::clicked, this, [this]() { showView(Detail); });
    connect(datasetListTable, &QTableWidget::cellDoubleClicked, this, [this](int, int) {
        if (datasetListTable->currentItem() && !datasetListTable->item(datasetListTable->currentRow(), 0)->data(Qt::UserRole).toString().isEmpty()) showView(Detail);
    });
    connect(train, &QPushButton::clicked, this, &DatasetWorkspacePage::trainRequested);
    selectionActions_ << detail << train << more;
    actions->addWidget(detail);
    actions->addWidget(more);
    actions->addWidget(train);
    actions->addStretch();
    previousPageButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("上一页")), QStringLiteral("DatasetPreviousPageButton"));
    nextPageButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("下一页")), QStringLiteral("DatasetNextPageButton"));
    connect(previousPageButton, &QPushButton::clicked, this, &DatasetWorkspacePage::previousPageRequested);
    connect(nextPageButton, &QPushButton::clicked, this, &DatasetWorkspacePage::nextPageRequested);
    actions->addWidget(previousPageButton);
    actions->addWidget(nextPageButton);
    catalog->addLayout(actions);

    auto* details = addMode(aitrain_app::workbenchText(QStringLiteral("样本与版本")));
    datasetDetailLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("请选择数据集。")));
    datasetDetailLabel->setObjectName(QStringLiteral("DatasetDetailLabel"));
    details->addWidget(datasetDetailLabel);
    auto* versionRow = new QHBoxLayout;
    versionRow->addWidget(new QLabel(aitrain_app::workbenchText(QStringLiteral("版本 / 快照"))));
    snapshotCombo = new QComboBox;
    snapshotCombo->setObjectName(QStringLiteral("DatasetSnapshotSelector"));
    snapshotCombo->setMinimumWidth(0);
    snapshotCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    versionRow->addWidget(snapshotCombo, 1);
    snapshotLoadMoreButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("更多版本")));
    versionRow->addWidget(snapshotLoadMoreButton);
    auto* detailQuality = workbenchButton(aitrain_app::workbenchText(QStringLiteral("检查质量")), QStringLiteral("RunDataQualityWorkflowButton"));
    auto* detailTrain = workbenchButton(aitrain_app::workbenchText(QStringLiteral("用于训练")), QStringLiteral("DatasetDetailTrainButton"), true);
    versionRow->addWidget(detailQuality);
    versionRow->addWidget(detailTrain);
    connect(snapshotCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &DatasetWorkspacePage::snapshotChanged);
    connect(snapshotLoadMoreButton, &QPushButton::clicked, this, &DatasetWorkspacePage::moreSnapshotsRequested);
    connect(detailQuality, &QPushButton::clicked, this, &DatasetWorkspacePage::qualityRequested);
    connect(detailTrain, &QPushButton::clicked, this, &DatasetWorkspacePage::trainRequested);
    selectionActions_ << detailQuality << detailTrain;
    details->addLayout(versionRow);
    auto* preview = new QSplitter(Qt::Horizontal);
    datasetPreviewTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("样本文件")), aitrain_app::workbenchText(QStringLiteral("大小"))});
    datasetPreviewTable->setObjectName(QStringLiteral("DatasetPreviewTable"));
    sampleImageLabel = new ImagePreviewLabel;
    sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("选择样本后显示预览。")));
    sampleImageLabel->setObjectName(QStringLiteral("DatasetSampleImage"));
    sampleImageLabel->setAlignment(Qt::AlignCenter);
    sampleImageLabel->setMinimumSize(0, 0);
    sampleImageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    sampleImageLabel->setWordWrap(true);
    preview->addWidget(datasetPreviewTable);
    preview->addWidget(sampleImageLabel);
    preview->setStretchFactor(0, 2);
    preview->setStretchFactor(1, 3);
    preview->setChildrenCollapsible(false);
    details->addWidget(preview, 1);
    connect(datasetPreviewTable, &QTableWidget::currentCellChanged, this, [this](int row, int, int, int) { emit sampleSelected(row); });
    auto* sampleFooter = new QHBoxLayout;
    sampleStatusLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("样本按需读取。")));
    sampleLoadMoreButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("加载更多文件")));
    connect(sampleLoadMoreButton, &QPushButton::clicked, this, &DatasetWorkspacePage::moreSamplesRequested);
    sampleFooter->addWidget(sampleStatusLabel, 1);
    sampleFooter->addWidget(sampleLoadMoreButton);
    details->addLayout(sampleFooter);

    auto* importLayout = addMode(aitrain_app::workbenchText(QStringLiteral("导入数据")));
    auto* importPanel = new InfoPanel(aitrain_app::workbenchText(QStringLiteral("建立项目数据快照")));
    auto* importForm = form(importPanel);
    datasetSnapshotTargetDatasetNameEdit = edit(QStringLiteral("DatasetSnapshotTargetDatasetName"));
    datasetPathEdit = edit(QStringLiteral("DatasetPathEdit"));
    auto* browse = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择目录")));
    connect(browse, &QPushButton::clicked, this, &DatasetWorkspacePage::browseDatasetRequested);
    datasetFormatCombo = new QComboBox;
    datasetFormatCombo->setObjectName(QStringLiteral("DatasetFormat"));
    QStringList formats;
    for (const auto& capability : aitrain::BuiltinCapabilityRegistry::instance().capabilities()) {
        for (const QString& format : capability.datasetFormats) {
            if (!formats.contains(format)) {
                formats.append(format);
                datasetFormatCombo->addItem(datasetFormatLabel(format), format);
            }
        }
    }
    datasetProbeStatusLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("选择目录后自动探测格式；导入时会执行完整校验。")));
    datasetProbeStatusLabel->setObjectName(QStringLiteral("DatasetProbeStatus"));
    importForm->addRow(aitrain_app::workbenchText(QStringLiteral("数据集名称")), datasetSnapshotTargetDatasetNameEdit);
    importForm->addRow(aitrain_app::workbenchText(QStringLiteral("来源目录")), pathRow(datasetPathEdit, browse));
    importForm->addRow(aitrain_app::workbenchText(QStringLiteral("格式")), datasetFormatCombo);
    importPanel->bodyLayout()->addWidget(datasetProbeStatusLabel);
    auto* submitImport = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入")), QStringLiteral("RunDatasetSnapshotImportWorkflowButton"), true);
    connect(submitImport, &QPushButton::clicked, this, &DatasetWorkspacePage::importRequested);
    importPanel->bodyLayout()->addWidget(submitImport, 0, Qt::AlignRight);
    importLayout->addWidget(importPanel, 0, Qt::AlignTop | Qt::AlignHCenter);
    operationStatusLabel = workbenchHint();
    operationStatusLabel->setObjectName(QStringLiteral("DatasetOperationStatus"));
    importLayout->addWidget(operationStatusLabel);
    importLayout->addStretch();

    auto* splitLayout = addMode(aitrain_app::workbenchText(QStringLiteral("划分数据")));
    auto* splitPanel = new InfoPanel(aitrain_app::workbenchText(QStringLiteral("从所选快照建立新数据集")));
    auto* splitForm = form(splitPanel);
    splitTargetDatasetNameEdit = edit(QStringLiteral("SplitTargetDatasetName"));
    splitTrainRatioEdit = edit(QStringLiteral("SplitTrainRatio"), QStringLiteral("0.8"));
    splitValRatioEdit = edit(QStringLiteral("SplitValRatio"), QStringLiteral("0.2"));
    splitTestRatioEdit = edit(QStringLiteral("SplitTestRatio"), QStringLiteral("0.0"));
    splitSeedEdit = edit(QStringLiteral("SplitSeed"), QStringLiteral("42"));
    splitForm->addRow(aitrain_app::workbenchText(QStringLiteral("新数据集名称")), splitTargetDatasetNameEdit);
    splitForm->addRow(aitrain_app::workbenchText(QStringLiteral("训练集比例")), splitTrainRatioEdit);
    splitForm->addRow(aitrain_app::workbenchText(QStringLiteral("验证集比例")), splitValRatioEdit);
    splitForm->addRow(aitrain_app::workbenchText(QStringLiteral("测试集比例")), splitTestRatioEdit);
    splitForm->addRow(aitrain_app::workbenchText(QStringLiteral("随机种子")), splitSeedEdit);
    auto* splitButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("开始划分")), QStringLiteral("RunDatasetSplitWorkflowButton"), true);
    connect(splitButton, &QPushButton::clicked, this, &DatasetWorkspacePage::splitRequested);
    splitPanel->bodyLayout()->addWidget(splitButton, 0, Qt::AlignRight);
    splitLayout->addWidget(splitPanel, 0, Qt::AlignTop | Qt::AlignHCenter);
    splitLayout->addStretch();

    auto* conversionLayout = addMode(aitrain_app::workbenchText(QStringLiteral("格式转换")));
    auto* conversionPanel = new InfoPanel(aitrain_app::workbenchText(QStringLiteral("导入并转换外部数据")));
    auto* conversionForm = form(conversionPanel);
    datasetConversionSourceFormatCombo = new QComboBox;
    for (const QString& source : supportedDatasetConversionSourceFormats())
        datasetConversionSourceFormatCombo->addItem(datasetConversionFormatLabel(source), source);
    datasetConversionTargetFormatCombo = new QComboBox;
    datasetConversionInputEdit = edit(QStringLiteral("DatasetConversionInput"));
    datasetConversionTargetDatasetNameEdit = edit(QStringLiteral("DatasetConversionTargetDatasetName"));
    datasetConversionBrowseInputButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择来源")));
    datasetConversionProbeStatusLabel = workbenchHint();
    datasetConversionProbeStatusLabel->setObjectName(QStringLiteral("DatasetConversionProbeStatus"));
    datasetConversionSourceErrorLabel = workbenchHint();
    datasetConversionTargetErrorLabel = workbenchHint();
    datasetConversionInputErrorLabel = workbenchHint();
    for (QLabel* label : {datasetConversionSourceErrorLabel, datasetConversionTargetErrorLabel, datasetConversionInputErrorLabel}) {
        label->setObjectName(QStringLiteral("FieldErrorText"));
        label->hide();
    }
    conversionForm->addRow(aitrain_app::workbenchText(QStringLiteral("来源格式")), datasetConversionSourceFormatCombo);
    conversionForm->addRow(QString(), datasetConversionSourceErrorLabel);
    conversionForm->addRow(aitrain_app::workbenchText(QStringLiteral("目标格式")), datasetConversionTargetFormatCombo);
    conversionForm->addRow(QString(), datasetConversionTargetErrorLabel);
    conversionForm->addRow(aitrain_app::workbenchText(QStringLiteral("来源")), pathRow(datasetConversionInputEdit, datasetConversionBrowseInputButton));
    conversionForm->addRow(QString(), datasetConversionInputErrorLabel);
    conversionForm->addRow(aitrain_app::workbenchText(QStringLiteral("新数据集名称")), datasetConversionTargetDatasetNameEdit);
    conversionPanel->bodyLayout()->addWidget(datasetConversionProbeStatusLabel);
    connect(datasetConversionSourceFormatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) { emit conversionSourceChanged(); });
    connect(datasetConversionBrowseInputButton, &QPushButton::clicked, this, &DatasetWorkspacePage::browseConversionRequested);
    auto* conversionActions = new QHBoxLayout;
    datasetConversionStartButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("转换数据集")), QStringLiteral("RunDatasetConversionWorkflowButton"), true);
    datasetConversionCancelButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("取消转换")));
    datasetConversionCancelButton->setEnabled(false);
    connect(datasetConversionStartButton, &QPushButton::clicked, this, &DatasetWorkspacePage::conversionRequested);
    connect(datasetConversionCancelButton, &QPushButton::clicked, this, &DatasetWorkspacePage::cancelRequested);
    conversionActions->addStretch();
    conversionActions->addWidget(datasetConversionCancelButton);
    conversionActions->addWidget(datasetConversionStartButton);
    conversionPanel->bodyLayout()->addLayout(conversionActions);
    conversionLayout->addWidget(conversionPanel, 0, Qt::AlignHCenter);
    datasetConversionStatusLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("请选择来源与支持的目标格式。")));
    datasetConversionProgressBar = new QProgressBar;
    datasetConversionProgressBar->setRange(0, 100);
    datasetConversionProgressBar->setValue(0);
    datasetConversionResultLabel = workbenchHint();
    datasetConversionLog = new QPlainTextEdit;
    datasetConversionLog->setReadOnly(true);
    datasetConversionLog->setMaximumBlockCount(2000);
    datasetConversionLog->setMinimumHeight(0);
    conversionLayout->addWidget(datasetConversionStatusLabel);
    conversionLayout->addWidget(datasetConversionProgressBar);
    conversionLayout->addWidget(datasetConversionResultLabel);
    conversionLayout->addWidget(datasetConversionLog, 1);

    auto* quality = addMode(aitrain_app::workbenchText(QStringLiteral("质量结果")));
    validationSummaryLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未检查所选快照。")));
    validationSummaryLabel->setObjectName(QStringLiteral("DatasetQualitySummary"));
    quality->addWidget(validationSummaryLabel);
    validationIssuesTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("级别")), aitrain_app::workbenchText(QStringLiteral("代码")), aitrain_app::workbenchText(QStringLiteral("文件")), aitrain_app::workbenchText(QStringLiteral("行号")), aitrain_app::workbenchText(QStringLiteral("说明"))});
    validationIssuesTable->setObjectName(QStringLiteral("DatasetQualityIssues"));
    quality->addWidget(validationIssuesTable, 1);
    auto* reportActions = new QHBoxLayout;
    auto* report = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开质量报告")));
    auto* issues = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开问题清单")));
    connect(report, &QPushButton::clicked, this, [this]() { emit reportRequested(false); });
    connect(issues, &QPushButton::clicked, this, [this]() { emit reportRequested(true); });
    reportActions->addWidget(report);
    reportActions->addWidget(issues);
    reportActions->addStretch();
    quality->addLayout(reportActions);

    auto* review = addMode(aitrain_app::workbenchText(QStringLiteral("样本复核")));
    auto* reviewTools = new QHBoxLayout;
    auto* chooseReview = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择质量或复核报告")));
    connect(chooseReview, &QPushButton::clicked, this, &DatasetWorkspacePage::chooseReviewRequested);
    reviewTools->addWidget(chooseReview);
    reviewSourceFilterCombo = new QComboBox;
    reviewSourceFilterCombo->addItem(aitrain_app::workbenchText(QStringLiteral("全部来源")), QString());
    reviewReasonFilterCombo = new QComboBox;
    reviewReasonFilterCombo->addItem(aitrain_app::workbenchText(QStringLiteral("全部问题")), QString());
    reviewSearchEdit = edit(QStringLiteral("ReviewSearch"));
    reviewSearchEdit->setPlaceholderText(aitrain_app::workbenchText(QStringLiteral("筛选已加载样本")));
    reviewTools->addWidget(reviewSourceFilterCombo);
    reviewTools->addWidget(reviewReasonFilterCombo);
    reviewTools->addWidget(reviewSearchEdit, 1);
    connect(reviewSourceFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) { emit reviewFilterChanged(); });
    connect(reviewReasonFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) { emit reviewFilterChanged(); });
    connect(reviewSearchEdit, &QLineEdit::textChanged, this, [this](const QString&) { emit reviewFilterChanged(); });
    review->addLayout(reviewTools);
    sampleReviewTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("来源")), aitrain_app::workbenchText(QStringLiteral("问题")), aitrain_app::workbenchText(QStringLiteral("类别")), aitrain_app::workbenchText(QStringLiteral("指标")), aitrain_app::workbenchText(QStringLiteral("图片")), aitrain_app::workbenchText(QStringLiteral("标签")), aitrain_app::workbenchText(QStringLiteral("说明"))});
    review->addWidget(sampleReviewTable, 1);
    sampleReviewSummaryLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("选择报告后查看问题样本。")));
    review->addWidget(sampleReviewSummaryLabel);
    auto* openReview = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看样本信息")));
    connect(openReview, &QPushButton::clicked, this, &DatasetWorkspacePage::openReviewSampleRequested);
    review->addWidget(openReview, 0, Qt::AlignRight);

    auto* annotation = addMode(aitrain_app::workbenchText(QStringLiteral("修复与标注")));
    annotationToolStatusLabel = workbenchHint(xAnyLabelingStatusText());
    annotation->addWidget(annotationToolStatusLabel);
    datasetRepairLoopLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("先检查数据质量，再依据修复清单创建外部标注会话。")));
    annotation->addWidget(datasetRepairLoopLabel);
    datasetRepairLoopTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("环节")), aitrain_app::workbenchText(QStringLiteral("状态")), aitrain_app::workbenchText(QStringLiteral("下一步"))});
    annotation->addWidget(datasetRepairLoopTable, 1);
    auto* annotationActions = new QHBoxLayout;
    auto* createSession = workbenchButton(aitrain_app::workbenchText(QStringLiteral("创建标注会话")), QStringLiteral("CreateAnnotationSessionButton"));
    auto* syncSession = workbenchButton(aitrain_app::workbenchText(QStringLiteral("同步标注结果")), QStringLiteral("SyncAnnotationSessionButton"));
    connect(createSession, &QPushButton::clicked, this, &DatasetWorkspacePage::createAnnotationRequested);
    connect(syncSession, &QPushButton::clicked, this, &DatasetWorkspacePage::syncAnnotationRequested);
    annotationActions->addWidget(createSession);
    annotationActions->addWidget(syncSession);
    annotationActions->addStretch();
    annotation->addLayout(annotationActions);

    auto* technical = addMode(aitrain_app::workbenchText(QStringLiteral("技术详情")));
    auto* technicalPanel = new InfoPanel(aitrain_app::workbenchText(QStringLiteral("当前快照的持久化身份")));
    auto* technicalForm = form(technicalPanel);
    dataQualityDatasetIdEdit = edit(QStringLiteral("DataQualityDatasetId"));
    dataQualityDatasetVersionIdEdit = edit(QStringLiteral("DataQualityDatasetVersionId"));
    dataQualitySnapshotIdEdit = edit(QStringLiteral("DataQualitySnapshotId"));
    dataQualitySnapshotArtifactIdEdit = edit(QStringLiteral("DataQualitySnapshotArtifactId"));
    for (QLineEdit* value : {dataQualityDatasetIdEdit, dataQualityDatasetVersionIdEdit, dataQualitySnapshotIdEdit, dataQualitySnapshotArtifactIdEdit}) value->setReadOnly(true);
    technicalForm->addRow(aitrain_app::workbenchText(QStringLiteral("数据集 ID")), dataQualityDatasetIdEdit);
    technicalForm->addRow(aitrain_app::workbenchText(QStringLiteral("版本 ID")), dataQualityDatasetVersionIdEdit);
    technicalForm->addRow(aitrain_app::workbenchText(QStringLiteral("快照 ID")), dataQualitySnapshotIdEdit);
    technicalForm->addRow(aitrain_app::workbenchText(QStringLiteral("产物 ID")), dataQualitySnapshotArtifactIdEdit);
    technical->addWidget(technicalPanel);
    validationOutput = new QPlainTextEdit;
    validationOutput->setReadOnly(true);
    validationOutput->setMinimumHeight(0);
    technical->addWidget(validationOutput, 1);
    setSelectionAvailable(false);
    showView(Catalog);
}

void DatasetWorkspacePage::setSelectionAvailable(bool available)
{
    for (QPushButton* action : selectionActions_) action->setEnabled(available);
}

void DatasetWorkspacePage::showView(View view)
{
    setMode(view, view == Quality || view == Technical ? Detail : Catalog);
}
