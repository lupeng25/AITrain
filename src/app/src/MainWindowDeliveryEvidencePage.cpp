#include "MainWindow.h"

#include "DeliveryEvidencePage.h"
#include "DeliveryEvidencePageController.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QTableWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildDeliveryEvidencePanel()
{
    deliveryEvidencePage_ = new DeliveryEvidenceWorkspacePage;
    auto* page = deliveryEvidencePage_;
    page->setWidgetResizable(true);
    page->setFrameShape(QFrame::NoFrame);
    page->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);
    auto* splitter = new QSplitter(Qt::Horizontal);

    auto* summaryPanel = new InfoPanel(tr("验收证据"));
    deliveryEvidencePage_->acceptanceSummaryLabel =
        inlineStatusLabel(tr("等待导入或运行验收证据。"));
    deliveryEvidencePage_->acceptanceSummaryLabel->setObjectName(
        QStringLiteral("DeliveryAcceptanceSummary"));
    connect(deliveryEvidencePage_->acceptanceSummaryLabel, &QObject::destroyed,
        this, [this]() { deliveryEvidencePage_->acceptanceSummaryLabel = nullptr; });
    summaryPanel->bodyLayout()->addWidget(deliveryEvidencePage_->acceptanceSummaryLabel);
    deliveryEvidencePage_->acceptanceTable = new QTableWidget(0, 4);
    deliveryEvidencePage_->acceptanceTable->setObjectName(
        QStringLiteral("DeliveryAcceptanceTable"));
    connect(deliveryEvidencePage_->acceptanceTable, &QObject::destroyed,
        this, [this]() { deliveryEvidencePage_->acceptanceTable = nullptr; });
    deliveryEvidencePage_->acceptanceTable->setHorizontalHeaderLabels(QStringList()
        << tr("项目") << tr("状态") << tr("证据") << tr("说明"));
    configureTable(deliveryEvidencePage_->acceptanceTable);
    deliveryEvidencePage_->acceptanceTable->horizontalHeader()->setSectionResizeMode(
        0, QHeaderView::ResizeToContents);
    deliveryEvidencePage_->acceptanceTable->horizontalHeader()->setSectionResizeMode(
        1, QHeaderView::ResizeToContents);
    deliveryEvidencePage_->acceptanceTable->horizontalHeader()->setSectionResizeMode(
        2, QHeaderView::Stretch);
    deliveryEvidencePage_->acceptanceTable->horizontalHeader()->setSectionResizeMode(
        3, QHeaderView::Stretch);
    deliveryEvidencePage_->acceptanceTable->setMinimumHeight(240);
    summaryPanel->bodyLayout()->addWidget(deliveryEvidencePage_->acceptanceTable);
    auto* importButton = new QPushButton(tr("导入外部验收结果"));
    connect(importButton, &QPushButton::clicked,
        deliveryEvidencePageController_,
        &DeliveryEvidencePageController::importAcceptanceEvidence);
    summaryPanel->bodyLayout()->addWidget(importButton, 0, Qt::AlignRight);

    auto* rightStack = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightStack);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(16);
    auto* ocrPanel = new InfoPanel(tr("客户域 OCR 官方报告受控验收"));
    const auto makePathRow = [this](QLineEdit** target,
        const QString& placeholder) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        *target = new QLineEdit;
        (*target)->setPlaceholderText(placeholder);
        auto* button = new QPushButton(tr("选择文件"));
        connect(button, &QPushButton::clicked, this,
            [this, target]() {
            deliveryEvidencePageController_->browseReport(*target);
        });
        rowLayout->addWidget(*target, 1);
        rowLayout->addWidget(button);
        return row;
    };
    const auto makeIdPair = [](QLineEdit** snapshotId,
        QLineEdit** artifactId, const QString& prefix) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        *snapshotId = new QLineEdit;
        (*snapshotId)->setObjectName(prefix + QStringLiteral("SnapshotId"));
        (*snapshotId)->setPlaceholderText(QStringLiteral("SnapshotId（可选）"));
        *artifactId = new QLineEdit;
        (*artifactId)->setObjectName(
            prefix + QStringLiteral("SnapshotArtifactId"));
        (*artifactId)->setPlaceholderText(
            QStringLiteral("Snapshot ArtifactId（可选）"));
        rowLayout->addWidget(*snapshotId);
        rowLayout->addWidget(*artifactId);
        return row;
    };
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto* importTitle = new QLabel(
        tr("步骤 1：受控导入（裸路径仅允许停留在此导入边界）"));
    importTitle->setObjectName(QStringLiteral("OcrImportSectionTitle"));
    ocrPanel->bodyLayout()->addWidget(importTitle);
    form->addRow(tr("Det 原始报告"),
        makePathRow(&deliveryEvidencePage_->ocrDetReportEdit,
            tr("PaddleOCR Det 官方 JSON")));
    deliveryEvidencePage_->ocrDetReportEdit->setObjectName(
        QStringLiteral("OcrDetRawReportPath"));
    form->addRow(tr("Det Snapshot"),
        makeIdPair(&deliveryEvidencePage_->ocrDetSnapshotIdEdit,
            &deliveryEvidencePage_->ocrDetSnapshotArtifactIdEdit, QStringLiteral("OcrDet")));
    form->addRow(tr("Rec 原始报告"),
        makePathRow(&deliveryEvidencePage_->ocrRecReportEdit,
            tr("PaddleOCR Rec 官方 JSON（含 accuracy/CER）")));
    deliveryEvidencePage_->ocrRecReportEdit->setObjectName(
        QStringLiteral("OcrRecRawReportPath"));
    form->addRow(tr("Rec Snapshot"),
        makeIdPair(&deliveryEvidencePage_->ocrRecSnapshotIdEdit,
            &deliveryEvidencePage_->ocrRecSnapshotArtifactIdEdit, QStringLiteral("OcrRec")));
    form->addRow(tr("System 原始报告"),
        makePathRow(&deliveryEvidencePage_->ocrSystemReportEdit,
            tr("PaddleOCR System 官方 JSON（必须含真实 accuracy）")));
    deliveryEvidencePage_->ocrSystemReportEdit->setObjectName(
        QStringLiteral("OcrSystemRawReportPath"));
    form->addRow(tr("System Snapshot"),
        makeIdPair(&deliveryEvidencePage_->ocrSystemSnapshotIdEdit,
            &deliveryEvidencePage_->ocrSystemSnapshotArtifactIdEdit,
            QStringLiteral("OcrSystem")));
    deliveryEvidencePage_->ocrCohortIdEdit = new QLineEdit;
    deliveryEvidencePage_->ocrCohortIdEdit->setObjectName(
        QStringLiteral("OcrAcceptanceCohortId"));
    deliveryEvidencePage_->ocrCohortIdEdit->setPlaceholderText(tr("同一验收批次标识"));
    form->addRow(tr("验收批次"), deliveryEvidencePage_->ocrCohortIdEdit);
    deliveryEvidencePage_->ocrDomainIdEdit = new QLineEdit;
    deliveryEvidencePage_->ocrDomainIdEdit->setObjectName(
        QStringLiteral("OcrCustomerDomainId"));
    deliveryEvidencePage_->ocrDomainIdEdit->setPlaceholderText(tr("客户域标识"));
    form->addRow(tr("客户域"), deliveryEvidencePage_->ocrDomainIdEdit);
    deliveryEvidencePage_->ocrEvidenceClassCombo = new QComboBox;
    deliveryEvidencePage_->ocrEvidenceClassCombo->setObjectName(
        QStringLiteral("OcrEvidenceClass"));
    deliveryEvidencePage_->ocrEvidenceClassCombo->addItems({
        QStringLiteral("customer_domain"), QStringLiteral("public"),
        QStringLiteral("generated"), QStringLiteral("smoke")});
    form->addRow(tr("证据分类"), deliveryEvidencePage_->ocrEvidenceClassCombo);
    auto* importOcrButton = primaryButton(tr("受控导入官方报告"));
    importOcrButton->setObjectName(
        QStringLiteral("ImportOcrOfficialReportsButton"));
    connect(importOcrButton, &QPushButton::clicked,
        deliveryEvidencePageController_,
        &DeliveryEvidencePageController::importOcrOfficialReports);
    form->addRow(QString(), importOcrButton);

    auto* acceptanceTitle =
        new QLabel(tr("步骤 2：仅使用已提交报告 ArtifactId 运行验收"));
    acceptanceTitle->setObjectName(
        QStringLiteral("OcrAcceptanceSectionTitle"));
    form->addRow(acceptanceTitle);
    deliveryEvidencePage_->ocrDetReportArtifactIdEdit = new QLineEdit;
    deliveryEvidencePage_->ocrDetReportArtifactIdEdit->setObjectName(
        QStringLiteral("OcrDetReportArtifactId"));
    deliveryEvidencePage_->ocrRecReportArtifactIdEdit = new QLineEdit;
    deliveryEvidencePage_->ocrRecReportArtifactIdEdit->setObjectName(
        QStringLiteral("OcrRecReportArtifactId"));
    deliveryEvidencePage_->ocrSystemReportArtifactIdEdit = new QLineEdit;
    deliveryEvidencePage_->ocrSystemReportArtifactIdEdit->setObjectName(
        QStringLiteral("OcrSystemReportArtifactId"));
    form->addRow(tr("Det 报告 ArtifactId"),
        deliveryEvidencePage_->ocrDetReportArtifactIdEdit);
    form->addRow(tr("Rec 报告 ArtifactId"),
        deliveryEvidencePage_->ocrRecReportArtifactIdEdit);
    form->addRow(tr("System 报告 ArtifactId"),
        deliveryEvidencePage_->ocrSystemReportArtifactIdEdit);
    auto* thresholdRow = new QWidget;
    auto* thresholdLayout = new QHBoxLayout(thresholdRow);
    thresholdLayout->setContentsMargins(0, 0, 0, 0);
    deliveryEvidencePage_->ocrMinDetHmeanEdit = new QLineEdit(QStringLiteral("0.50"));
    deliveryEvidencePage_->ocrMinAccEdit = new QLineEdit(QStringLiteral("0.70"));
    deliveryEvidencePage_->ocrMaxCerEdit = new QLineEdit(QStringLiteral("0.30"));
    deliveryEvidencePage_->ocrMinSystemAccEdit = new QLineEdit(QStringLiteral("0.70"));
    thresholdLayout->addWidget(new QLabel(tr("Det hmean ≥")));
    thresholdLayout->addWidget(deliveryEvidencePage_->ocrMinDetHmeanEdit);
    thresholdLayout->addWidget(new QLabel(tr("Rec accuracy ≥")));
    thresholdLayout->addWidget(deliveryEvidencePage_->ocrMinAccEdit);
    thresholdLayout->addWidget(new QLabel(tr("CER ≤")));
    thresholdLayout->addWidget(deliveryEvidencePage_->ocrMaxCerEdit);
    thresholdLayout->addWidget(new QLabel(tr("System accuracy ≥")));
    thresholdLayout->addWidget(deliveryEvidencePage_->ocrMinSystemAccEdit);
    form->addRow(tr("门槛"), thresholdRow);
    ocrPanel->bodyLayout()->addLayout(form);
    deliveryEvidencePage_->ocrStatusLabel =
        inlineStatusLabel(tr("尚未导入官方报告或运行 OCR Acceptance。"));
    deliveryEvidencePage_->ocrStatusLabel->setObjectName(
        QStringLiteral("OcrAcceptanceStatus"));
    ocrPanel->bodyLayout()->addWidget(deliveryEvidencePage_->ocrStatusLabel);
    auto* runButton = primaryButton(tr("运行 OCR Acceptance"));
    runButton->setObjectName(
        QStringLiteral("RunOcrAcceptanceWorkflowButton"));
    connect(runButton, &QPushButton::clicked,
        deliveryEvidencePageController_,
        &DeliveryEvidencePageController::runOcrAcceptance);
    ocrPanel->bodyLayout()->addWidget(runButton, 0, Qt::AlignRight);
    ocrPanel->bodyLayout()->addWidget(mutedLabel(tr(
        "Total-Text、generated smoke 和 .deps 示例只能证明流程可跑，"
        "不能作为客户域生产 OCR 精度证明。")));

    auto* diagnosticsPanel = new InfoPanel(tr("诊断包"));
    deliveryEvidencePage_->diagnosticsStatusLabel = inlineStatusLabel(tr("诊断包尚未生成。"));
    diagnosticsPanel->bodyLayout()->addWidget(deliveryEvidencePage_->diagnosticsStatusLabel);
    diagnosticsPanel->bodyLayout()->addWidget(mutedLabel(tr(
        "诊断包包含 Worker self-check、环境 profile、GPU/驱动、最近任务日志、"
        "失败请求、artifact index、内置能力状态和授权摘要。")));
    auto* diagnosticsButton = primaryButton(tr("一键诊断包"));
    connect(diagnosticsButton, &QPushButton::clicked,
        deliveryEvidencePageController_,
        &DeliveryEvidencePageController::collectDiagnostics);
    diagnosticsPanel->bodyLayout()->addWidget(
        diagnosticsButton, 0, Qt::AlignRight);

    auto* left = new QWidget;
    auto* leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->addWidget(summaryPanel);
    rightLayout->addWidget(ocrPanel, 4);
    rightLayout->addWidget(diagnosticsPanel, 1);
    splitter->addWidget(left);
    splitter->addWidget(rightStack);
    splitter->setStretchFactor(0, 4);
    splitter->setStretchFactor(1, 5);
    layout->addWidget(splitter);
    page->setWidget(content);
    deliveryEvidencePageController_->attach(deliveryEvidencePage_);
    return page;
}
