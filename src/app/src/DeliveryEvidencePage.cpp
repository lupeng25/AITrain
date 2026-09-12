#include "WorkbenchTranslation.h"
#include "DeliveryEvidencePage.h"
#include <QComboBox>
#include <QDoubleValidator>
#include <QFormLayout>
#include <QGridLayout>
#include <QLineEdit>
using namespace aitrain_app;

DeliveryEvidenceWorkspacePage::DeliveryEvidenceWorkspacePage(QWidget* parent) : WorkspaceViewHost(parent)
{
    setObjectName(QStringLiteral("DeliveryEvidenceWorkspacePage"));
    auto* refresh = workbenchButton(aitrain_app::workbenchText(QStringLiteral("刷新"))); auto* import = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入官方报告")), {}, true);
    auto* accept = workbenchButton(aitrain_app::workbenchText(QStringLiteral("运行 OCR 验收"))); toolbar->addWidget(refresh); toolbar->addWidget(accept); toolbar->addWidget(import);
    connect(refresh, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::refreshRequested);
    connect(import, &QPushButton::clicked, this, [this]() { setMode(Import); });
    connect(accept, &QPushButton::clicked, this, [this]() { setMode(Acceptance); });
    auto* catalog = addMode(aitrain_app::workbenchText(QStringLiteral("验收报告")));
    catalog->addWidget(catalogSearchField(QStringLiteral("EvidenceCatalogSearch"), aitrain_app::workbenchText(QStringLiteral("搜索整个项目：报告类型、任务或文件名"))));
    acceptanceSummaryLabel = workbenchHint(); acceptanceSummaryLabel->setObjectName(QStringLiteral("DeliveryAcceptanceSummary")); catalog->addWidget(acceptanceSummaryLabel);
    acceptanceTable = workbenchTable({aitrain_app::workbenchText(QStringLiteral("报告类别")), aitrain_app::workbenchText(QStringLiteral("运行状态")), aitrain_app::workbenchText(QStringLiteral("证据校验")), aitrain_app::workbenchText(QStringLiteral("说明"))}); acceptanceTable->setObjectName(QStringLiteral("DeliveryAcceptanceTable")); catalog->addWidget(acceptanceTable, 1);
    auto* actions = new QHBoxLayout; auto* open = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看任务与报告"))); auto* external = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入外部验收结果"))); moreButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("载入更多")));
    actions->addWidget(open); actions->addWidget(external); actions->addStretch(); actions->addWidget(moreButton); catalog->addLayout(actions);
    connect(open, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::openSelectedRequested);
    connect(external, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::externalImportRequested);
    connect(moreButton, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::moreRequested);
    auto* source = addMode(aitrain_app::workbenchText(QStringLiteral("导入 PaddleOCR 官方报告")));
    auto* fields = new QFormLayout; fields->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow); fields->setVerticalSpacing(8);
    QLineEdit** reports[] = {&ocrDetReportEdit, &ocrRecReportEdit, &ocrSystemReportEdit};
    QLineEdit** snapshots[] = {&ocrDetSnapshotIdEdit, &ocrRecSnapshotIdEdit, &ocrSystemSnapshotIdEdit};
    QLineEdit** artifacts[] = {&ocrDetSnapshotArtifactIdEdit, &ocrRecSnapshotArtifactIdEdit, &ocrSystemSnapshotArtifactIdEdit};
    const QStringList names = {QStringLiteral("Det"), QStringLiteral("Rec"), QStringLiteral("System")};
    for (int i = 0; i < 3; ++i) {
        *reports[i] = new QLineEdit; (*reports[i])->setObjectName(QStringLiteral("Ocr%1RawReportPath").arg(names[i])); (*reports[i])->setMinimumWidth(0);
        auto* row = new QWidget; auto* layout = new QHBoxLayout(row); layout->setContentsMargins(0, 0, 0, 0); layout->addWidget(*reports[i], 1);
        auto* browse = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择文件"))); layout->addWidget(browse); QLineEdit* target = *reports[i];
        connect(browse, &QPushButton::clicked, this, [this, target]() { emit browseRequested(target); }); fields->addRow(names[i] + aitrain_app::workbenchText(QStringLiteral(" 报告")), row);
        *snapshots[i] = new QLineEdit(this); (*snapshots[i])->setObjectName(QStringLiteral("Ocr%1SnapshotId").arg(names[i])); (*snapshots[i])->hide();
        *artifacts[i] = new QLineEdit(this); (*artifacts[i])->setObjectName(QStringLiteral("Ocr%1SnapshotArtifactId").arg(names[i])); (*artifacts[i])->hide();
        auto* binding = new QWidget; auto* bindLayout = new QHBoxLayout(binding); bindLayout->setContentsMargins(0, 0, 0, 0); snapshotLabels[i] = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未绑定数据版本")));
        auto* bind = workbenchButton(aitrain_app::workbenchText(QStringLiteral("绑定数据版本"))); bindLayout->addWidget(snapshotLabels[i], 1); bindLayout->addWidget(bind); fields->addRow(names[i] + aitrain_app::workbenchText(QStringLiteral(" 数据")), binding);
        connect(bind, &QPushButton::clicked, this, [this, i]() { emit bindSnapshotRequested(i); });
    }
    ocrCohortIdEdit = new QLineEdit; ocrCohortIdEdit->setObjectName(QStringLiteral("OcrAcceptanceCohortId")); ocrDomainIdEdit = new QLineEdit; ocrDomainIdEdit->setObjectName(QStringLiteral("OcrCustomerDomainId"));
    auto* identity = new QWidget; auto* identityLayout = new QHBoxLayout(identity); identityLayout->setContentsMargins(0, 0, 0, 0); identityLayout->addWidget(ocrCohortIdEdit, 1); identityLayout->addWidget(new QLabel(aitrain_app::workbenchText(QStringLiteral("客户域")))); identityLayout->addWidget(ocrDomainIdEdit, 1); fields->addRow(aitrain_app::workbenchText(QStringLiteral("验收批次")), identity);
    ocrEvidenceClassCombo = new QComboBox; ocrEvidenceClassCombo->setObjectName(QStringLiteral("OcrEvidenceClass")); ocrEvidenceClassCombo->addItems({QStringLiteral("customer_domain"), QStringLiteral("public"), QStringLiteral("generated"), QStringLiteral("smoke")}); fields->addRow(aitrain_app::workbenchText(QStringLiteral("证据分类")), ocrEvidenceClassCombo);
    source->addLayout(fields); source->addStretch(); auto* startImport = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入三份官方报告")), QStringLiteral("ImportOcrOfficialReportsButton"), true); source->addWidget(startImport, 0, Qt::AlignRight);
    connect(startImport, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::importOcrRequested);
    auto* acceptance = addMode(aitrain_app::workbenchText(QStringLiteral("运行 OCR 验收")));
    auto* acceptedFields = new QFormLayout; QLineEdit** ids[] = {&ocrDetReportArtifactIdEdit, &ocrRecReportArtifactIdEdit, &ocrSystemReportArtifactIdEdit};
    for (int i = 0; i < 3; ++i) {
        *ids[i] = new QLineEdit(this); (*ids[i])->setObjectName(QStringLiteral("Ocr%1ReportArtifactId").arg(names[i])); (*ids[i])->hide();
        auto* row = new QWidget; auto* layout = new QHBoxLayout(row); layout->setContentsMargins(0, 0, 0, 0); reportLabels[i] = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未选择报告")));
        auto* choose = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择已提交报告"))); layout->addWidget(reportLabels[i], 1); layout->addWidget(choose); acceptedFields->addRow(names[i], row);
        connect(choose, &QPushButton::clicked, this, [this, i]() { emit selectReportRequested(i); });
    }
    acceptance->addLayout(acceptedFields);
    auto* thresholds = new QGridLayout; QLineEdit** limits[] = {&ocrMinDetHmeanEdit, &ocrMinAccEdit, &ocrMaxCerEdit, &ocrMinSystemAccEdit};
    const QStringList labels = {QStringLiteral("Det hmean ≥"), QStringLiteral("Rec accuracy ≥"), QStringLiteral("Rec CER ≤"), QStringLiteral("System accuracy ≥")};
    const QStringList defaults = {QStringLiteral("0.50"), QStringLiteral("0.70"), QStringLiteral("0.30"), QStringLiteral("0.70")};
    for (int i = 0; i < 4; ++i) { *limits[i] = new QLineEdit(defaults[i]); (*limits[i])->setValidator(new QDoubleValidator(0, 1, 6, *limits[i])); thresholds->addWidget(new QLabel(labels[i]), i / 2, (i % 2) * 2); thresholds->addWidget(*limits[i], i / 2, (i % 2) * 2 + 1); }
    acceptance->addLayout(thresholds); acceptance->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("公共数据、生成数据和 smoke 报告只能证明对应流程，不能替代客户域精度证据。")))); acceptance->addStretch();
    auto* run = workbenchButton(aitrain_app::workbenchText(QStringLiteral("运行 OCR 验收")), QStringLiteral("RunOcrAcceptanceWorkflowButton"), true); acceptance->addWidget(run, 0, Qt::AlignRight); connect(run, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::acceptanceRequested);
    auto* diagnostics = addMode(aitrain_app::workbenchText(QStringLiteral("诊断包"))); diagnosticsStatusLabel = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未生成诊断包。"))); diagnostics->addWidget(diagnosticsStatusLabel);
    diagnostics->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("收集环境检查、GPU / 驱动信息、最近任务日志、失败请求和授权摘要，用于排查运行问题。")))); diagnostics->addStretch();
    auto* collect = workbenchButton(aitrain_app::workbenchText(QStringLiteral("生成诊断包")), {}, true); diagnostics->addWidget(collect, 0, Qt::AlignRight); connect(collect, &QPushButton::clicked, this, &DeliveryEvidenceWorkspacePage::diagnosticsRequested);
    ocrStatusLabel = workbenchHint(); ocrStatusLabel->setObjectName(QStringLiteral("OcrAcceptanceStatus")); static_cast<QVBoxLayout*>(layout())->addWidget(ocrStatusLabel);
    connect(views, &QStackedWidget::currentChanged, this, [refresh, import, accept](int index) { refresh->setVisible(index == Catalog); import->setVisible(index == Catalog); accept->setVisible(index == Catalog); });
    setMode(Catalog);
}
