#include "WorkbenchTranslation.h"
#include "ModelRegistryPage.h"
#include "MainWindowSupport.h"
#include <QDateTime>
#include <QDir>
#include <QFormLayout>
#include <QLineEdit>
#include <QSignalBlocker>
using namespace aitrain_app;

namespace {
QString modelName(const ModelPackageListItem& item)
{
    return QStringLiteral("%1 · %2").arg(taskTypeLabel(item.taskType), item.createdAt);
}
QString validationLabel(const QString& state)
{
    if (state.isEmpty()) return aitrain_app::workbenchText(QStringLiteral("尚无独立验证"));
    if (state == QStringLiteral("succeeded")) return aitrain_app::workbenchText(QStringLiteral("已完成"));
    if (state == QStringLiteral("failed")) return aitrain_app::workbenchText(QStringLiteral("失败"));
    if (state == QStringLiteral("canceled")) return aitrain_app::workbenchText(QStringLiteral("已取消"));
    return aitrain_app::workbenchText(QStringLiteral("验证中"));
}
}

ModelRegistryWorkspacePage::ModelRegistryWorkspacePage(QWidget* parent)
    : WorkspaceViewHost(parent)
{
    setObjectName(QStringLiteral("ModelRegistryWorkspacePage"));
    auto* refresh = workbenchButton(aitrain_app::workbenchText(QStringLiteral("刷新")));
    auto* reports = workbenchButton(aitrain_app::workbenchText(QStringLiteral("验收报告")));
    auto* create = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入模型")), QStringLiteral("ModelImportNew"), true);
    toolbar->addWidget(refresh); toolbar->addWidget(reports); toolbar->addWidget(create);
    connect(refresh, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::refreshRequested);
    connect(reports, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::reportsRequested);
    connect(create, &QPushButton::clicked, this, [this]() { setMode(Import); });
    auto* catalog = addMode(aitrain_app::workbenchText(QStringLiteral("模型目录")));
    catalog->addWidget(catalogSearchField(QStringLiteral("ModelCatalogSearch"), aitrain_app::workbenchText(QStringLiteral("搜索整个项目：模型族、任务、来源或日期"))));
    summaryLabel_ = workbenchHint(); catalog->addWidget(summaryLabel_);
    packageTable_ = workbenchTable({aitrain_app::workbenchText(QStringLiteral("名称")), aitrain_app::workbenchText(QStringLiteral("任务")), aitrain_app::workbenchText(QStringLiteral("来源")), aitrain_app::workbenchText(QStringLiteral("格式")), aitrain_app::workbenchText(QStringLiteral("最近验证"))});
    packageTable_->setObjectName(QStringLiteral("ModelPackageTable")); catalog->addWidget(packageTable_, 1);
    auto* actions = new QHBoxLayout;
    auto* detail = workbenchButton(aitrain_app::workbenchText(QStringLiteral("查看详情")));
    auto* run = workbenchButton(aitrain_app::workbenchText(QStringLiteral("验证与交付")));
    moreButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("载入更多")));
    actions->addWidget(detail); actions->addWidget(run); actions->addStretch(); actions->addWidget(moreButton); catalog->addLayout(actions);
    connect(detail, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::showDetails);
    connect(packageTable_, &QTableWidget::cellDoubleClicked, this, [this](int, int) { showDetails(); });
    const auto runSelected = [this]() { const QString id = selectedModelPackageId(); if (!id.isEmpty()) emit useForRuntimeRequested(id); };
    connect(run, &QPushButton::clicked, this, runSelected);
    connect(moreButton, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::moreRequested);
    auto* details = addMode(aitrain_app::workbenchText(QStringLiteral("模型详情")));
    detailSummary_ = workbenchHint(); detailSummary_->setTextFormat(Qt::PlainText); details->addWidget(detailSummary_); details->addStretch();
    auto* detailActions = new QHBoxLayout;
    auto* source = workbenchButton(aitrain_app::workbenchText(QStringLiteral("来源任务与报告")));
    auto* latestReport = workbenchButton(aitrain_app::workbenchText(QStringLiteral("最近验证报告")));
    latestReport->setObjectName(QStringLiteral("ModelLatestValidationButton"));
    auto* technical = workbenchButton(aitrain_app::workbenchText(QStringLiteral("技术详情")));
    auto* validate = workbenchButton(aitrain_app::workbenchText(QStringLiteral("验证与交付")), {}, true);
    detailActions->addWidget(source); detailActions->addWidget(latestReport); detailActions->addWidget(technical); detailActions->addStretch(); detailActions->addWidget(validate); details->addLayout(detailActions);
    connect(latestReport, &QPushButton::clicked, this, [this]() {
        for (const auto& item : packages_) if (item.modelPackageId == selectedModelPackageId() && !item.latestValidationTaskId.isEmpty()) emit latestValidationRequested(item.latestValidationTaskId);
    });
    connect(validate, &QPushButton::clicked, this, runSelected);
    connect(technical, &QPushButton::clicked, this, [this]() { setMode(Technical, Detail); });
    connect(source, &QPushButton::clicked, this, [this]() {
        for (const auto& item : packages_) if (item.modelPackageId == selectedModelPackageId()) emit sourceTaskRequested(item.sourceTaskId);
    });
    auto* form = addMode(aitrain_app::workbenchText(QStringLiteral("导入模型")));
    form->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("选择模型文件和对应的模型说明。模型语义、张量约定及来源由说明文件提供，导入后会校验并登记。"))));
    sourceEdit_ = new QLineEdit; sourceEdit_->setObjectName(QStringLiteral("ModelImportSource"));
    manifestEdit_ = new QLineEdit; manifestEdit_->setObjectName(QStringLiteral("ModelImportManifest"));
    const auto path = [this](QLineEdit* edit, bool model) {
        auto* row = new QWidget; auto* layout = new QHBoxLayout(row); layout->setContentsMargins(0, 0, 0, 0);
        edit->setMinimumWidth(0); layout->addWidget(edit, 1); auto* browse = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择文件"))); layout->addWidget(browse);
        connect(browse, &QPushButton::clicked, this, model ? &ModelRegistryWorkspacePage::browseSourceRequested : &ModelRegistryWorkspacePage::browseManifestRequested); return row;
    };
    auto* fields = new QFormLayout; fields->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    fields->addRow(aitrain_app::workbenchText(QStringLiteral("模型文件")), path(sourceEdit_, true)); fields->addRow(aitrain_app::workbenchText(QStringLiteral("模型说明 JSON")), path(manifestEdit_, false)); form->addLayout(fields);
    auto* editor = workbenchButton(aitrain_app::workbenchText(QStringLiteral("检查 / 编辑模型说明")));
    connect(editor, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::editManifestRequested); form->addWidget(editor, 0, Qt::AlignLeft);
    importStatusLabel_ = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未导入。"))); form->addWidget(importStatusLabel_); form->addStretch();
    auto* import = workbenchButton(aitrain_app::workbenchText(QStringLiteral("导入模型")), QStringLiteral("ModelImportStart"), true); form->addWidget(import, 0, Qt::AlignRight);
    connect(import, &QPushButton::clicked, this, &ModelRegistryWorkspacePage::importRequested);
    auto* tech = addMode(aitrain_app::workbenchText(QStringLiteral("模型技术详情")));
    technicalSummary_ = workbenchHint(); technicalSummary_->setTextFormat(Qt::PlainText); technicalSummary_->setTextInteractionFlags(Qt::TextSelectableByMouse); tech->addWidget(technicalSummary_); tech->addStretch();
    connect(views, &QStackedWidget::currentChanged, this, [create, reports, refresh](int index) { create->setVisible(index == Catalog); reports->setVisible(index == Catalog); refresh->setVisible(index == Catalog); });
    setMode(Catalog);
}

QString ModelRegistryWorkspacePage::sourceFilePath() const { return QDir::fromNativeSeparators(sourceEdit_->text().trimmed()); }
QString ModelRegistryWorkspacePage::manifestFilePath() const { return QDir::fromNativeSeparators(manifestEdit_->text().trimmed()); }
QString ModelRegistryWorkspacePage::selectedModelPackageId() const
{
    const auto* item = packageTable_->item(packageTable_->currentRow(), 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}
void ModelRegistryWorkspacePage::setSourceFilePath(const QString& path) { sourceEdit_->setText(QDir::toNativeSeparators(path)); }
void ModelRegistryWorkspacePage::setManifestFilePath(const QString& path) { manifestEdit_->setText(QDir::toNativeSeparators(path)); }
void ModelRegistryWorkspacePage::setImportStatus(const QString& status) { importStatusLabel_->setText(status); }
void ModelRegistryWorkspacePage::selectPackage(const QString& id, bool openDetail)
{
    for (int row = 0; row < packageTable_->rowCount(); ++row) if (packageTable_->item(row, 0)->data(Qt::UserRole).toString() == id) {
        packageTable_->selectRow(row); if (openDetail) showDetails(); return;
    }
}
void ModelRegistryWorkspacePage::showDetails()
{
    for (const auto& item : packages_) if (item.modelPackageId == selectedModelPackageId()) {
        detailSummary_->setText(aitrain_app::workbenchText(QStringLiteral("%1\n\n任务：%2\n来源：%3\n格式：%4\n登记时间：%5\n\n可用路线：%6\n\n%7"))
            .arg(modelName(item), taskTypeLabel(item.taskType), backendLabel(item.sourceBackend), item.artifactFormat, item.createdAt, item.runtimeRoutes.join(QStringLiteral("、")), item.limitations.join(QStringLiteral("\n"))));
        technicalSummary_->setText(aitrain_app::workbenchText(QStringLiteral("模型包：%1\n来源任务：%2\n来源快照：%3\n模型产物：%4\nSHA-256：%5\n解码器：%6\n导出器：%7"))
            .arg(item.modelPackageId, item.sourceTaskId, item.sourceSnapshotId, item.sourceArtifactId, item.sourceArtifactSha256, item.decoder, item.exporterVersion));
        detailSummary_->setText(detailSummary_->text() + aitrain_app::workbenchText(QStringLiteral("\n\n最近独立验证：%1\n模型合同：%2")).arg(validationLabel(item.latestValidationState), item.verified ? aitrain_app::workbenchText(QStringLiteral("已校验")) : aitrain_app::workbenchText(QStringLiteral("待校验"))));
        findChild<QPushButton*>(QStringLiteral("ModelLatestValidationButton"))->setEnabled(!item.latestValidationTaskId.isEmpty());
        setMode(Detail); return;
    }
}
void ModelRegistryWorkspacePage::renderPackages(const QVector<ModelPackageListItem>& packages, const QString& status)
{
    const QString selected = selectedModelPackageId();
    packages_ = packages; summaryLabel_->setText(status);
    const QSignalBlocker blocker(packageTable_); packageTable_->setRowCount(0);
    for (const auto& item : packages) {
        const int row = packageTable_->rowCount(); packageTable_->insertRow(row);
        const QStringList values = {modelName(item), taskTypeLabel(item.taskType), backendLabel(item.sourceBackend), item.artifactFormat,
            validationLabel(item.latestValidationState)};
        for (int c = 0; c < values.size(); ++c) packageTable_->setItem(row, c, new QTableWidgetItem(values[c]));
        packageTable_->item(row, 0)->setData(Qt::UserRole, item.modelPackageId);
    }
    selectPackage(selected);
}
