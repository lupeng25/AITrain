#include "WorkbenchTranslation.h"
#include "RuntimeDeliveryPage.h"
#include "MainWindowSupport.h"
#include <QComboBox>
#include <QDateTime>
#include <QFormLayout>
#include <QSignalBlocker>
using namespace aitrain_app;

RuntimeDeliveryWorkspacePage::RuntimeDeliveryWorkspacePage(QWidget* parent) : WorkspaceViewHost(parent)
{
    setObjectName(QStringLiteral("RuntimeDeliveryWorkspacePage"));
    auto* form = addMode(aitrain_app::workbenchText(QStringLiteral("验证与交付")));
    model_ = new QComboBox; model_->setObjectName(QStringLiteral("DeploymentModelPackageCombo"));
    route_ = new QComboBox; route_->setObjectName(QStringLiteral("DeploymentRuntimeRouteCombo"));
    auto* fields = new QFormLayout; fields->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    fields->addRow(aitrain_app::workbenchText(QStringLiteral("模型")), model_); fields->addRow(aitrain_app::workbenchText(QStringLiteral("运行路线")), route_); form->addLayout(fields);
    auto* sampleRow = new QHBoxLayout; sample_ = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未选择验证样本。")));
    auto* choose = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择数据版本与样本")), QStringLiteral("DeliverySelectSample")); sampleRow->addWidget(sample_, 1); sampleRow->addWidget(choose); form->addLayout(sampleRow);
    reasons_ = workbenchHint(); reasons_->setObjectName(QStringLiteral("RuntimeRouteReasons")); form->addWidget(reasons_);
    form->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("将执行模型校验、样本推理、计时和部署检查，并生成交付报告。样本计时用于本次检查，不代表性能 SLA。"))));
    form->addStretch(); auto* run = workbenchButton(aitrain_app::workbenchText(QStringLiteral("开始验证与交付")), QStringLiteral("RuntimeDeliveryStart"), true); form->addWidget(run, 0, Qt::AlignRight);
    connect(choose, &QPushButton::clicked, this, &RuntimeDeliveryWorkspacePage::selectSampleRequested);
    connect(run, &QPushButton::clicked, this, [this]() { emit runRequested(RuntimeDeliveryMode::DeploymentValidation); });
    connect(model_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        binding_.modelPackageId = model_->currentData().toString();
        emit modelSelectionChanged(RuntimeDeliveryMode::DeploymentValidation, binding_.modelPackageId);
    });
    auto* result = addMode(aitrain_app::workbenchText(QStringLiteral("验证结果")));
    resultSummary = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未运行。"))); resultSummary->setObjectName(QStringLiteral("DeploymentResultSummary")); result->addWidget(resultSummary);
    overlay = new ImagePreviewLabel; overlay->setText(aitrain_app::workbenchText(QStringLiteral("尚无已提交的可视化结果。"))); overlay->setObjectName(QStringLiteral("InferenceOverlayCanvas")); result->addWidget(overlay, 1);
    auto* actions = new QHBoxLayout; taskButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("任务详情与产物"))); reportButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开交付报告")), {}, true);
    actions->addWidget(taskButton); actions->addStretch(); actions->addWidget(reportButton); result->addLayout(actions);
    taskButton->setEnabled(false); reportButton->setEnabled(false);
    connect(taskButton, &QPushButton::clicked, this, &RuntimeDeliveryWorkspacePage::taskRequested);
    connect(reportButton, &QPushButton::clicked, this, &RuntimeDeliveryWorkspacePage::reportRequested);
    setMode(0);
}
RuntimeDeliveryFormData RuntimeDeliveryWorkspacePage::formData(RuntimeDeliveryMode) const
{
    auto data = binding_; data.modelPackageId = model_->currentData().toString(); data.runtimeRoute = route_->currentData().toString(); return data;
}
QString RuntimeDeliveryWorkspacePage::selectedModelPackageId(RuntimeDeliveryMode) const { return model_->currentData().toString(); }
void RuntimeDeliveryWorkspacePage::setModelPackages(const QVector<ModelPackageListItem>& packages)
{
    const QString previous = binding_.modelPackageId; const QSignalBlocker blocker(model_);
    model_->clear(); model_->addItem(aitrain_app::workbenchText(QStringLiteral("请选择已登记模型")), QString());
    for (const auto& item : packages) model_->addItem(QStringLiteral("%1 · %2 · %3").arg(taskTypeLabel(item.taskType), item.artifactFormat, item.createdAt), item.modelPackageId);
    const int index = model_->findData(previous); model_->setCurrentIndex(index < 0 ? 0 : index);
}
void RuntimeDeliveryWorkspacePage::setRouteEvaluation(RuntimeDeliveryMode, const QStringList& routes, const QStringList& reasons)
{
    const QString previous = route_->currentData().toString(); const QSignalBlocker blocker(route_); route_->clear();
    route_->addItem(routes.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("当前无可执行路线")) : aitrain_app::workbenchText(QStringLiteral("请选择运行路线")), QString());
    for (const QString& route : routes) route_->addItem(route, route);
    const int index = route_->findData(previous); route_->setCurrentIndex(index >= 0 ? index : routes.size() == 1 ? 1 : 0);
    reasons_->setText(reasons.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("选择模型后显示本机运行条件。")) : reasons.join(QLatin1Char('\n')));
}
void RuntimeDeliveryWorkspacePage::setDatasetSelection(const DatasetSelection& selected)
{
    binding_.sampleDatasetId = selected.snapshot.datasetId.toString(); binding_.sampleDatasetVersionId = selected.snapshot.datasetVersionId.toString();
    binding_.sampleSnapshotId = selected.snapshot.snapshotId.toString(); binding_.sampleSnapshotArtifactId = selected.snapshot.artifactId.toString();
    binding_.sampleRelativePath = selected.sampleRelativePath;
    sample_->setText(QStringLiteral("%1\n%2").arg(selected.displayName, selected.sampleRelativePath));
}
void RuntimeDeliveryWorkspacePage::clearContext()
{
    binding_ = {}; sample_->setText(aitrain_app::workbenchText(QStringLiteral("尚未选择验证样本。"))); resultSummary->setText(aitrain_app::workbenchText(QStringLiteral("尚未运行。")));
    overlay->setText(aitrain_app::workbenchText(QStringLiteral("尚无已提交的可视化结果。"))); reportButton->setEnabled(false); taskButton->setEnabled(false); setMode(0);
}
void RuntimeDeliveryWorkspacePage::setRunning(RuntimeDeliveryMode)
{
    resultSummary->setText(aitrain_app::workbenchText(QStringLiteral("任务已提交，等待验证结果。"))); overlay->setText(aitrain_app::workbenchText(QStringLiteral("正在验证，等待已提交的可视化结果。"))); reportButton->setEnabled(false); taskButton->setEnabled(true); setMode(1);
}
void RuntimeDeliveryWorkspacePage::selectModelPackageForInference(const QString& id) { const int index = model_->findData(id); if (index >= 0) model_->setCurrentIndex(index); setMode(0); }
void RuntimeDeliveryWorkspacePage::showTab(int) { setMode(0); }
