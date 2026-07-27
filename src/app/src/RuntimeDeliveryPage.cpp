#include "RuntimeDeliveryPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QComboBox>
#include <QDir>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QTabWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

RuntimeDeliveryWorkspacePage::RuntimeDeliveryWorkspacePage(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);
    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("RUNTIME DELIVERY"),
        tr("Runtime Delivery"),
        tr("基于已登记模型包、产品合同和当前环境运行推理、Benchmark 与部署验证。"),
        nullptr,
        QStringList() << QStringLiteral("ONNX Runtime")
                      << QStringLiteral("NCNN")
                      << QStringLiteral("TensorRT")
                      << tr("六步交付")));
    tabs_ = new QTabWidget;
    tabs_->setObjectName(QStringLiteral("DeploymentTabs"));
    tabs_->addTab(buildForm(RuntimeDeliveryMode::DeploymentValidation),
        tr("部署验证"));
    tabs_->addTab(buildForm(RuntimeDeliveryMode::InferenceValidation),
        tr("推理验证"));
    layout->addWidget(tabs_, 1);
}

QWidget* RuntimeDeliveryWorkspacePage::buildForm(RuntimeDeliveryMode mode)
{
    FormControls& form = controls(mode);
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 12, 0, 0);
    layout->setSpacing(16);

    auto* setup = new InfoPanel(mode == RuntimeDeliveryMode::DeploymentValidation
        ? tr("模型包部署验证") : tr("推理验证输入"));
    form.model = new QComboBox;
    form.model->setObjectName(mode == RuntimeDeliveryMode::DeploymentValidation
        ? QStringLiteral("DeploymentModelPackageCombo")
        : QStringLiteral("InferenceModelPackageCombo"));
    form.route = new QComboBox;
    form.route->setObjectName(mode == RuntimeDeliveryMode::DeploymentValidation
        ? QStringLiteral("DeploymentRuntimeRouteCombo")
        : QStringLiteral("InferenceRuntimeRouteCombo"));
    form.model->addItem(tr("请先打开项目并导入已验证模型包"), QString());
    form.route->addItem(tr("请先选择模型包"), QString());
    form.datasetId = new QLineEdit;
    form.datasetVersionId = new QLineEdit;
    form.snapshotId = new QLineEdit;
    form.snapshotArtifactId = new QLineEdit;
    form.relativePath = new QLineEdit;
    form.datasetId->setPlaceholderText(QStringLiteral("DatasetId"));
    form.datasetVersionId->setPlaceholderText(QStringLiteral("DatasetVersionId"));
    form.snapshotId->setPlaceholderText(QStringLiteral("SnapshotId"));
    form.snapshotArtifactId->setPlaceholderText(QStringLiteral("Snapshot ArtifactId"));
    form.relativePath->setPlaceholderText(
        tr("样本在 Snapshot Artifact 内的相对路径，例如 images/0001.png"));
    auto* fields = new QFormLayout;
    fields->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    fields->addRow(tr("已验证模型包"), form.model);
    fields->addRow(tr("Runtime 路线"), form.route);
    fields->addRow(tr("样本 DatasetId"), form.datasetId);
    fields->addRow(tr("样本 VersionId"), form.datasetVersionId);
    fields->addRow(tr("样本 SnapshotId"), form.snapshotId);
    fields->addRow(tr("样本 ArtifactId"), form.snapshotArtifactId);
    fields->addRow(tr("样本相对路径"), form.relativePath);
    setup->bodyLayout()->addLayout(fields);
    form.reasons = mutedLabel(tr(
        "选择模型包后会按 Manifest 路线顺序显示产品状态和本机可用性。"));
    form.reasons->setObjectName(QStringLiteral("RuntimeRouteReasons"));
    allowLabelToShrink(form.reasons);
    setup->bodyLayout()->addWidget(form.reasons);
    setup->bodyLayout()->addWidget(emptyStateLabel(tr(
        "只接受持久化 ModelPackageId 与已提交 Snapshot Artifact 内的相对路径；"
        "不会接受裸模型、engine、checkpoint 或样本文件路径，也不会自动切换 Runtime 路线。")));
    auto* runButton = primaryButton(tr("运行完整 Runtime Delivery"));
    connect(runButton, &QPushButton::clicked, this,
        [this, mode]() { emit runRequested(mode); });
    auto* actions = new QFrame;
    actions->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QHBoxLayout(actions);
    actionLayout->addStretch();
    actionLayout->addWidget(runButton);
    setup->bodyLayout()->addWidget(actions);

    auto* flow = new InfoPanel(tr("Runtime Delivery 六步链路"));
    auto* flowGrid = new QGridLayout;
    const QStringList titles = {
        tr("解析模型包"), tr("校验 Manifest"), tr("推理 Smoke"),
        tr("Benchmark"), tr("部署验证"), tr("交付报告")};
    const QStringList captions = {
        tr("只接受 ModelPackageId 与已提交 Artifact"),
        tr("校验路线、decoder、哈希与依赖"),
        tr("执行用户明确选择的 Runtime 路线"),
        tr("固定样本 smoke timing，非性能 SLA"),
        tr("提交预测、overlay 与验证报告"),
        tr("生成终态 Evidence 与 Model Card")};
    for (int index = 0; index < titles.size(); ++index) {
        flowGrid->addWidget(createInferenceStep(
            QString::number(index + 1), titles.at(index), captions.at(index)),
            index / 2, index % 2);
    }
    flow->bodyLayout()->addLayout(flowGrid);

    auto* result = new InfoPanel(tr("运行状态"));
    form.result = inlineStatusLabel(
        tr("尚未运行 Runtime Delivery 六步工作流。"));
    form.result->setObjectName(mode == RuntimeDeliveryMode::InferenceValidation
        ? QStringLiteral("InferenceResultSummary")
        : QStringLiteral("DeploymentResultSummary"));
    result->bodyLayout()->addWidget(form.result);
    if (mode == RuntimeDeliveryMode::InferenceValidation) {
        form.overlay = new QLabel(tr("暂无 overlay\n运行推理后显示可视化产物。"));
        form.overlay->setObjectName(QStringLiteral("InferenceOverlayCanvas"));
        form.overlay->setAlignment(Qt::AlignCenter);
        form.overlay->setMinimumHeight(260);
        result->bodyLayout()->addWidget(form.overlay);
    }
    connect(form.model,
        static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
        this, [this, mode](int) {
            emit modelSelectionChanged(mode, selectedModelPackageId(mode));
        });

    layout->addWidget(setup);
    layout->addWidget(flow);
    layout->addWidget(result, 1);
    scroll->setWidget(content);
    return scroll;
}

RuntimeDeliveryWorkspacePage::FormControls&
RuntimeDeliveryWorkspacePage::controls(RuntimeDeliveryMode mode)
{
    return mode == RuntimeDeliveryMode::DeploymentValidation
        ? deployment_ : inference_;
}

const RuntimeDeliveryWorkspacePage::FormControls&
RuntimeDeliveryWorkspacePage::controls(RuntimeDeliveryMode mode) const
{
    return mode == RuntimeDeliveryMode::DeploymentValidation
        ? deployment_ : inference_;
}

RuntimeDeliveryFormData RuntimeDeliveryWorkspacePage::formData(
    RuntimeDeliveryMode mode) const
{
    const FormControls& form = controls(mode);
    RuntimeDeliveryFormData result;
    result.modelPackageId = form.model->currentData().toString().trimmed();
    result.runtimeRoute = form.route->currentData().toString().trimmed();
    result.sampleDatasetId = form.datasetId->text().trimmed();
    result.sampleDatasetVersionId = form.datasetVersionId->text().trimmed();
    result.sampleSnapshotId = form.snapshotId->text().trimmed();
    result.sampleSnapshotArtifactId = form.snapshotArtifactId->text().trimmed();
    result.sampleRelativePath =
        QDir::fromNativeSeparators(form.relativePath->text().trimmed());
    return result;
}

QString RuntimeDeliveryWorkspacePage::selectedModelPackageId(
    RuntimeDeliveryMode mode) const
{
    return controls(mode).model->currentData().toString();
}

void RuntimeDeliveryWorkspacePage::setModelPackages(
    const QVector<ModelPackageListItem>& packages)
{
    for (RuntimeDeliveryMode mode : {
             RuntimeDeliveryMode::DeploymentValidation,
             RuntimeDeliveryMode::InferenceValidation}) {
        QComboBox* combo = controls(mode).model;
        const QString previous = combo->currentData().toString();
        const QSignalBlocker blocker(combo);
        combo->clear();
        combo->addItem(tr("请选择已验证模型包"), QString());
        for (const ModelPackageListItem& package : packages) {
            combo->addItem(QStringLiteral("%1 · %2 · %3")
                    .arg(package.modelFamily, package.taskType,
                        package.modelPackageId.left(8)),
                package.modelPackageId);
        }
        const int restored = combo->findData(previous);
        combo->setCurrentIndex(restored >= 0 ? restored : 0);
        if (restored < 0) {
            setRouteEvaluation(mode, {}, {});
        }
    }
}

void RuntimeDeliveryWorkspacePage::setRouteEvaluation(
    RuntimeDeliveryMode mode, const QStringList& availableRoutes,
    const QStringList& reasons)
{
    QComboBox* route = controls(mode).route;
    const QString previous = route->currentData().toString();
    const QSignalBlocker blocker(route);
    route->clear();
    if (availableRoutes.isEmpty()) {
        route->addItem(tr("无可执行路线"), QString());
    } else {
        route->addItem(tr("请选择 Runtime 路线"), QString());
        for (const QString& value : availableRoutes) {
            route->addItem(value, value);
        }
        if (availableRoutes.size() == 1) {
            route->setCurrentIndex(1);
        } else {
            const int restored = route->findData(previous);
            route->setCurrentIndex(restored >= 0 ? restored : 0);
        }
    }
    controls(mode).reasons->setText(reasons.isEmpty()
        ? tr("没有可显示的路线诊断。")
        : reasons.join(QStringLiteral("\n")));
}

void RuntimeDeliveryWorkspacePage::setRunning(RuntimeDeliveryMode mode)
{
    controls(mode).result->setText(tr(
        "Runtime Delivery 已派发：等待六步状态与最终 Evidence。"
        "底层同步 infer 返回前不能中途抢占。"));
    if (controls(mode).overlay) {
        setInferenceOverlayText(controls(mode).overlay, tr(
            "Runtime Delivery 运行中\n"
            "最终预测与 overlay 请在“任务与产物”中查看已提交 Artifact。"));
    }
}

void RuntimeDeliveryWorkspacePage::selectModelPackageForInference(
    const QString& modelPackageId)
{
    const int index = inference_.model->findData(modelPackageId);
    if (index >= 0) {
        inference_.model->setCurrentIndex(index);
    }
}

void RuntimeDeliveryWorkspacePage::showTab(int tabIndex)
{
    if (tabs_ && tabIndex >= 0 && tabIndex < tabs_->count()) {
        tabs_->setCurrentIndex(tabIndex);
    }
}
