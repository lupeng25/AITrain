#include "ProjectPage.h"

#include "InfoPanel.h"

#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QVBoxLayout>

using namespace aitrain_app;

ProjectWorkspacePage::ProjectWorkspacePage(
    const QString& defaultRoot, QWidget* parent)
    : QScrollArea(parent)
{
    setWidgetResizable(true);
    setFrameShape(QFrame::NoFrame);

    auto* content = new QWidget;
    auto* layout = new QVBoxLayout(content);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    createButton_ = primaryButton(tr("创建项目"));
    openButton_ = new QPushButton(tr("打开项目"));
    rebuildButton_ = dangerButton(tr("重建项目"));
    createButton_->setObjectName(QStringLiteral("ProjectCreateButton"));
    openButton_->setObjectName(QStringLiteral("ProjectOpenButton"));
    rebuildButton_->setObjectName(QStringLiteral("ProjectRebuildButton"));
    for (QPushButton* button : {createButton_, openButton_, rebuildButton_}) {
        button->setProperty("projectSessionAction", true);
    }

    auto* headerPanel = new QFrame;
    headerPanel->setObjectName(QStringLiteral("WorkspaceToolbar"));
    auto* headerLayout = new QHBoxLayout(headerPanel);
    headerLayout->setContentsMargins(14, 10, 14, 10);
    headerLayout->setSpacing(12);
    auto* contextLabel = new QLabel(tr("本地项目与元数据"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    statusLabel_ = inlineStatusLabel(tr("未打开项目。"));
    statusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    auto* policyStatus = inlineStatusLabel(
        tr("工作区由 .aitrain 管理，产物和元数据通过登记身份访问。"));
    policyStatus->setObjectName(QStringLiteral("WorkspaceToolbarMeta"));
    allowLabelToShrink(statusLabel_);
    allowLabelToShrink(policyStatus);
    headerLayout->addWidget(contextLabel);
    headerLayout->addWidget(statusLabel_);
    headerLayout->addWidget(policyStatus, 1);
    headerLayout->addWidget(createButton_);
    headerLayout->addWidget(openButton_);
    headerLayout->addWidget(rebuildButton_);

    auto* formPanel = new InfoPanel(tr("项目设置"));
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    form->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    form->setFormAlignment(Qt::AlignTop);
    form->setHorizontalSpacing(14);
    form->setVerticalSpacing(10);
    projectNameEdit_ = new QLineEdit(tr("本地训练项目"));
    projectNameEdit_->setObjectName(QStringLiteral("ProjectNameEdit"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(defaultRoot));
    projectRootEdit_->setObjectName(QStringLiteral("ProjectRootEdit"));
    browseButton_ = new QPushButton(tr("选择目录"));

    auto* pathRow = new QWidget;
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(projectRootEdit_);
    pathLayout->addWidget(browseButton_);
    form->addRow(tr("项目名称"), projectNameEdit_);
    form->addRow(tr("项目目录"), pathRow);
    formPanel->bodyLayout()->addLayout(form);
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    actionLayout->setContentsMargins(10, 8, 10, 8);
    auto* actionHint = mutedLabel(
        tr("打开项目后，项目摘要只读取 .aitrain/project.sqlite 中已持久化的事实。"));
    allowLabelToShrink(actionHint);
    actionLayout->addWidget(actionHint, 0, 0);
    formPanel->bodyLayout()->addWidget(actionStrip);
    formPanel->bodyLayout()->addStretch();

    auto* summaryPanel = new InfoPanel(tr("项目摘要"));
    auto* summaryGrid = new QGridLayout;
    summaryGrid->setHorizontalSpacing(10);
    summaryGrid->setVerticalSpacing(10);
    auto* pathCard = createCompactSummaryCard(
        tr("当前项目"), tr("未打开"), tr("项目登记身份"));
    pathSummaryLabel_ = pathCard->findChild<QLabel*>(
        QStringLiteral("CompactMetricValue"));
    auto* sqliteCard = createCompactSummaryCard(
        QStringLiteral("SQLite"), tr("未连接"), tr("项目元数据状态"));
    sqliteSummaryLabel_ = sqliteCard->findChild<QLabel*>(
        QStringLiteral("CompactMetricValue"));
    auto* datasetCard = createCompactSummaryCard(
        tr("数据集"), QStringLiteral("0"), tr("已登记数据集"));
    datasetSummaryLabel_ = datasetCard->findChild<QLabel*>(
        QStringLiteral("CompactMetricValue"));
    datasetSummaryLabel_->setObjectName(QStringLiteral("ProjectDatasetSummary"));
    auto* taskCard = createCompactSummaryCard(
        tr("任务"), QStringLiteral("0"), tr("训练、校验、导出、推理"));
    taskSummaryLabel_ = taskCard->findChild<QLabel*>(
        QStringLiteral("CompactMetricValue"));
    taskSummaryLabel_->setObjectName(QStringLiteral("ProjectTaskSummary"));
    auto* modelCard = createCompactSummaryCard(
        tr("模型包"), QStringLiteral("0"), tr("已登记模型包"));
    modelSummaryLabel_ = modelCard->findChild<QLabel*>(
        QStringLiteral("CompactMetricValue"));
    modelSummaryLabel_->setObjectName(
        QStringLiteral("ProjectModelPackageSummary"));
    summaryGrid->addWidget(pathCard, 0, 0, 1, 2);
    summaryGrid->addWidget(sqliteCard, 0, 2);
    summaryGrid->addWidget(datasetCard, 1, 0);
    summaryGrid->addWidget(taskCard, 1, 1);
    summaryGrid->addWidget(modelCard, 1, 2);
    for (int column = 0; column < 3; ++column) {
        summaryGrid->setColumnStretch(column, 1);
    }
    summaryPanel->bodyLayout()->addLayout(summaryGrid);

    auto* structurePanel = new InfoPanel(tr("标准目录结构"));
    auto* structure = new QPlainTextEdit;
    structure->setReadOnly(true);
    structure->setMaximumHeight(170);
    structure->setPlainText(QStringLiteral(
        ".aitrain/\n  artifacts/\n    committed/\n    .staging/\n  project.sqlite"));
    structurePanel->bodyLayout()->addWidget(structure);
    structurePanel->bodyLayout()->addWidget(mutedLabel(
        tr("项目页只负责项目会话；训练、导出和推理仍通过 Worker 执行，GUI 不打开或复制物理产物路径。")));
    summaryPanel->bodyLayout()->addWidget(structurePanel);

    layout->addWidget(headerPanel);
    layout->addWidget(formPanel);
    layout->addWidget(summaryPanel);
    layout->addStretch();
    setWidget(content);

    connect(browseButton_, &QPushButton::clicked, this, [this]() {
        const QString directory = QFileDialog::getExistingDirectory(
            this, tr("选择项目目录"));
        if (!directory.isEmpty()) setProjectRoot(directory);
    });
    connect(createButton_, &QPushButton::clicked, this, [this]() {
        emit operationRequested(ProjectSessionOperation::Create);
    });
    connect(openButton_, &QPushButton::clicked, this, [this]() {
        emit operationRequested(ProjectSessionOperation::Open);
    });
    connect(rebuildButton_, &QPushButton::clicked,
        this, &ProjectWorkspacePage::requestRebuild);
}

QString ProjectWorkspacePage::projectName() const
{
    return projectNameEdit_->text().trimmed();
}

QString ProjectWorkspacePage::projectRoot() const
{
    return QDir::fromNativeSeparators(projectRootEdit_->text().trimmed());
}

void ProjectWorkspacePage::setProjectRoot(const QString& root)
{
    projectRootEdit_->setText(QDir::toNativeSeparators(root));
}

void ProjectWorkspacePage::setBusy(bool busy)
{
    for (QPushButton* button : {createButton_, openButton_, rebuildButton_,
             browseButton_}) {
        button->setEnabled(!busy);
    }
    projectNameEdit_->setEnabled(!busy);
    projectRootEdit_->setEnabled(!busy);
}

void ProjectWorkspacePage::setStatus(const QString& status)
{
    statusLabel_->setText(status);
}

void ProjectWorkspacePage::render(const ProjectPageViewModel& viewModel)
{
    const bool hasProject =
        viewModel.workspaceOpen && viewModel.summary.available;
    if (!viewModel.queryError.isEmpty()) {
        setStatus(tr("项目汇总读取失败：%1").arg(viewModel.queryError));
    } else {
        setStatus(hasProject
            ? tr("已打开：%1").arg(viewModel.projectName)
            : tr("未打开项目。"));
    }
    pathSummaryLabel_->setText(hasProject
        ? (viewModel.projectName.isEmpty() ? tr("已打开")
                                           : viewModel.projectName)
        : tr("未打开"));
    sqliteSummaryLabel_->setText(hasProject ? tr("已连接") : tr("未连接"));

    const ProjectSummaryViewModel& summary = viewModel.summary;
    datasetSummaryLabel_->setText(QString::number(summary.datasetCount));
    datasetSummaryLabel_->setToolTip(tr("版本 %1，快照 %2")
        .arg(summary.datasetVersionCount)
        .arg(summary.datasetSnapshotCount));
    taskSummaryLabel_->setText(QString::number(summary.taskCount));
    taskSummaryLabel_->setToolTip(tr("活动 %1，成功 %2，失败 %3，取消 %4")
        .arg(summary.activeTaskCount)
        .arg(summary.succeededTaskCount)
        .arg(summary.failedTaskCount)
        .arg(summary.canceledTaskCount));
    modelSummaryLabel_->setText(QString::number(summary.modelPackageCount));
    modelSummaryLabel_->setToolTip(tr("已校验模型包 %1；已提交产物 %2")
        .arg(summary.verifiedModelPackageCount)
        .arg(summary.committedArtifactCount));
}

void ProjectWorkspacePage::showOperationError(const QString& message)
{
    setStatus(tr("项目操作失败：%1").arg(message));
    QMessageBox::critical(this, tr("项目"),
        tr("无法完成项目操作：%1").arg(message));
}

void ProjectWorkspacePage::requestRebuild()
{
    const QMessageBox::StandardButton answer = QMessageBox::warning(this,
        tr("重建项目"),
        tr("重建会永久清除该项目 .aitrain 中的全部项目元数据和已提交 Artifact。"
           "不会自动备份，也不会删除项目根目录中的其他文件。是否继续？"),
        QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
    if (answer == QMessageBox::Yes) {
        emit operationRequested(ProjectSessionOperation::Rebuild);
    }
}
