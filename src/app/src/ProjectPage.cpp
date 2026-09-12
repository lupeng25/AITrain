#include "WorkbenchTranslation.h"
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

ProjectWorkspacePage::ProjectWorkspacePage(const QString& defaultRoot, QWidget* parent)
    : WorkspaceViewHost(parent)
{
    auto* home = addMode(aitrain_app::workbenchText(QStringLiteral("最近项目")));
    statusLabel_ = workbenchHint(aitrain_app::workbenchText(QStringLiteral("打开或创建项目开始工作。"))); statusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus")); static_cast<QVBoxLayout*>(layout())->addWidget(statusLabel_);
    recentTable_ = workbenchTable({aitrain_app::workbenchText(QStringLiteral("项目")), aitrain_app::workbenchText(QStringLiteral("位置"))}); recentTable_->setObjectName(QStringLiteral("RecentProjectsTable")); home->addWidget(recentTable_, 1);
    auto* actions = new QHBoxLayout; auto* recent = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开选中项目")));
    auto* choose = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开其他项目"))); auto* create = workbenchButton(aitrain_app::workbenchText(QStringLiteral("新建项目")), {}, true);
    actions->addWidget(recent); actions->addWidget(choose); actions->addStretch(); actions->addWidget(create); home->addLayout(actions);
    const auto openRecent = [this]() { const auto* item = recentTable_->item(recentTable_->currentRow(), 0); if (!item) return; setProjectRoot(item->data(Qt::UserRole).toString()); emit operationRequested(ProjectSessionOperation::Open); };
    connect(recent, &QPushButton::clicked, this, openRecent); connect(recentTable_, &QTableWidget::cellDoubleClicked, this, [openRecent](int, int) { openRecent(); });
    connect(create, &QPushButton::clicked, this, [this]() { setMode(1); });
    connect(choose, &QPushButton::clicked, this, [this]() { const QString directory = QFileDialog::getExistingDirectory(this, aitrain_app::workbenchText(QStringLiteral("打开项目"))); if (!directory.isEmpty()) { setProjectRoot(directory); emit operationRequested(ProjectSessionOperation::Open); } });
    auto* manageButton = workbenchButton(aitrain_app::workbenchText(QStringLiteral("项目管理"))); toolbar->addWidget(manageButton); connect(manageButton, &QPushButton::clicked, this, [this]() { setMode(1); });
    auto* manage = addMode(aitrain_app::workbenchText(QStringLiteral("新建与管理项目")));
    projectNameEdit_ = new QLineEdit(aitrain_app::workbenchText(QStringLiteral("本地训练项目"))); projectNameEdit_->setObjectName(QStringLiteral("ProjectNameEdit"));
    projectRootEdit_ = new QLineEdit(QDir::toNativeSeparators(defaultRoot)); projectRootEdit_->setObjectName(QStringLiteral("ProjectRootEdit"));
    auto* fields = new QFormLayout; fields->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow); fields->addRow(aitrain_app::workbenchText(QStringLiteral("项目名称")), projectNameEdit_);
    auto* row = new QWidget; auto* rowLayout = new QHBoxLayout(row); rowLayout->setContentsMargins(0, 0, 0, 0); rowLayout->addWidget(projectRootEdit_, 1); browseButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("选择目录"))); rowLayout->addWidget(browseButton_); fields->addRow(aitrain_app::workbenchText(QStringLiteral("项目位置")), row); manage->addLayout(fields);
    auto* summary = new QFormLayout;
    pathSummaryLabel_ = workbenchHint(); sqliteSummaryLabel_ = workbenchHint(); datasetSummaryLabel_ = workbenchHint(); taskSummaryLabel_ = workbenchHint(); modelSummaryLabel_ = workbenchHint();
    datasetSummaryLabel_->setObjectName(QStringLiteral("ProjectDatasetSummary")); taskSummaryLabel_->setObjectName(QStringLiteral("ProjectTaskSummary")); modelSummaryLabel_->setObjectName(QStringLiteral("ProjectModelPackageSummary"));
    summary->addRow(aitrain_app::workbenchText(QStringLiteral("当前项目")), pathSummaryLabel_); summary->addRow(aitrain_app::workbenchText(QStringLiteral("元数据")), sqliteSummaryLabel_); summary->addRow(aitrain_app::workbenchText(QStringLiteral("数据集")), datasetSummaryLabel_); summary->addRow(aitrain_app::workbenchText(QStringLiteral("任务")), taskSummaryLabel_); summary->addRow(aitrain_app::workbenchText(QStringLiteral("模型")), modelSummaryLabel_); manage->addLayout(summary); manage->addStretch();
    auto* manageActions = new QHBoxLayout; createButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("创建项目")), QStringLiteral("ProjectCreateButton"), true); openButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("打开此目录")), QStringLiteral("ProjectOpenButton")); rebuildButton_ = workbenchButton(aitrain_app::workbenchText(QStringLiteral("重建项目…")), QStringLiteral("ProjectRebuildButton"));
    manageActions->addWidget(rebuildButton_); manageActions->addStretch(); manageActions->addWidget(openButton_); manageActions->addWidget(createButton_); manage->addLayout(manageActions);
    for (auto* button : {recent, choose, create, manageButton, createButton_, openButton_, rebuildButton_}) button->setProperty("projectSessionAction", true);
    connect(browseButton_, &QPushButton::clicked, this, [this]() { const QString directory = QFileDialog::getExistingDirectory(this, aitrain_app::workbenchText(QStringLiteral("选择项目目录"))); if (!directory.isEmpty()) setProjectRoot(directory); });
    connect(createButton_, &QPushButton::clicked, this, [this]() { emit operationRequested(ProjectSessionOperation::Create); });
    connect(openButton_, &QPushButton::clicked, this, [this]() { emit operationRequested(ProjectSessionOperation::Open); });
    connect(rebuildButton_, &QPushButton::clicked, this, &ProjectWorkspacePage::requestRebuild);
    setMode(0);
}

void ProjectWorkspacePage::setRecentProjects(const QVariantList& projects)
{
    recentTable_->setRowCount(0);
    for (const auto& value : projects) {
        const auto item = value.toMap(); const int row = recentTable_->rowCount(); recentTable_->insertRow(row);
        auto* name = new QTableWidgetItem(item.value(QStringLiteral("name")).toString()); name->setData(Qt::UserRole, item.value(QStringLiteral("root"))); recentTable_->setItem(row, 0, name);
        auto* path = new QTableWidgetItem(QDir::toNativeSeparators(item.value(QStringLiteral("root")).toString())); path->setToolTip(path->text()); recentTable_->setItem(row, 1, path);
    }
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
    for (auto* button : findChildren<QPushButton*>()) if (button->property("projectSessionAction").toBool()) button->setEnabled(!busy);
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
