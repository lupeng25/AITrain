#include "ModelRegistryPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QDir>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QVBoxLayout>

using namespace aitrain_app;

ModelRegistryWorkspacePage::ModelRegistryWorkspacePage(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);
    auto* refreshButton = primaryButton(tr("刷新模型库"));
    connect(refreshButton, &QPushButton::clicked,
        this, &ModelRegistryWorkspacePage::refreshRequested);

    auto* toolbar = new InfoPanel(tr("模型库"));
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionLayout = new QGridLayout(actionStrip);
    auto* runtimeButton = new QPushButton(tr("选中模型包用于推理"));
    connect(runtimeButton, &QPushButton::clicked, this, [this]() {
        const QString id = selectedModelPackageId();
        if (id.isEmpty()) {
            QMessageBox::information(this, tr("模型库"),
                tr("请先选择一个已验证模型包。"));
            return;
        }
        emit useForRuntimeRequested(id);
    });
    actionLayout->addWidget(runtimeButton, 0, 0);
    summaryLabel_ = mutedLabel(
        tr("推理与部署验证只使用已登记且经 Manifest 校验的模型包；任务评估证据请从任务与产物页查看。"));
    allowLabelToShrink(summaryLabel_);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addWidget(summaryLabel_);

    auto* packagePanel = new InfoPanel(tr("已验证模型包"));
    auto* importForm = new QFormLayout;
    importForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    sourceEdit_ = new QLineEdit;
    sourceEdit_->setPlaceholderText(tr("选择待导入的常规模型文件（例如 .onnx）"));
    manifestEdit_ = new QLineEdit;
    manifestEdit_->setPlaceholderText(tr("选择用户确认的 Manifest 草稿 JSON"));
    const auto pathRow = [this](QLineEdit* edit, bool source) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        auto* browseButton = new QPushButton(tr("选择文件"));
        connect(browseButton, &QPushButton::clicked, this,
            source ? &ModelRegistryWorkspacePage::browseSourceRequested
                   : &ModelRegistryWorkspacePage::browseManifestRequested);
        rowLayout->addWidget(edit, 1);
        rowLayout->addWidget(browseButton);
        return row;
    };
    importForm->addRow(tr("模型文件"), pathRow(sourceEdit_, true));
    importForm->addRow(tr("Manifest 草稿"), pathRow(manifestEdit_, false));
    packagePanel->bodyLayout()->addLayout(importForm);

    auto* importStrip = new QFrame;
    importStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* importLayout = new QHBoxLayout(importStrip);
    importStatusLabel_ = mutedLabel(tr(
        "导入由 Worker 执行；Manifest 草稿必须明确模型语义、张量契约、来源快照和已验证状态。导入过程将生成任务 ID 与模型 SHA-256。"));
    allowLabelToShrink(importStatusLabel_);
    auto* importButton = primaryButton(tr("导入模型包"));
    connect(importButton, &QPushButton::clicked,
        this, &ModelRegistryWorkspacePage::importRequested);
    importLayout->addWidget(importStatusLabel_, 1);
    importLayout->addWidget(importButton);
    packagePanel->bodyLayout()->addWidget(importStrip);

    packageTable_ = new QTableWidget(0, 6);
    packageTable_->setObjectName(QStringLiteral("ModelPackageTable"));
    packageTable_->setHorizontalHeaderLabels(QStringList()
        << tr("模型包 ID") << tr("模型族") << tr("任务") << tr("来源后端")
        << tr("解码器") << tr("登记时间"));
    packageTable_->setAlternatingRowColors(true);
    packageTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    packageTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    packageTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    packageTable_->verticalHeader()->setVisible(false);
    packageTable_->verticalHeader()->setDefaultSectionSize(42);
    packageTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    for (int column = 1; column < 6; ++column) {
        packageTable_->horizontalHeader()->setSectionResizeMode(
            column, QHeaderView::ResizeToContents);
    }
    packagePanel->bodyLayout()->addWidget(packageTable_);

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("MODEL REGISTRY"), tr("模型库工作台"),
        tr("管理已验证模型包，并将任务评估证据交由任务与产物页统一查看。"),
        refreshButton, QStringList() << tr("已验证模型包")));
    layout->addWidget(toolbar);
    auto* tabs = new QTabWidget;
    tabs->setObjectName(QStringLiteral("ModelWorkspaceTabs"));
    tabs->addTab(packagePanel, tr(" 模型包"));
    layout->addWidget(tabs, 1);
}

QString ModelRegistryWorkspacePage::sourceFilePath() const
{
    return QDir::fromNativeSeparators(sourceEdit_->text().trimmed());
}

QString ModelRegistryWorkspacePage::manifestFilePath() const
{
    return QDir::fromNativeSeparators(manifestEdit_->text().trimmed());
}

QString ModelRegistryWorkspacePage::selectedModelPackageId() const
{
    const auto selected = packageTable_->selectedItems();
    if (selected.isEmpty()) {
        return {};
    }
    const int row = selected.first()->row();
    const auto* item = packageTable_->item(row, 0);
    return item ? item->data(Qt::UserRole).toString() : QString();
}

void ModelRegistryWorkspacePage::setSourceFilePath(const QString& path)
{
    sourceEdit_->setText(QDir::toNativeSeparators(path));
}

void ModelRegistryWorkspacePage::setManifestFilePath(const QString& path)
{
    manifestEdit_->setText(QDir::toNativeSeparators(path));
}

void ModelRegistryWorkspacePage::setImportStatus(const QString& status)
{
    importStatusLabel_->setText(status);
}

void ModelRegistryWorkspacePage::renderPackages(
    const QVector<ModelPackageListItem>& packages, const QString& status)
{
    summaryLabel_->setText(status);
    packageTable_->setRowCount(0);
    if (packages.isEmpty()) {
        packageTable_->insertRow(0);
        packageTable_->setItem(0, 0, new QTableWidgetItem(tr("暂无已验证模型包")));
        return;
    }
    for (const ModelPackageListItem& package : packages) {
        const int row = packageTable_->rowCount();
        packageTable_->insertRow(row);
        auto* idItem = new QTableWidgetItem(package.modelPackageId);
        idItem->setData(Qt::UserRole, package.modelPackageId);
        packageTable_->setItem(row, 0, idItem);
        packageTable_->setItem(row, 1, new QTableWidgetItem(package.modelFamily));
        packageTable_->setItem(row, 2, new QTableWidgetItem(package.taskType));
        packageTable_->setItem(row, 3, new QTableWidgetItem(package.sourceBackend));
        packageTable_->setItem(row, 4, new QTableWidgetItem(package.decoder));
        packageTable_->setItem(row, 5, new QTableWidgetItem(package.createdAt));
    }
}
