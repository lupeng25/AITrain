#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

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
#include <QSizePolicy>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildModelRegistryPage()
{
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);

    auto* headerRefreshButton = primaryButton(QStringLiteral("刷新模型库"));
    connect(headerRefreshButton, &QPushButton::clicked, this, &MainWindow::refreshModelRegistry);

    auto* toolbar = new InfoPanel(QStringLiteral("模型库"));
    auto* actionStrip = new QFrame;
    actionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* actionGrid = new QGridLayout(actionStrip);
    actionGrid->setContentsMargins(10, 8, 10, 8);
    actionGrid->setHorizontalSpacing(10);
    actionGrid->setVerticalSpacing(8);
    auto* inferButton = new QPushButton(QStringLiteral("选中模型包用于推理"));
    connect(inferButton, &QPushButton::clicked, this, [this]() {
        if (!ModelPackageTable_ || ModelPackageTable_->selectedItems().isEmpty()) {
            QMessageBox::information(this, uiText("模型库"), uiText("请先选择一个已验证模型包。"));
            return;
        }
        const int row = ModelPackageTable_->selectedItems().first()->row();
        const QString modelPackageId = ModelPackageTable_->item(row, 0)
            ? ModelPackageTable_->item(row, 0)->data(Qt::UserRole).toString()
            : QString();
        if (modelPackageId.isEmpty() || !inferenceModelPackageCombo_) {
            QMessageBox::information(this, uiText("模型库"), uiText("选中行不包含可用的模型包 ID。"));
            return;
        }
        const int comboIndex = inferenceModelPackageCombo_->findData(modelPackageId);
        if (comboIndex < 0) {
            QMessageBox::warning(this, uiText("模型库"), uiText("模型包目录已变更，请刷新模型库后重试。"));
            return;
        }
        inferenceModelPackageCombo_->setCurrentIndex(comboIndex);
        showDeploymentTab(1);
    });
    actionGrid->addWidget(inferButton, 0, 0);
    for (int column = 0; column < 1; ++column) {
        actionGrid->setColumnStretch(column, 1);
    }
    modelRegistrySummaryLabel_ = mutedLabel(uiText("推理与部署验证只使用已登记且经 Manifest 校验的模型包；任务评估证据请从任务与产物页查看。"));
    allowLabelToShrink(modelRegistrySummaryLabel_);
    toolbar->bodyLayout()->addWidget(actionStrip);
    toolbar->bodyLayout()->addWidget(modelRegistrySummaryLabel_);

    auto* ModelPackagePanel = new InfoPanel(QStringLiteral("已验证模型包"));
    auto* importForm = new QFormLayout;
    importForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    importForm->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    modelImportSourceEdit_ = new QLineEdit;
    modelImportSourceEdit_->setPlaceholderText(QStringLiteral("选择待导入的常规模型文件（例如 .onnx）"));
    modelImportManifestEdit_ = new QLineEdit;
    modelImportManifestEdit_->setPlaceholderText(QStringLiteral("选择用户确认的 Manifest 草稿 JSON"));
    const auto makeImportPathRow = [this](QLineEdit* edit, const QString& title, const QString& filter) {
        auto* row = new QWidget;
        auto* rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        auto* browseButton = new QPushButton(uiText("选择文件"));
        connect(browseButton, &QPushButton::clicked, this, [this, edit, title, filter]() {
            const QString file = QFileDialog::getOpenFileName(this, title, currentProjectPath_, filter);
            if (!file.isEmpty()) edit->setText(QDir::toNativeSeparators(file));
        });
        rowLayout->addWidget(edit, 1);
        rowLayout->addWidget(browseButton);
        return row;
    };
    importForm->addRow(QStringLiteral("模型文件"), makeImportPathRow(modelImportSourceEdit_, uiText("选择待导入模型"), QStringLiteral("Model files (*.onnx);;All files (*.*)")));
    importForm->addRow(QStringLiteral("Manifest 草稿"), makeImportPathRow(modelImportManifestEdit_, uiText("选择 Manifest 草稿"), QStringLiteral("JSON files (*.json);;All files (*.*)")));
    ModelPackagePanel->bodyLayout()->addLayout(importForm);
    auto* importActionStrip = new QFrame;
    importActionStrip->setObjectName(QStringLiteral("ActionStrip"));
    auto* importActionLayout = new QHBoxLayout(importActionStrip);
    importActionLayout->setContentsMargins(10, 8, 10, 8);
    modelImportResultLabel_ = mutedLabel(uiText("导入由 Worker 执行；Manifest 草稿必须明确模型语义、张量契约、来源快照和已验证状态。导入过程将生成任务 ID 与模型 SHA-256。"));
    allowLabelToShrink(modelImportResultLabel_);
    auto* importButton = primaryButton(uiText("导入模型包"));
    connect(importButton, &QPushButton::clicked, this, &MainWindow::importModelPackage);
    importActionLayout->addWidget(modelImportResultLabel_, 1);
    importActionLayout->addWidget(importButton);
    ModelPackagePanel->bodyLayout()->addWidget(importActionStrip);
    ModelPackageTable_ = new QTableWidget(0, 6);
    ModelPackageTable_->setHorizontalHeaderLabels(QStringList()
        << QStringLiteral("模型包 ID")
        << QStringLiteral("模型族")
        << QStringLiteral("任务")
        << QStringLiteral("来源后端")
        << QStringLiteral("解码器")
        << QStringLiteral("登记时间"));
    configureTable(ModelPackageTable_);
    ModelPackageTable_->setWordWrap(true);
    ModelPackageTable_->verticalHeader()->setDefaultSectionSize(42);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    ModelPackageTable_->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    ModelPackagePanel->bodyLayout()->addWidget(ModelPackageTable_);

    modelWorkspaceTabs_ = new QTabWidget;
    modelWorkspaceTabs_->setObjectName(QStringLiteral("ModelWorkspaceTabs"));
    modelWorkspaceTabs_->addTab(ModelPackagePanel, uiText(" 模型包"));

    layout->addWidget(createWorkbenchHeader(
        QStringLiteral("MODEL REGISTRY"),
        uiText("模型库工作台"),
        uiText("管理已验证模型包，并将任务评估证据交由任务与产物页统一查看。"),
        headerRefreshButton,
        QStringList() << uiText("已验证模型包")));
    layout->addWidget(toolbar);
    layout->addWidget(modelWorkspaceTabs_, 1);
    return page;
}
