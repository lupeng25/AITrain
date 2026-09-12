#include "WorkbenchTranslation.h"
#include "ModelRegistryPageController.h"

#include "ModelRegistryPage.h"
#include "ProjectObjectSelectors.h"
#include "TaskRuntimeController.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMessageBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QStackedWidget>
#include <QSignalBlocker>
#include "aitrain/model/ModelManifest.h"

ModelRegistryPageController::ModelRegistryPageController(
    const aitrain::ProjectQueryService* queryService,
    TaskRuntimeController* taskRuntime, QObject* parent)
    : QObject(parent)
    , queryService_(queryService)
    , presenter_(queryService, this)
    , taskRuntime_(taskRuntime)
{
    connect(&presenter_, &ModelRegistryPresenter::modelPackagesChanged,
        this, [this]() {
            render();
            emit packagesChanged();
        });
}

void ModelRegistryPageController::attach(ModelRegistryWorkspacePage* page)
{
    page_ = page;
    aitrain_app::bindCatalogSearch(page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch")), this, [this](const QString& text) {
        presenter_.setCatalogFilter({text, {}, {}}); presenter_.clear(); refresh();
    });
    connect(page_, &ModelRegistryWorkspacePage::refreshRequested,
        this, &ModelRegistryPageController::refresh);
    connect(page_, &ModelRegistryWorkspacePage::latestValidationRequested,
        this, &ModelRegistryPageController::openLatestValidation);
    connect(page_, &ModelRegistryWorkspacePage::moreRequested, this, [this]() { presenter_.loadMore(); });
    connect(page_, &ModelRegistryWorkspacePage::editManifestRequested, this, &ModelRegistryPageController::editManifest);
    connect(page_, &ModelRegistryWorkspacePage::browseSourceRequested,
        this, &ModelRegistryPageController::browseSource);
    connect(page_, &ModelRegistryWorkspacePage::browseManifestRequested,
        this, &ModelRegistryPageController::browseManifest);
    connect(page_, &ModelRegistryWorkspacePage::importRequested,
        this, &ModelRegistryPageController::importModel);
    connect(page_, &ModelRegistryWorkspacePage::useForRuntimeRequested,
        this, &ModelRegistryPageController::runtimeModelRequested);
    render();
}

void ModelRegistryPageController::setProjectContext(
    bool projectOpen, const QString& projectRoot)
{
    if (projectRoot_ != projectRoot) {
        presenter_.setCatalogFilter({});
        if (page_) { const QSignalBlocker blocker(page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch"))); page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch"))->clear(); }
        presenter_.clear(); editedManifest_ = {}; editedManifestPath_.clear(); importTaskId_.clear();
        if (page_) { page_->setSourceFilePath({}); page_->setManifestFilePath({}); page_->setMode(ModelRegistryWorkspacePage::Catalog); }
    }
    projectOpen_ = projectOpen;
    projectRoot_ = projectRoot;
    if (!projectOpen_) {
        presenter_.clear();
    }
}

void ModelRegistryPageController::setWorkerExecutable(const QString& executable)
{
    workerExecutable_ = executable;
}

void ModelRegistryPageController::refresh()
{
    if (!projectOpen_) {
        presenter_.clear();
        render();
        return;
    }
    const int previouslyLoaded = presenter_.modelPackageCount();
    const QString selected = page_ ? page_->selectedModelPackageId() : QString();
    if (!presenter_.refresh({50, {}})) return;
    while (presenter_.hasMore() && presenter_.modelPackageCount() < previouslyLoaded) {
        if (!presenter_.loadMore()) break;
    }
    if (page_ && !selected.isEmpty()) page_->selectPackage(selected);
}

void ModelRegistryPageController::render()
{
    if (!page_) {
        return;
    }
    QString status;
    if (!projectOpen_) {
        status = tr("请先打开项目。");
    } else if (!presenter_.lastError().isEmpty()) {
        status = tr("读取模型包目录失败：%1").arg(presenter_.lastError());
    } else {
        status = tr("已显示 %1 个模型。模型合同校验不代表客户场景精度验收。")
            .arg(presenter_.modelPackageCount());
    }
    page_->renderPackages(presenter_.modelPackages(), status);
    page_->moreButton->setVisible(presenter_.hasMore());
}

void ModelRegistryPageController::browseSource()
{
    const QString path = QFileDialog::getOpenFileName(
        page_, tr("选择待导入模型"), projectRoot_,
        tr("模型文件 (*.onnx);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        page_->setSourceFilePath(path);
    }
}

void ModelRegistryPageController::browseManifest()
{
    const QString path = QFileDialog::getOpenFileName(
        page_, tr("选择 Manifest 草稿"), projectRoot_,
        tr("JSON 文件 (*.json);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        page_->setManifestFilePath(path);
    }
}

void ModelRegistryPageController::importModel()
{
    if (taskRuntime_->isRunning()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("Worker 正在执行任务，稍后再导入模型。"));
        return;
    }
    const QString sourcePath = page_->sourceFilePath();
    const QString manifestPath = page_->manifestFilePath();
    if (!projectOpen_ || !QFileInfo(sourcePath).isFile()
        || !QFileInfo(manifestPath).isFile()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("请先打开项目，并选择常规模型文件和用户确认的 Manifest 草稿 JSON。"));
        return;
    }
    QFile manifestFile(manifestPath);
    if (manifestFile.size() > 1024 * 1024) {
        page_->setImportStatus(aitrain_app::workbenchText(QStringLiteral("模型说明超过 1 MB，请检查是否误选了模型文件。")));
        return;
    }
    if (!manifestFile.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(page_, tr("模型导入"),
            tr("无法读取 Manifest 草稿：%1").arg(manifestPath));
        return;
    }
    const QJsonDocument document = QJsonDocument::fromJson(manifestFile.readAll());
    if (!document.isObject()) {
        QMessageBox::warning(page_, tr("模型导入"),
            tr("Manifest 草稿必须是 JSON 对象。"));
        return;
    }
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    aitrain::worker_protocol::ModelImportCommand command;
    command.context.taskId = taskId;
    command.context.projectRoot = projectRoot_;
    command.sourceFilePath = sourcePath;
    command.manifestDraft = manifestPath == editedManifestPath_ && !editedManifest_.isEmpty()
        ? editedManifest_ : document.object();
    aitrain::ModelManifest checked;
    QString manifestError;
    if (!aitrain::decodeModelManifestImportDraft(command.manifestDraft, &checked, &manifestError)) {
        page_->setImportStatus(aitrain_app::workbenchText(QStringLiteral("模型说明未通过校验：%1")).arg(manifestError));
        return;
    }
    QString error;
    if (!taskRuntime_->start(workerExecutable_,
            aitrain::worker_protocol::TaskCommand{command}, &error)) {
        QMessageBox::critical(page_, tr("模型导入"), error);
        return;
    }
    importInProgress_ = true;
    importTaskId_ = taskId.toString();
    page_->setImportStatus(
        tr("正在导入模型并计算 SHA-256：%1")
            .arg(QDir::toNativeSeparators(sourcePath)));
    emit taskStarted(taskId.toString(), QStringLiteral("model_import"));
    emit importStarted();
}

void ModelRegistryPageController::finishImport(
    bool succeeded, const QString& message)
{
    if (!importInProgress_) {
        return;
    }
    importInProgress_ = false;
    if (page_) {
        page_->setImportStatus(succeeded
            ? tr("模型导入完成。")
            : tr("模型导入失败：%1").arg(message));
    }
    if (succeeded && page_) {
        const QSignalBlocker blocker(page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch")));
        page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch"))->clear(); presenter_.setCatalogFilter({});
    }
    refresh();
    if (succeeded && page_) for (const auto& item : presenter_.modelPackages()) {
        if (item.sourceTaskId == importTaskId_) { page_->selectPackage(item.modelPackageId, true); break; }
    }
}

const QVector<ModelPackageListItem>& ModelRegistryPageController::packages() const
{
    return presenter_.modelPackages();
}

void ModelRegistryPageController::editManifest()
{
    QFile file(page_->manifestFilePath());
    if (file.size() > 1024 * 1024 || !file.open(QIODevice::ReadOnly)) {
        page_->setImportStatus(aitrain_app::workbenchText(QStringLiteral("请先选择不超过 1 MB 的模型说明 JSON。"))); return;
    }
    const auto document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) { page_->setImportStatus(aitrain_app::workbenchText(QStringLiteral("模型说明必须是 JSON 对象。"))); return; }
    QJsonObject draft = page_->manifestFilePath() == editedManifestPath_ && !editedManifest_.isEmpty() ? editedManifest_ : document.object();
    QDialog dialog(page_); dialog.setWindowTitle(aitrain_app::workbenchText(QStringLiteral("检查与编辑模型说明"))); dialog.resize(900, 600);
    auto* layout = new QVBoxLayout(&dialog);
    auto* body = new QHBoxLayout; auto* navigation = new QListWidget; navigation->addItems({aitrain_app::workbenchText(QStringLiteral("模型语义")), aitrain_app::workbenchText(QStringLiteral("完整说明（专家）"))}); navigation->setFixedWidth(150);
    navigation->setWordWrap(true);
    navigation->setResizeMode(QListView::Adjust);
    navigation->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* stack = new QStackedWidget; body->addWidget(navigation); body->addWidget(stack, 1); layout->addLayout(body, 1);
    auto* basic = new QWidget; auto* fields = new QFormLayout(basic);
    QMap<QString, QLineEdit*> edits;
    const QStringList keys = {QStringLiteral("modelFamily"), QStringLiteral("taskType"), QStringLiteral("sourceBackend"), QStringLiteral("artifactFormat"), QStringLiteral("decoder"), QStringLiteral("exporterVersion")};
    const QStringList captions = {aitrain_app::workbenchText(QStringLiteral("模型族")), aitrain_app::workbenchText(QStringLiteral("任务类型")), aitrain_app::workbenchText(QStringLiteral("来源后端")), aitrain_app::workbenchText(QStringLiteral("产物格式")), aitrain_app::workbenchText(QStringLiteral("解码器")), aitrain_app::workbenchText(QStringLiteral("导出器版本"))};
    for (int i = 0; i < keys.size(); ++i) { auto* edit = new QLineEdit(draft.value(keys[i]).toString()); edits.insert(keys[i], edit); fields->addRow(captions[i], edit); }
    fields->addRow(aitrain_app::workbenchHint(aitrain_app::workbenchText(QStringLiteral("来源快照、输入输出张量、预处理和类别必须与模型一致。完整说明保留所有字段；校验不会自动声明精度或补造模型语义。"))));
    stack->addWidget(basic);
    auto* raw = new QPlainTextEdit; raw->setPlainText(QString::fromUtf8(QJsonDocument(draft).toJson(QJsonDocument::Indented))); stack->addWidget(raw);
    connect(navigation, &QListWidget::currentRowChanged, &dialog, [&, stack](int index) {
        if (index == 1) { for (auto it = edits.cbegin(); it != edits.cend(); ++it) draft.insert(it.key(), it.value()->text()); raw->setPlainText(QString::fromUtf8(QJsonDocument(draft).toJson(QJsonDocument::Indented))); }
        else if (stack->currentIndex() == 1) {
            const auto parsed = QJsonDocument::fromJson(raw->toPlainText().toUtf8());
            if (parsed.isObject()) { draft = parsed.object(); for (auto it = edits.cbegin(); it != edits.cend(); ++it) it.value()->setText(draft.value(it.key()).toString()); }
        }
        stack->setCurrentIndex(index);
    });
    auto* error = aitrain_app::workbenchHint(); layout->addWidget(error);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Apply); layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked, &dialog, [&]() {
        if (stack->currentIndex() == 1) {
            const auto parsed = QJsonDocument::fromJson(raw->toPlainText().toUtf8());
            if (!parsed.isObject()) { error->setText(aitrain_app::workbenchText(QStringLiteral("完整说明不是有效 JSON 对象。"))); return; } draft = parsed.object();
        } else for (auto it = edits.cbegin(); it != edits.cend(); ++it) draft.insert(it.key(), it.value()->text());
        aitrain::ModelManifest checked; QString message;
        if (!aitrain::decodeModelManifestImportDraft(draft, &checked, &message)) { error->setText(message); return; }
        editedManifest_ = draft; editedManifestPath_ = page_->manifestFilePath(); dialog.accept();
    });
    navigation->setCurrentRow(0);
    if (dialog.exec() == QDialog::Accepted) page_->setImportStatus(aitrain_app::workbenchText(QStringLiteral("模型说明已校验，修改保存在本次导入草稿中。点击导入后由 Worker 执行。")));
}

void ModelRegistryPageController::selectPackage(const QString& id)
{
    if (!page_) return;
    auto* search = page_->findChild<QLineEdit*>(QStringLiteral("ModelCatalogSearch"));
    if (!search->text().isEmpty()) { const QSignalBlocker blocker(search); search->clear(); presenter_.setCatalogFilter({}); presenter_.clear(); refresh(); }
    bool found = false;
    do {
        for (const auto& item : presenter_.modelPackages()) if (item.modelPackageId == id) { found = true; break; }
        if (found || !presenter_.hasMore()) break;
    } while (presenter_.loadMore());
    if (found) page_->selectPackage(id, true);
}

void ModelRegistryPageController::openLatestValidation(const QString& taskText)
{
    aitrain::TaskId taskId; QString error, cursor;
    if (!page_ || !queryService_ || !projectOpen_ || !aitrain::TaskId::parse(taskText, &taskId, &error)) return;
    do {
        const auto artifacts = queryService_->taskArtifacts(taskId, {100, cursor}, &error);
        if (!error.isEmpty()) break;
        for (const auto& artifact : artifacts.items) if (artifact.kind == QStringLiteral("runtime_delivery_report")) {
            aitrain_app::showArtifactReport(page_, queryService_, artifact.id.toString(),
                QStringLiteral("delivery_report.json"), aitrain_app::workbenchText(QStringLiteral("模型最近验证报告")));
            return;
        }
        cursor = artifacts.hasMore ? artifacts.nextCursor : QString();
    } while (!cursor.isEmpty());
    // 失败或尚未生成报告时定位原任务，保留真实失败原因。
    emit page_->sourceTaskRequested(taskText);
}
