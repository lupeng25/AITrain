#include "WorkbenchTranslation.h"
#include "ProjectObjectSelectors.h"

#include "MainWindowSupport.h"
#include "WorkbenchWidgets.h"

#include <QDialog>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QJsonDocument>
#include <QPointer>
#include <QSignalBlocker>
#include <QSplitter>
#include <QTextBrowser>

namespace aitrain_app {
namespace {
class ReportBrowser final : public QTextBrowser {
public:
    explicit ReportBrowser(QWidget* parent = nullptr) : QTextBrowser(parent)
    {
        setOpenLinks(false);
        setOpenExternalLinks(false);
    }
protected:
    QVariant loadResource(int, const QUrl&) override { return {}; }
};
}

bool isImageMember(const QString& relativePath)
{
    const QString suffix = QFileInfo(relativePath).suffix().toLower();
    return QStringList{QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("png"),
        QStringLiteral("bmp"), QStringLiteral("webp"), QStringLiteral("tif"), QStringLiteral("tiff")}.contains(suffix);
}

QString selectProjectArtifact(QWidget* parent,
    const aitrain::ProjectQueryService* query, const QStringList& kinds, const QString& title)
{
    if (!query) return {};
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("ProjectArtifactSelector"));
    dialog.setWindowTitle(title);
    dialog.resize(760, 500);
    auto* layout = new QVBoxLayout(&dialog);
    auto* table = workbenchTable({aitrain_app::workbenchText(QStringLiteral("报告 / 会话")), aitrain_app::workbenchText(QStringLiteral("创建时间"))});
    table->setObjectName(QStringLiteral("ArtifactSelectorTable"));
    layout->addWidget(table, 1);
    auto* status = workbenchHint();
    layout->addWidget(status);
    auto* row = new QHBoxLayout;
    auto* previous = workbenchButton(aitrain_app::workbenchText(QStringLiteral("上一页")));
    auto* next = workbenchButton(aitrain_app::workbenchText(QStringLiteral("下一页")));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText(aitrain_app::workbenchText(QStringLiteral("选择")));
    buttons->button(QDialogButtonBox::Cancel)->setText(aitrain_app::workbenchText(QStringLiteral("取消")));
    row->addWidget(previous);
    row->addWidget(next);
    row->addStretch();
    row->addWidget(buttons);
    layout->addLayout(row);
    QVector<QString> cursors{QString()};
    QString nextCursor;
    QVector<aitrain::ArtifactSnapshot> items;
    const auto refresh = [&]() {
        QString error;
        const auto result = query->artifactCatalog(kinds, {50, cursors.constLast()}, &error);
        items = result.items;
        nextCursor = result.nextCursor;
        table->setRowCount(items.size());
        for (int index = 0; index < items.size(); ++index) {
            table->setItem(index, 0, new QTableWidgetItem(artifactDisplayName(items[index].kind)));
            table->item(index, 0)->setData(Qt::UserRole, items[index].id.toString());
            table->setItem(index, 1, new QTableWidgetItem(items[index].createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
        }
        status->setText(!error.isEmpty() ? error : items.isEmpty()
            ? aitrain_app::workbenchText(QStringLiteral("没有符合条件的已提交报告或会话。"))
            : aitrain_app::workbenchText(QStringLiteral("第 %1 页 · %2 项")).arg(cursors.size()).arg(items.size()));
        previous->setEnabled(cursors.size() > 1);
        next->setEnabled(result.hasMore);
        buttons->button(QDialogButtonBox::Ok)->setEnabled(!items.isEmpty());
        if (!items.isEmpty()) table->selectRow(0);
    };
    QObject::connect(previous, &QPushButton::clicked, &dialog, [&]() { cursors.removeLast(); refresh(); });
    QObject::connect(next, &QPushButton::clicked, &dialog, [&]() { cursors.append(nextCursor); refresh(); });
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    QObject::connect(table, &QTableWidget::cellDoubleClicked, &dialog, [&dialog](int, int) { dialog.accept(); });
    refresh();
    if (dialog.exec() != QDialog::Accepted || table->currentRow() < 0 || table->currentRow() >= items.size()) return {};
    return items.at(table->currentRow()).id.toString();
}

bool selectProjectDataset(QWidget* parent, const aitrain::ProjectQueryService* query,
    DatasetSelection* selection, bool requireSample)
{
    if (!query || !selection) return false;
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("ProjectDatasetSelector"));
    dialog.setWindowTitle(requireSample ? aitrain_app::workbenchText(QStringLiteral("选择数据版本与样本")) : aitrain_app::workbenchText(QStringLiteral("选择数据版本")));
    dialog.resize(880, 580);
    auto* layout = new QVBoxLayout(&dialog);
    auto* split = new QSplitter(Qt::Horizontal);
    auto* datasets = workbenchTable({aitrain_app::workbenchText(QStringLiteral("数据集")), aitrain_app::workbenchText(QStringLiteral("格式"))});
    auto* versions = workbenchTable({aitrain_app::workbenchText(QStringLiteral("快照创建时间")), aitrain_app::workbenchText(QStringLiteral("文件数"))});
    datasets->setObjectName(QStringLiteral("DatasetSelectorTable"));
    versions->setObjectName(QStringLiteral("SnapshotSelectorTable"));
    split->addWidget(datasets);
    split->addWidget(versions);
    split->setChildrenCollapsible(false);
    layout->addWidget(split, 1);
    auto* sampleFiles = workbenchTable({aitrain_app::workbenchText(QStringLiteral("样本图像文件"))});
    sampleFiles->setObjectName(QStringLiteral("SnapshotSampleSelectorTable"));
    sampleFiles->setVisible(requireSample);
    layout->addWidget(sampleFiles, 1);
    auto* status = workbenchHint();
    layout->addWidget(status);
    auto* controls = new QHBoxLayout;
    auto* moreDatasets = workbenchButton(aitrain_app::workbenchText(QStringLiteral("更多数据集")));
    auto* moreVersions = workbenchButton(aitrain_app::workbenchText(QStringLiteral("更多版本")));
    auto* moreFiles = workbenchButton(aitrain_app::workbenchText(QStringLiteral("更多图像文件")));
    moreFiles->setVisible(requireSample);
    controls->addWidget(moreDatasets);
    controls->addWidget(moreVersions);
    controls->addWidget(moreFiles);
    controls->addStretch();
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText(aitrain_app::workbenchText(QStringLiteral("选择")));
    buttons->button(QDialogButtonBox::Cancel)->setText(aitrain_app::workbenchText(QStringLiteral("取消")));
    buttons->button(QDialogButtonBox::Ok)->setEnabled(false);
    controls->addWidget(buttons);
    layout->addLayout(controls);
    QVector<aitrain::DatasetCatalogReadModel> data;
    QVector<aitrain::DatasetSnapshotReadModel> snapshots;
    QString dataCursor, versionCursor, fileCursor;
    QStringList images;
    const auto updateAccept = [&]() {
        buttons->button(QDialogButtonBox::Ok)->setEnabled(versions->currentRow() >= 0
            && versions->currentRow() < snapshots.size() && (!requireSample || sampleFiles->currentRow() >= 0));
    };
    const auto loadFiles = [&](bool append) {
        if (!requireSample || versions->currentRow() < 0 || versions->currentRow() >= snapshots.size()) { updateAccept(); return; }
        if (!append) { images.clear(); sampleFiles->setRowCount(0); fileCursor.clear(); }
        QString error;
        const auto page = query->artifactFiles(snapshots.at(versions->currentRow()).artifactId, {100, fileCursor}, &error);
        fileCursor = page.nextCursor;
        for (const auto& file : page.items) {
            if (!isImageMember(file.relativePath)) continue;
            const int row = sampleFiles->rowCount();
            images.append(file.relativePath);
            sampleFiles->insertRow(row);
            sampleFiles->setItem(row, 0, new QTableWidgetItem(file.relativePath));
        }
        moreFiles->setEnabled(page.hasMore);
        status->setText(!error.isEmpty() ? error : images.isEmpty()
            ? (page.hasMore ? aitrain_app::workbenchText(QStringLiteral("本页没有图像，请继续加载文件。")) : aitrain_app::workbenchText(QStringLiteral("此快照没有可选择的图像文件。")))
            : aitrain_app::workbenchText(QStringLiteral("选择项目快照中的图像；文件会在执行时重新校验。")));
        if (sampleFiles->rowCount()) sampleFiles->selectRow(0);
        updateAccept();
    };
    const auto loadVersions = [&](bool append) {
        if (datasets->currentRow() < 0 || datasets->currentRow() >= data.size()) return;
        if (!append) { snapshots.clear(); versions->setRowCount(0); versionCursor.clear(); sampleFiles->setRowCount(0); }
        QString error;
        const auto page = query->datasetSnapshots(data.at(datasets->currentRow()).datasetId, {50, versionCursor}, &error);
        versionCursor = page.nextCursor;
        const QSignalBlocker blocker(versions);
        for (const auto& item : page.items) {
            snapshots.append(item);
            const int row = versions->rowCount();
            versions->insertRow(row);
            versions->setItem(row, 0, new QTableWidgetItem(item.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss"))));
            versions->setItem(row, 1, new QTableWidgetItem(QString::number(item.fileCount)));
        }
        moreVersions->setEnabled(page.hasMore);
        status->setText(error);
        if (!snapshots.isEmpty()) versions->selectRow(0);
        loadFiles(false);
        updateAccept();
    };
    const auto loadData = [&]() {
        QString error;
        const auto page = query->datasetCatalog({50, dataCursor}, &error);
        dataCursor = page.nextCursor;
        for (const auto& item : page.items) {
            data.append(item);
            const int row = datasets->rowCount();
            datasets->insertRow(row);
            const QString name = item.displayName.isEmpty()
                ? aitrain_app::workbenchText(QStringLiteral("数据集 · %1")).arg(item.latestCreatedAt.toLocalTime().toString(QStringLiteral("MM-dd HH:mm:ss"))) : item.displayName;
            datasets->setItem(row, 0, new QTableWidgetItem(name));
            datasets->setItem(row, 1, new QTableWidgetItem(datasetFormatLabel(item.datasetFormat)));
        }
        moreDatasets->setEnabled(page.hasMore);
        status->setText(!error.isEmpty() ? error : data.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("请先导入数据。")) : QString());
        if (!data.isEmpty() && datasets->currentRow() < 0) datasets->selectRow(0);
    };
    QObject::connect(datasets, &QTableWidget::currentCellChanged, &dialog, [&](int, int, int, int) { loadVersions(false); });
    QObject::connect(versions, &QTableWidget::currentCellChanged, &dialog, [&](int, int, int, int) { loadFiles(false); updateAccept(); });
    QObject::connect(sampleFiles, &QTableWidget::currentCellChanged, &dialog, [&](int, int, int, int) { updateAccept(); });
    QObject::connect(moreDatasets, &QPushButton::clicked, &dialog, loadData);
    QObject::connect(moreVersions, &QPushButton::clicked, &dialog, [&]() { loadVersions(true); });
    QObject::connect(moreFiles, &QPushButton::clicked, &dialog, [&]() { loadFiles(true); });
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    loadData();
    if (dialog.exec() != QDialog::Accepted || versions->currentRow() < 0 || versions->currentRow() >= snapshots.size()) return false;
    selection->snapshot = snapshots.at(versions->currentRow());
    selection->displayName = datasets->item(datasets->currentRow(), 0)->text();
    selection->sampleRelativePath = requireSample && sampleFiles->currentRow() >= 0
        ? sampleFiles->item(sampleFiles->currentRow(), 0)->text() : QString();
    return !requireSample || !selection->sampleRelativePath.isEmpty();
}

void showArtifactReport(QWidget* parent, const aitrain::ProjectQueryService* query,
    const QString& artifactText, const QString& relativePath, const QString& title)
{
    if (!query) return;
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("ArtifactReportDialog"));
    dialog.setWindowTitle(title.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("报告")) : title);
    dialog.resize(920, 640);
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(workbenchHint(relativePath));
    auto* text = new ReportBrowser;
    text->setObjectName(QStringLiteral("ArtifactReportContent"));
    text->setPlainText(aitrain_app::workbenchText(QStringLiteral("正在读取并校验报告…")));
    layout->addWidget(text, 1);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close);
    buttons->button(QDialogButtonBox::Close)->setText(aitrain_app::workbenchText(QStringLiteral("关闭")));
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    aitrain::ArtifactId artifactId;
    QString error;
    QPointer<ReportBrowser> browser(text);
    if (!aitrain::ArtifactId::parse(artifactText, &artifactId, &error)
        || !query->artifactFilePreviewAsync(artifactId, relativePath, &dialog,
            [browser, relativePath](bool ok, aitrain::ArtifactFilePreview preview, QString readError) {
                if (!browser) return;
                if (!ok) { browser->setPlainText(aitrain_app::workbenchText(QStringLiteral("无法读取报告：%1")).arg(readError)); return; }
                if (preview.truncated) { browser->setPlainText(aitrain_app::workbenchText(QStringLiteral("报告超过 4 MB 阅读上限，未显示截断内容。原文件仍保留在任务产物中。"))); return; }
                const QString content = QString::fromUtf8(preview.content);
                if (relativePath.endsWith(QStringLiteral(".html"), Qt::CaseInsensitive)) browser->setHtml(content);
                else browser->setPlainText(content);
            }, 4 * 1024 * 1024, &error)) text->setPlainText(error);
    dialog.exec();
}

} // namespace aitrain_app
