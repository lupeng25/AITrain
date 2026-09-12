#include "WorkbenchTranslation.h"
#include "DatasetPageController.h"

#include "DatasetCatalogPresenter.h"
#include "DatasetPage.h"
#include "MainWindowSupport.h"
#include "ProjectObjectSelectors.h"

#include <QBuffer>
#include <QComboBox>
#include <QImageReader>
#include <QLineEdit>
#include <QPointer>
#include <QSignalBlocker>

using namespace aitrain_app;

void DatasetPageController::renderCatalog()
{
    if (!page_) return;
    auto* table = page_->datasetListTable;
    const auto& datasets = catalogPresenter_->datasets();
    const QString selectedId = selectAfterRefresh_.isEmpty() ? state_.currentDatasetId : selectAfterRefresh_;
    int selectedRow = -1;
    {
        const QSignalBlocker blocker(table);
        table->setRowCount(datasets.size());
        for (int row = 0; row < datasets.size(); ++row) {
            const auto& item = datasets.at(row);
            auto* name = new QTableWidgetItem(item.displayName);
            name->setData(Qt::UserRole, item.datasetId);
            table->setItem(row, 0, name);
            auto* format = new QTableWidgetItem(datasetFormatLabel(item.datasetFormat));
            format->setData(Qt::UserRole, item.datasetFormat);
            table->setItem(row, 1, format);
            auto* quality = new QTableWidgetItem(item.latestQualityTaskId.isEmpty()
                ? aitrain_app::workbenchText(QStringLiteral("未检查")) : aitrain_app::workbenchText(QStringLiteral("有检查记录")));
            quality->setData(Qt::UserRole, item.latestSnapshotId);
            table->setItem(row, 2, quality);
            table->setItem(row, 3, new QTableWidgetItem(sampleCounts_.contains(item.latestSnapshotId)
                ? QString::number(sampleCounts_.value(item.latestSnapshotId)) : aitrain_app::workbenchText(QStringLiteral("未统计"))));
            auto* version = new QTableWidgetItem(aitrain_app::workbenchText(QStringLiteral("%1 个版本")).arg(item.versionCount));
            version->setData(Qt::UserRole, item.latestArtifactId);
            version->setData(Qt::UserRole + 1, item.latestVersionId);
            table->setItem(row, 4, version);
            if (item.datasetId == selectedId) selectedRow = row;
        }
        if (selectedRow < 0 && !datasets.isEmpty()) selectedRow = 0;
        if (selectedRow >= 0) table->selectRow(selectedRow);
    }
    const QString error = catalogPresenter_->lastError();
    page_->catalogStatusLabel->setText(!projectOpen_ ? aitrain_app::workbenchText(QStringLiteral("请先打开或创建项目。"))
        : !error.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("读取失败：%1")).arg(error)
        : datasets.isEmpty() ? (catalogSearch_.isEmpty() ? aitrain_app::workbenchText(QStringLiteral("还没有数据集，点击“导入数据”开始。")) : aitrain_app::workbenchText(QStringLiteral("整个项目中没有匹配的数据集。")))
        : aitrain_app::workbenchText(QStringLiteral("第 %1 页 · %2 项；样本数与文件数分别统计。")).arg(catalogCursors_.size()).arg(datasets.size()));
    page_->previousPageButton->setEnabled(catalogCursors_.size() > 1);
    page_->nextPageButton->setEnabled(catalogPresenter_->hasMore());
    page_->setSelectionAvailable(selectedRow >= 0);
    if (selectedRow >= 0) selectCatalogRow();
    selectAfterRefresh_.clear();
}

void DatasetPageController::selectCatalogRow()
{
    if (!page_) return;
    const int row = page_->datasetListTable->currentRow();
    const auto& datasets = catalogPresenter_->datasets();
    if (row < 0 || row >= datasets.size()) { page_->setSelectionAvailable(false); return; }
    const auto& item = datasets.at(row);
    const bool changedDataset = state_.currentDatasetId != item.datasetId;
    state_.currentDatasetId = item.datasetId;
    state_.currentDisplayName = item.displayName;
    state_.currentFormat = item.datasetFormat;
    state_.currentPath.clear();
    if (changedDataset) {
        state_.currentSnapshotId.clear();
        state_.currentSampleRelativePath.clear();
        page_->splitTargetDatasetNameEdit->setText(item.displayName + aitrain_app::workbenchText(QStringLiteral(" · 划分")));
    }
    state_.latestQualityTaskId = item.latestQualityTaskId;
    loadSnapshots();
}

void DatasetPageController::loadSnapshots(bool append)
{
    if (!page_ || !projectOpen_) return;
    aitrain::DatasetId datasetId;
    QString error;
    if (!aitrain::DatasetId::parse(state_.currentDatasetId, &datasetId, &error)) return;
    if (!append) { snapshots_.clear(); snapshotCursor_.clear(); }
    const auto page = queryService_->datasetSnapshots(datasetId, {50, snapshotCursor_}, &error);
    snapshotCursor_ = page.nextCursor;
    snapshots_ += page.items;
    int selectedIndex = -1;
    {
        const QSignalBlocker blocker(page_->snapshotCombo);
        page_->snapshotCombo->clear();
        for (int index = 0; index < snapshots_.size(); ++index) {
            const auto& item = snapshots_.at(index);
            page_->snapshotCombo->addItem(item.createdAt.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss")), item.snapshotId.toString());
            if (item.snapshotId.toString() == state_.currentSnapshotId) selectedIndex = index;
        }
        if (selectedIndex < 0 && !snapshots_.isEmpty()) selectedIndex = 0;
        page_->snapshotCombo->setCurrentIndex(selectedIndex);
    }
    page_->snapshotLoadMoreButton->setEnabled(page.hasMore);
    if (!error.isEmpty()) page_->datasetDetailLabel->setText(error);
    if (selectedIndex >= 0) selectSnapshot(selectedIndex);
    else { state_.currentValid = false; page_->setSelectionAvailable(false); }
}

void DatasetPageController::selectSnapshot(int index)
{
    if (!page_ || index < 0 || index >= snapshots_.size()) return;
    const auto& snapshot = snapshots_.at(index);
    const bool changed = state_.currentSnapshotId != snapshot.snapshotId.toString();
    state_.currentDatasetId = snapshot.datasetId.toString();
    state_.currentDatasetVersionId = snapshot.datasetVersionId.toString();
    state_.currentSnapshotId = snapshot.snapshotId.toString();
    state_.currentSnapshotArtifactId = snapshot.artifactId.toString();
    state_.currentFormat = snapshot.datasetFormat;
    state_.currentValid = true;
    state_.latestQualityTaskId = snapshot.latestQualityTaskId.toString();
    for (auto* field : {page_->dataQualityDatasetIdEdit, page_->splitSourceDatasetIdEdit}) field->setText(state_.currentDatasetId);
    for (auto* field : {page_->dataQualityDatasetVersionIdEdit, page_->splitSourceDatasetVersionIdEdit}) field->setText(state_.currentDatasetVersionId);
    for (auto* field : {page_->dataQualitySnapshotIdEdit, page_->splitSourceSnapshotIdEdit}) field->setText(state_.currentSnapshotId);
    for (auto* field : {page_->dataQualitySnapshotArtifactIdEdit, page_->splitSourceSnapshotArtifactIdEdit}) field->setText(state_.currentSnapshotArtifactId);
    page_->datasetDetailLabel->setText(aitrain_app::workbenchText(QStringLiteral("%1 · %2 · 已提交快照"))
        .arg(state_.currentDisplayName, datasetFormatLabel(state_.currentFormat)));
    page_->setSelectionAvailable(true);
    if (changed) {
        ++qualityGeneration_;
        state_.latestQualityArtifactId.clear();
        state_.latestRepairArtifactId.clear();
        page_->validationIssuesTable->setRowCount(0);
        page_->validationSummaryLabel->setText(aitrain_app::workbenchText(QStringLiteral("尚未检查所选快照。")));
        loadSamples();
    }
    if (!state_.latestQualityTaskId.isEmpty()) loadQualityReport(state_.latestQualityTaskId);
    emit selectionChanged();
}

void DatasetPageController::loadSamples(bool append)
{
    if (!page_ || !state_.currentValid || !projectOpen_) return;
    aitrain::ArtifactId artifactId;
    QString error;
    if (!aitrain::ArtifactId::parse(state_.currentSnapshotArtifactId, &artifactId, &error)) return;
    auto* table = page_->datasetPreviewTable;
    if (!append) {
        ++previewGeneration_;
        const QSignalBlocker blocker(table);
        table->setRowCount(0);
        sampleCursor_.clear();
        page_->sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("选择图像后显示预览。")));
    }
    const auto files = queryService_->artifactFiles(artifactId, {100, sampleCursor_}, &error);
    sampleCursor_ = files.nextCursor;
    for (const auto& file : files.items) {
        if (!isImageMember(file.relativePath)) continue;
        const int row = table->rowCount();
        table->insertRow(row);
        table->setItem(row, 0, new QTableWidgetItem(file.relativePath));
        table->setItem(row, 1, new QTableWidgetItem(QStringLiteral("%1 KB").arg((file.byteCount + 1023) / 1024)));
    }
    page_->sampleLoadMoreButton->setEnabled(files.hasMore);
    page_->sampleStatusLabel->setText(!error.isEmpty() ? error
        : aitrain_app::workbenchText(QStringLiteral("已加载 %1 个图像文件%2")).arg(table->rowCount())
            .arg(files.hasMore ? aitrain_app::workbenchText(QStringLiteral("；可继续加载。")) : QStringLiteral("。")));
    if (table->rowCount() && table->currentRow() < 0) table->selectRow(0);
    // YOLO 快照的图像即训练样本；其他格式可能同时包含 mask，不能据此冒充样本数。
    if (!files.hasMore && error.isEmpty() && state_.currentFormat.startsWith(QStringLiteral("yolo_"))) {
        sampleCounts_.insert(state_.currentSnapshotId, table->rowCount());
        const int row = page_->datasetListTable->currentRow();
        if (row >= 0 && page_->datasetListTable->item(row, 2)->data(Qt::UserRole).toString() == state_.currentSnapshotId)
            page_->datasetListTable->item(row, 3)->setText(QString::number(table->rowCount()));
    }
}

void DatasetPageController::previewSample(int row)
{
    if (!page_ || row < 0 || row >= page_->datasetPreviewTable->rowCount()) return;
    const QString member = page_->datasetPreviewTable->item(row, 0)->text();
    state_.currentSampleRelativePath = member;
    if (page_->views->currentIndex() != DatasetWorkspacePage::Detail) return;
    aitrain::ArtifactId artifactId;
    QString error;
    if (!aitrain::ArtifactId::parse(state_.currentSnapshotArtifactId, &artifactId, &error)) return;
    const quint64 generation = ++previewGeneration_;
    page_->sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("正在读取并校验图像…")));
    QPointer<DatasetPageController> self(this);
    if (!queryService_->artifactFilePreviewAsync(artifactId, member, this,
            [self, generation](bool ok, aitrain::ArtifactFilePreview preview, QString readError) {
                if (!self || !self->page_ || generation != self->previewGeneration_) return;
                if (!ok) { self->page_->sampleImageLabel->setText(readError); return; }
                if (preview.truncated) { self->page_->sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("图像超过 4 MB 预览上限；原文件仍保留在数据快照中。"))); return; }
                QBuffer buffer(&preview.content);
                buffer.open(QIODevice::ReadOnly);
                QImageReader reader(&buffer);
                const QSize size = reader.size();
                if (!size.isValid() || qint64(size.width()) * size.height() > 40000000) {
                    self->page_->sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("图像尺寸过大或格式无法预览；原文件仍保留在快照中。")));
                    return;
                }
                reader.setScaledSize(size.scaled(1200, 900, Qt::KeepAspectRatio));
                const QImage image = reader.read();
                if (image.isNull()) self->page_->sampleImageLabel->setText(aitrain_app::workbenchText(QStringLiteral("无法解码该图像。")));
                else self->page_->sampleImageLabel->setImage(QPixmap::fromImage(image));
            }, 4 * 1024 * 1024, &error)) page_->sampleImageLabel->setText(error);
}
