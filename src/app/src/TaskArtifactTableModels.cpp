#include "TaskArtifactTableModels.h"

#include <QHash>
#include <QStringList>

#include <utility>

TaskListTableModel::TaskListTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int TaskListTableModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : rows_.size();
}

int TaskListTableModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : 7;
}

QVariant TaskListTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= rows_.size()) return {};
    const TaskListItem& row = rows_.at(index.row());
    if (role == TaskIdRole) return row.taskId;
    if (role == TaskKindRole) return QString();
    if (role == TaskStateRole) return row.state;
    if (role != Qt::DisplayRole && role != Qt::ToolTipRole) return {};
    switch (index.column()) {
    case 0: return row.taskId.left(8);
    case 1: return tr("任务");
    case 2: return row.capabilityId;
    case 3: return row.taskType;
    case 4: return row.stateLabel;
    case 5: return row.updatedAt;
    case 6: return row.message;
    default: return {};
    }
}

QVariant TaskListTableModel::headerData(
    int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
    static const QStringList headers{tr("任务"), tr("类别"), tr("内置能力"),
        tr("类型"), tr("状态"), tr("更新时间"), tr("消息")};
    return headers.value(section);
}

void TaskListTableModel::setRows(QVector<TaskListItem> rows)
{
    beginResetModel();
    rows_ = std::move(rows);
    endResetModel();
}

TaskListFilterProxyModel::TaskListFilterProxyModel(QObject* parent)
    : QSortFilterProxyModel(parent)
{
    setDynamicSortFilter(true);
}

void TaskListFilterProxyModel::setTaskKind(const QString& kind)
{
    if (kind_ == kind) return;
    kind_ = kind;
    invalidateFilter();
}

void TaskListFilterProxyModel::setTaskState(const QString& state)
{
    if (state_ == state) return;
    state_ = state;
    invalidateFilter();
}

void TaskListFilterProxyModel::setQuery(const QString& query)
{
    const QString normalized = query.trimmed();
    if (query_ == normalized) return;
    query_ = normalized;
    invalidateFilter();
}

bool TaskListFilterProxyModel::filterAcceptsRow(
    int sourceRow, const QModelIndex& sourceParent) const
{
    const QModelIndex first = sourceModel()->index(sourceRow, 0, sourceParent);
    if (!kind_.isEmpty()
        && first.data(TaskListTableModel::TaskKindRole).toString() != kind_) return false;
    if (!state_.isEmpty()
        && first.data(TaskListTableModel::TaskStateRole).toString() != state_) return false;
    if (query_.isEmpty()) return true;
    for (int column = 0; column < sourceModel()->columnCount(sourceParent); ++column) {
        if (sourceModel()->index(sourceRow, column, sourceParent).data().toString()
                .contains(query_, Qt::CaseInsensitive)) return true;
    }
    return false;
}

ArtifactFileTableModel::ArtifactFileTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int ArtifactFileTableModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : rows_.size();
}

int ArtifactFileTableModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : 4;
}

QVariant ArtifactFileTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= rows_.size()) return {};
    const ArtifactFileItem& row = rows_.at(index.row());
    if (role == ArtifactIdRole) return row.artifactId;
    if (role == RelativePathRole) return row.relativePath;
    if (role == ByteCountRole) return row.byteCount;
    if (role != Qt::DisplayRole && role != Qt::ToolTipRole) return {};
    switch (index.column()) {
    case 0: return row.kind;
    case 1: return row.relativePath.isEmpty() ? tr("（无文件清单）") : row.relativePath;
    case 2: return tr("SHA-256 %1 · %2 bytes")
        .arg(row.sha256.isEmpty() ? QStringLiteral("--") : row.sha256)
        .arg(row.byteCount);
    case 3: return row.createdAt;
    default: return {};
    }
}

QVariant ArtifactFileTableModel::headerData(
    int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
    static const QStringList headers{
        tr("产物类型"), tr("包内相对路径"), tr("完整性"), tr("提交时间")};
    return headers.value(section);
}

void ArtifactFileTableModel::setRows(QVector<ArtifactFileItem> rows)
{
    beginResetModel();
    rows_ = std::move(rows);
    endResetModel();
}

ArtifactTableModel::ArtifactTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int ArtifactTableModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : rows_.size();
}

int ArtifactTableModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : 4;
}

QVariant ArtifactTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= rows_.size()) return {};
    const Row& row = rows_.at(index.row());
    if (role == ArtifactIdRole) return row.artifactId;
    if (role != Qt::DisplayRole && role != Qt::ToolTipRole) return {};
    switch (index.column()) {
    case 0: return row.artifactId.left(8);
    case 1: return row.kind;
    case 2: return tr("%1 个文件 / %2 bytes").arg(row.fileCount).arg(row.byteCount);
    case 3: return row.createdAt;
    default: return {};
    }
}

QVariant ArtifactTableModel::headerData(
    int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
    static const QStringList headers{
        tr("Artifact"), tr("类型"), tr("清单"), tr("提交时间")};
    return headers.value(section);
}

void ArtifactTableModel::setFiles(const QVector<ArtifactFileItem>& files)
{
    beginResetModel();
    rows_.clear();
    QHash<QString, int> indexes;
    for (const ArtifactFileItem& file : files) {
        auto existing = indexes.constFind(file.artifactId);
        if (existing == indexes.cend()) {
            Row row;
            row.artifactId = file.artifactId;
            row.kind = file.kind;
            row.createdAt = file.createdAt;
            rows_.append(row);
            const int index = rows_.size() - 1;
            indexes.insert(file.artifactId, index);
            existing = indexes.constFind(file.artifactId);
        }
        Row& row = rows_[existing.value()];
        if (!file.relativePath.isEmpty()) ++row.fileCount;
        row.byteCount += file.byteCount;
    }
    endResetModel();
}

MetricTableModel::MetricTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int MetricTableModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : rows_.size();
}

int MetricTableModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : 4;
}

QVariant MetricTableModel::data(const QModelIndex& index, int role) const
{
    if (!index.isValid() || index.row() < 0 || index.row() >= rows_.size()
        || (role != Qt::DisplayRole && role != Qt::ToolTipRole)) return {};
    const MetricItem& row = rows_.at(index.row());
    switch (index.column()) {
    case 0: return row.name;
    case 1: return QString::number(row.value, 'g', 12);
    case 2: return row.occurredAt;
    case 3: return tr("持久化事件");
    default: return {};
    }
}

QVariant MetricTableModel::headerData(
    int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
    static const QStringList headers{
        tr("指标"), tr("值"), tr("发生时间"), tr("来源")};
    return headers.value(section);
}

void MetricTableModel::setRows(QVector<MetricItem> rows)
{
    beginResetModel();
    rows_ = std::move(rows);
    endResetModel();
}
