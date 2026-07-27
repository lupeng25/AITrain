#pragma once

#include "TaskArtifactPresenter.h"

#include <QAbstractTableModel>
#include <QSortFilterProxyModel>

class TaskListTableModel final : public QAbstractTableModel {
    Q_OBJECT

public:
    enum Role {
        TaskIdRole = Qt::UserRole + 1,
        TaskKindRole,
        TaskStateRole
    };

    explicit TaskListTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;
    void setRows(QVector<TaskListItem> rows);

private:
    QVector<TaskListItem> rows_;
};

class TaskListFilterProxyModel final : public QSortFilterProxyModel {
    Q_OBJECT

public:
    explicit TaskListFilterProxyModel(QObject* parent = nullptr);

    void setTaskKind(const QString& kind);
    void setTaskState(const QString& state);
    void setQuery(const QString& query);

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const override;

private:
    QString kind_;
    QString state_;
    QString query_;
};

class ArtifactFileTableModel final : public QAbstractTableModel {
    Q_OBJECT

public:
    enum Role {
        ArtifactIdRole = Qt::UserRole + 1,
        RelativePathRole,
        ByteCountRole
    };

    explicit ArtifactFileTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;
    void setRows(QVector<ArtifactFileItem> rows);

private:
    QVector<ArtifactFileItem> rows_;
};

class ArtifactTableModel final : public QAbstractTableModel {
    Q_OBJECT

public:
    enum Role {
        ArtifactIdRole = Qt::UserRole + 1
    };

    explicit ArtifactTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;
    void setFiles(const QVector<ArtifactFileItem>& files);

private:
    struct Row final {
        QString artifactId;
        QString kind;
        QString createdAt;
        int fileCount = 0;
        qint64 byteCount = 0;
    };
    QVector<Row> rows_;
};

class MetricTableModel final : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit MetricTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    int columnCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;
    void setRows(QVector<MetricItem> rows);

private:
    QVector<MetricItem> rows_;
};
