#pragma once

#include <QFlags>
#include <QObject>

enum class RefreshDomain : quint32 {
    None = 0,
    TaskList = 1u << 0,
    DatasetCatalog = 1u << 1,
    ModelRegistry = 1u << 2,
    SelectedTask = 1u << 3,
    ProjectSummary = 1u << 4,
    EnvironmentReport = 1u << 5,
    DeliveryEvidence = 1u << 6
};
Q_DECLARE_FLAGS(RefreshDomains, RefreshDomain)
Q_DECLARE_OPERATORS_FOR_FLAGS(RefreshDomains)

class WorkspaceReadModelCoordinator final : public QObject {
    Q_OBJECT

public:
    explicit WorkspaceReadModelCoordinator(QObject* parent = nullptr);

    void invalidate(RefreshDomains domains);
    void setGeneration(quint64 generation);
    quint64 generation() const;

signals:
    void refreshTaskList(quint64 generation);
    void refreshDatasetCatalog(quint64 generation);
    void refreshModelRegistry(quint64 generation);
    void refreshSelectedTask(quint64 generation);
    void refreshProjectSummary(quint64 generation);
    void refreshEnvironmentReport(quint64 generation);
    void refreshDeliveryEvidence(quint64 generation);

private:
    void flush();

    RefreshDomains pending_;
    quint64 generation_ = 0;
    bool flushScheduled_ = false;
};
