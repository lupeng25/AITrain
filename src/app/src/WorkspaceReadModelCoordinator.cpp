#include "WorkspaceReadModelCoordinator.h"

#include <QTimer>

WorkspaceReadModelCoordinator::WorkspaceReadModelCoordinator(QObject* parent)
    : QObject(parent)
{
}

void WorkspaceReadModelCoordinator::invalidate(RefreshDomains domains)
{
    pending_ |= domains;
    if (flushScheduled_) return;
    flushScheduled_ = true;
    QTimer::singleShot(0, this, [this] { flush(); });
}

void WorkspaceReadModelCoordinator::setGeneration(quint64 generation)
{
    generation_ = generation;
    pending_ = {};
}

quint64 WorkspaceReadModelCoordinator::generation() const
{
    return generation_;
}

void WorkspaceReadModelCoordinator::flush()
{
    flushScheduled_ = false;
    const RefreshDomains domains = pending_;
    pending_ = {};
    const quint64 currentGeneration = generation_;
    if (domains.testFlag(RefreshDomain::TaskList))
        emit refreshTaskList(currentGeneration);
    if (domains.testFlag(RefreshDomain::DatasetCatalog))
        emit refreshDatasetCatalog(currentGeneration);
    if (domains.testFlag(RefreshDomain::ModelRegistry))
        emit refreshModelRegistry(currentGeneration);
    if (domains.testFlag(RefreshDomain::SelectedTask))
        emit refreshSelectedTask(currentGeneration);
    if (domains.testFlag(RefreshDomain::ProjectSummary))
        emit refreshProjectSummary(currentGeneration);
    if (domains.testFlag(RefreshDomain::EnvironmentReport))
        emit refreshEnvironmentReport(currentGeneration);
    if (domains.testFlag(RefreshDomain::DeliveryEvidence))
        emit refreshDeliveryEvidence(currentGeneration);
}
