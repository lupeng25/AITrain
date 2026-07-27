#pragma once

#include "DashboardPage.h"

#include <QObject>
#include <QPointer>

class DashboardPageController final : public QObject {
    Q_OBJECT

public:
    explicit DashboardPageController(
        const aitrain::ProjectQueryService* queryService,
        QObject* parent = nullptr);

    void attachPage(DashboardWorkspacePage* page);
    void setContext(bool projectOpen, const QString& projectName,
        int capabilityCount, const QString& environmentStatus,
        const QString& gpuStatus);
    void refresh();

private:
    void render();

    ProjectSummaryPresenter* summaryPresenter_ = nullptr;
    TaskArtifactPresenter* recentTaskPresenter_ = nullptr;
    QPointer<DashboardWorkspacePage> page_;
    DashboardViewModel viewModel_;
};
