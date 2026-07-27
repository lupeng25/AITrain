#pragma once

#include "ProjectPage.h"

#include <QObject>
#include <QPointer>

class ProjectSessionController;

namespace aitrain {
class ProjectQueryService;
}

// 项目页 Controller 拥有表单命令构造和项目摘要 Presenter。
class ProjectPageController final : public QObject {
    Q_OBJECT

public:
    explicit ProjectPageController(
        const aitrain::ProjectQueryService* queryService,
        ProjectSessionController* sessionController,
        QObject* parent = nullptr);

    void attachPage(ProjectWorkspacePage* page);
    void setContext(bool workspaceOpen, const QString& projectName);
    void setDefaultRoot(const QString& root, bool apply);
    void refresh();

signals:
    void sessionRequestAccepted();

private:
    void request(aitrain_app::ProjectSessionOperation operation);
    void render();

    ProjectSessionController* sessionController_ = nullptr;
    ProjectSummaryPresenter* summaryPresenter_ = nullptr;
    QPointer<ProjectWorkspacePage> page_;
    ProjectPageViewModel viewModel_;
};
