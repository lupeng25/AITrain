#pragma once

#include "MainWindowSupport.h"
#include "ProjectSummaryPresenter.h"

#include <QScrollArea>

class QLabel;
class QLineEdit;
class QPushButton;

struct ProjectPageViewModel final {
    bool workspaceOpen = false;
    QString projectName;
    QString queryError;
    ProjectSummaryViewModel summary;
};

// 项目页只拥有视觉控件和破坏性操作确认，不直接访问 Workspace 或 Store。
class ProjectWorkspacePage final : public QScrollArea {
    Q_OBJECT

public:
    explicit ProjectWorkspacePage(const QString& defaultRoot,
        QWidget* parent = nullptr);

    QString projectName() const;
    QString projectRoot() const;
    void setProjectRoot(const QString& root);
    void setBusy(bool busy);
    void setStatus(const QString& status);
    void render(const ProjectPageViewModel& viewModel);
    void showOperationError(const QString& message);

signals:
    void operationRequested(aitrain_app::ProjectSessionOperation operation);

private:
    void requestRebuild();

    QLineEdit* projectNameEdit_ = nullptr;
    QLineEdit* projectRootEdit_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* pathSummaryLabel_ = nullptr;
    QLabel* sqliteSummaryLabel_ = nullptr;
    QLabel* datasetSummaryLabel_ = nullptr;
    QLabel* taskSummaryLabel_ = nullptr;
    QLabel* modelSummaryLabel_ = nullptr;
    QPushButton* createButton_ = nullptr;
    QPushButton* openButton_ = nullptr;
    QPushButton* rebuildButton_ = nullptr;
    QPushButton* browseButton_ = nullptr;
};
