#include "MainWindow.h"

#include "DashboardPage.h"
#include "DashboardPageController.h"
#include "ProjectPage.h"
#include "ProjectPageController.h"
#include "ProjectSessionController.h"
#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildDashboardPage()
{
    dashboardPage_ = new DashboardWorkspacePage;
    dashboardPageController_->attachPage(dashboardPage_);
    connect(dashboardPage_, &DashboardWorkspacePage::routeRequested, this,
        [this](DashboardRoute route) {
            switch (route) {
            case DashboardRoute::Project:
                showPage(ProjectPage, uiText("项目")); break;
            case DashboardRoute::Dataset:
                showPage(DatasetPage, uiText("数据集")); break;
            case DashboardRoute::Training:
                showPage(TrainingPage, uiText("训练实验")); break;
            case DashboardRoute::TaskArtifact:
                showPage(TaskQueuePage, uiText("任务与产物")); break;
            case DashboardRoute::ModelRegistry:
                showPage(ModelRegistryPage, uiText("模型库")); break;
            case DashboardRoute::RuntimeDelivery:
                showPage(DeploymentPage, uiText("部署验证")); break;
            }
        });
    return dashboardPage_;
}

QWidget* MainWindow::buildProjectPage()
{
    projectPage_ = new ProjectWorkspacePage(configuredDefaultProjectPath());
    projectPageController_->attachPage(projectPage_);
    projectPageController_->setContext(
        projectSessionController_ && projectSessionController_->isOpen(),
        currentProjectName());
    updateProjectSummary();
    return projectPage_;
}
