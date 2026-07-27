#include "MainWindow.h"

#include "EvaluationReportView.h"
#include "EnvironmentPage.h"
#include "EnvironmentPageController.h"
#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "SettingsPage.h"
#include "SettingsPageController.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QComboBox>
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
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildSystemSettingsPage()
{
    settingsPage_ = new SettingsWorkspacePage(
        licenseOwner_, licenseExpiry_, this);
    settingsPageController_->attach(settingsPage_);
    connect(settingsPage_, &SettingsWorkspacePage::openProjectRequested,
        this, [this]() { showPage(ProjectPage, tr("项目")); });
    connect(settingsPage_, &SettingsWorkspacePage::openEnvironmentRequested,
        this, [this]() { showPage(EnvironmentPage, tr("环境")); });
    connect(settingsPage_, &SettingsWorkspacePage::runEnvironmentRequested,
        this, [this]() {
            showPage(EnvironmentPage, tr("环境"));
            environmentPageController_->runCheck();
        });
    return settingsPage_;
}

QWidget* MainWindow::buildEnvironmentPage()
{
    environmentPage_ = new EnvironmentWorkspacePage(
        buildDeliveryEvidencePanel(), this);
    environmentPageController_->attach(environmentPage_);
    return environmentPage_;
}
