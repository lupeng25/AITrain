#include "WorkbenchTranslation.h"
#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "ModelRegistryPage.h"
#include "ModelRegistryPageController.h"
#include "TaskArtifactPageController.h"

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
#include <QSizePolicy>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

using namespace aitrain_app;

QWidget* MainWindow::buildModelRegistryPage()
{
    modelRegistryPage_ = new ModelRegistryWorkspacePage(this);
    modelRegistryPageController_->attach(modelRegistryPage_);
    connect(modelRegistryPage_, &ModelRegistryWorkspacePage::reportsRequested, this, [this]() { showPage(EvidencePage, aitrain_app::workbenchText(QStringLiteral("验收报告"))); });
    connect(modelRegistryPage_, &ModelRegistryWorkspacePage::sourceTaskRequested, this, [this](const QString& id) { showPage(TaskQueuePage, aitrain_app::workbenchText(QStringLiteral("任务记录"))); taskArtifactPageController_->openTask(id); });
    modelRegistryPageController_->refresh();
    return modelRegistryPage_;
}
