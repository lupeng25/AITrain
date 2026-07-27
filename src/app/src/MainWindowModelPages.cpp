#include "MainWindow.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"
#include "ModelRegistryPage.h"
#include "ModelRegistryPageController.h"

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
    modelRegistryPageController_->refresh();
    return modelRegistryPage_;
}
