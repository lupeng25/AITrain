#include "MainWindow.h"
#include "TaskRuntimeController.h"
#include "DatasetPageController.h"
#include "DeliveryEvidencePageController.h"
#include "DatasetPage.h"
#include "ProjectSessionController.h"
#include "SettingsPageController.h"
#include "WorkspaceReadModelCoordinator.h"
#include "EnvironmentCheckPresenter.h"
#include "EnvironmentPageController.h"
#include "ModelRegistryPageController.h"
#include "RuntimeDeliveryPageController.h"
#include "TrainingPageController.h"

#include "DatasetConversionUiModel.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/CapabilityRegistry.h"
#include "aitrain/core/VisionModelRuntime.h"
#include "aitrain/core/WorkerProtocol.h"

#include <QApplication>
#include <QCheckBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUuid>

using namespace aitrain_app;

void MainWindow::activateProjectUi(const QString& projectName,
    const QString& canonicalRoot, quint64 generation)
{
    Q_UNUSED(projectName)
    modelRegistryPageController_->setProjectContext(true, canonicalRoot);
    environmentPageController_->setProjectContext(true, canonicalRoot);
    runtimeDeliveryPageController_->setProjectContext(true, canonicalRoot);
    trainingPageController_->setProjectContext(true, canonicalRoot);
    datasetPageController_->setProjectContext(true, canonicalRoot);
    deliveryEvidencePageController_->setProjectContext(true, canonicalRoot);
    readModelCoordinator_->setGeneration(generation);
    clearSelectedTaskDetails();
    datasetPageController_->reset();
    if (datasetPage_) {
        for (QLineEdit* field : {
                 datasetPage_->dataQualityDatasetIdEdit,
                 datasetPage_->dataQualityDatasetVersionIdEdit,
                 datasetPage_->dataQualitySnapshotIdEdit,
                 datasetPage_->dataQualitySnapshotArtifactIdEdit,
                 datasetPage_->splitSourceDatasetIdEdit,
                 datasetPage_->splitSourceDatasetVersionIdEdit,
                 datasetPage_->splitSourceSnapshotIdEdit,
                 datasetPage_->splitSourceSnapshotArtifactIdEdit}) {
            if (field) field->clear();
        }
    }

    updateHeaderState();
    readModelCoordinator_->invalidate(RefreshDomain::TaskList
        | RefreshDomain::DatasetCatalog | RefreshDomain::ModelRegistry
        | RefreshDomain::ProjectSummary | RefreshDomain::DeliveryEvidence);
    if (settingsPageController_) {
        settingsPageController_->refresh();
    }
    refreshTrainingDefaults();
    statusBar()->showMessage(uiText("项目已打开：%1").arg(currentProjectName()), 5000);
}
