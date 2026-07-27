#include "MainWindow.h"
#include "DatasetPage.h"
#include "DatasetPageController.h"
#include "ModelRegistryPageController.h"
#include "RuntimeDeliveryPageController.h"
#include "ModelRegistryPresenter.h"
#include "TaskArtifactPageController.h"
#include "ProjectSessionController.h"

#include "EvaluationReportView.h"
#include "InfoPanel.h"
#include "LanguageSupport.h"
#include "MainWindowSupport.h"
#include "aitrain/core/VisionModelRuntime.h"

#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QDateTime>
#include <QDesktopServices>
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
#include <QTableView>
#include <QTableWidgetItem>
#include <QTime>
#include <QToolButton>
#include <QVBoxLayout>
#include <QUrl>
#include <QUuid>

using namespace aitrain_app;

void MainWindow::updateRecentTasks()
{
    if (taskArtifactPageController_ && projectSessionController_
        && projectSessionController_->isOpen()) {
        taskArtifactPageController_->refresh({100, {}});
    }
    updateDashboardSummary();
}
void MainWindow::updateDatasetList()
{
    datasetPageController_->refreshCatalog();
    updateDashboardSummary();
}
void MainWindow::updateAnnotationToolStatus()
{
    if (datasetPage_ && datasetPage_->annotationToolStatusLabel) {
        datasetPage_->annotationToolStatusLabel->setText(xAnyLabelingStatusText());
        datasetPage_->annotationToolStatusLabel->setToolTip(QDir::toNativeSeparators(resolvedXAnyLabelingProgram()));
    }
}

void MainWindow::setDatasetRepairLoopRows(const QString& summary, const QVector<QStringList>& rows)
{
    if (!datasetPage_) {
        return;
    }
    if (datasetPage_->datasetRepairLoopLabel) {
        datasetPage_->datasetRepairLoopLabel->setText(summary);
    }
    if (!datasetPage_->datasetRepairLoopTable) {
        return;
    }

    datasetPage_->datasetRepairLoopTable->setRowCount(0);
    if (rows.isEmpty()) {
        datasetPage_->datasetRepairLoopTable->insertRow(0);
        datasetPage_->datasetRepairLoopTable->setItem(0, 0, new QTableWidgetItem(uiText("等待")));
        datasetPage_->datasetRepairLoopTable->setItem(0, 1, new QTableWidgetItem(uiText("未开始")));
        datasetPage_->datasetRepairLoopTable->setItem(0, 2, new QTableWidgetItem(uiText("生成质量报告后进入修复闭环。")));
        return;
    }

    for (const QStringList& rowValues : rows) {
        const int row = datasetPage_->datasetRepairLoopTable->rowCount();
        datasetPage_->datasetRepairLoopTable->insertRow(row);
        for (int column = 0; column < datasetPage_->datasetRepairLoopTable->columnCount(); ++column) {
            const QString value = rowValues.value(column);
            auto* item = new QTableWidgetItem(value);
            item->setToolTip(value);
            datasetPage_->datasetRepairLoopTable->setItem(row, column, item);
        }
    }
}

void MainWindow::clearSelectedTaskDetails()
{
    if (taskArtifactPageController_) taskArtifactPageController_->clearSelection();
}

void MainWindow::updateModelRegistry()
{
    modelRegistryPageController_->setProjectContext(
        projectSessionController_ && projectSessionController_->isOpen(),
        currentProjectPath());
    modelRegistryPageController_->refresh();
}

void MainWindow::syncModelPackageCombos()
{
    const QVector<ModelPackageListItem>& packages =
        modelRegistryPageController_->packages();
    runtimeDeliveryPageController_->setModelPackages(packages);
}
