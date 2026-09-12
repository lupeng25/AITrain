#pragma once

#include <QObject>

class DeliveryEvidencePresenter;
class DiagnosticBundlePresenter;
class DeliveryEvidenceWorkspacePage;
class QLineEdit;
class TaskRuntimeController;
struct TaskViewState;
namespace aitrain { class ProjectQueryService; }

class DeliveryEvidencePageController final : public QObject
{
    Q_OBJECT

public:
    explicit DeliveryEvidencePageController(
        const aitrain::ProjectQueryService* queryService,
        TaskRuntimeController* taskRuntime, QObject* parent = nullptr);

    void attach(DeliveryEvidenceWorkspacePage* page);
    void setProjectContext(bool projectOpen, const QString& projectRoot);
    void setWorkerExecutable(const QString& executable);
    void browseReport(QLineEdit* target);
    void refresh();
    void applyTaskViewState(const TaskViewState& state);

public slots:
    void importOcrOfficialReports();
    void runOcrAcceptance();
    void collectDiagnostics();
    void importAcceptanceEvidence();

signals:
    void taskStarted(const QString& taskId, const QString& workflowKind);
    void statusChanged(const QString& text);

private:
    void render();
    void bindSnapshot(int index);
    void selectReport(int index);

    DeliveryEvidenceWorkspacePage* page_ = nullptr;
    DeliveryEvidencePresenter* presenter_ = nullptr;
    DiagnosticBundlePresenter* diagnosticPresenter_ = nullptr;
    TaskRuntimeController* taskRuntime_ = nullptr;
    bool projectOpen_ = false;
    QString projectRoot_;
    QString workerExecutable_;
    const aitrain::ProjectQueryService* queryService_ = nullptr;
    QString activeTaskId_;
};
