#include "aitrain/worker/ActiveWorkflowContext.h"

#include <QTest>

class WorkerRuntimeTests : public QObject {
    Q_OBJECT

private slots:
    void enforcesSingleActiveTaskAndIdempotentCancel();
    void durableTerminalMustPrecedeClear();
};

void WorkerRuntimeTests::enforcesSingleActiveTaskAndIdempotentCancel()
{
    aitrain::ActiveWorkflowContext context;
    QString error;
    QVERIFY(context.begin(aitrain::TaskCommandKind::Training, &error));
    QVERIFY(!context.begin(aitrain::TaskCommandKind::Diagnostics, &error));
    QVERIFY(context.requestCancel());
    QVERIFY(context.requestCancel());
    QCOMPARE(context.phase(), aitrain::ActiveWorkflowPhase::CancelRequested);
    QVERIFY(context.cancellationRequested());
}

void WorkerRuntimeTests::durableTerminalMustPrecedeClear()
{
    aitrain::ActiveWorkflowContext context;
    QString error;
    QVERIFY(!context.markDurableTerminal(&error));
    QVERIFY(context.begin(aitrain::TaskCommandKind::EnvironmentCheck, &error));
    QVERIFY(context.markDurableTerminal(&error));
    QCOMPARE(context.phase(), aitrain::ActiveWorkflowPhase::DurableTerminal);
    QVERIFY(!context.requestCancel());
    context.clear();
    QCOMPARE(context.phase(), aitrain::ActiveWorkflowPhase::Inactive);
}

QTEST_APPLESS_MAIN(WorkerRuntimeTests)
#include "tst_worker_runtime.moc"
