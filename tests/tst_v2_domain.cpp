#include "aitrain/v2/DomainTypes.h"

#include <QTest>

class V2DomainTests : public QObject {
    Q_OBJECT

private slots:
    void identifiersAreTypedAndStrict();
    void taskStateMachineRejectsInvalidTransitions();
    void terminalTransitionsAreIdempotent();
    void failureCodesRoundTrip();
};

void V2DomainTests::identifiersAreTypedAndStrict()
{
    const aitrain::v2::TaskId taskId = aitrain::v2::TaskId::create();
    QVERIFY(taskId.isValid());

    aitrain::v2::TaskId parsed;
    QString error;
    QVERIFY2(aitrain::v2::TaskId::parse(taskId.toString(), &parsed, &error), qPrintable(error));
    QVERIFY(parsed == taskId);

    QVERIFY(!aitrain::v2::TaskId::parse(QStringLiteral("not-a-uuid"), &parsed, &error));
    QVERIFY(!aitrain::v2::TaskId::parse(QString(), &parsed, &error));
}

void V2DomainTests::taskStateMachineRejectsInvalidTransitions()
{
    using aitrain::v2::TaskState;
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Created, TaskState::Queued));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Queued, TaskState::Starting));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Starting, TaskState::Running));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Running, TaskState::Succeeded));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Running, TaskState::CancelRequested));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::CancelRequested, TaskState::Canceled));

    QVERIFY(!aitrain::v2::isValidTaskStateTransition(TaskState::Created, TaskState::Running));
    QVERIFY(!aitrain::v2::isValidTaskStateTransition(TaskState::Queued, TaskState::Succeeded));
    QVERIFY(!aitrain::v2::isValidTaskStateTransition(TaskState::Succeeded, TaskState::Running));
    QVERIFY(!aitrain::v2::isValidTaskStateTransition(TaskState::Canceled, TaskState::Failed));
}

void V2DomainTests::terminalTransitionsAreIdempotent()
{
    using aitrain::v2::TaskState;
    QVERIFY(aitrain::v2::isIdempotentTerminalTransition(TaskState::Succeeded, TaskState::Succeeded));
    QVERIFY(aitrain::v2::isIdempotentTerminalTransition(TaskState::Failed, TaskState::Failed));
    QVERIFY(aitrain::v2::isIdempotentTerminalTransition(TaskState::Canceled, TaskState::Canceled));
    QVERIFY(!aitrain::v2::isIdempotentTerminalTransition(TaskState::Failed, TaskState::Canceled));
    QVERIFY(aitrain::v2::isValidTaskStateTransition(TaskState::Succeeded, TaskState::Succeeded));
}

void V2DomainTests::failureCodesRoundTrip()
{
    const aitrain::v2::FailureCode expected = aitrain::v2::FailureCode::RuntimeNotImplemented;
    const QString text = aitrain::v2::failureCodeToString(expected);
    QCOMPARE(text, QStringLiteral("runtime_not_implemented"));

    aitrain::v2::FailureCode parsed = aitrain::v2::FailureCode::None;
    QVERIFY(aitrain::v2::failureCodeFromString(text, &parsed));
    QCOMPARE(parsed, expected);
    QVERIFY(!aitrain::v2::failureCodeFromString(QStringLiteral("hardware-blocked"), &parsed));

    aitrain::v2::Failure failure;
    QVERIFY(!failure.isFailure());
    failure.code = aitrain::v2::FailureCode::InvalidDataset;
    QVERIFY(failure.isFailure());
}

QTEST_MAIN(V2DomainTests)
#include "tst_v2_domain.moc"
