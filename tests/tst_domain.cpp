#include "aitrain/domain/DomainTypes.h"

#include <QTest>

class DomainTests : public QObject {
    Q_OBJECT

private slots:
    void identifiersAreTypedAndStrict();
    void taskStateMachineRejectsInvalidTransitions();
    void terminalTransitionsAreIdempotent();
    void failureCodesRoundTrip();
};

void DomainTests::identifiersAreTypedAndStrict()
{
    const aitrain::TaskId taskId = aitrain::TaskId::create();
    QVERIFY(taskId.isValid());

    aitrain::TaskId parsed;
    QString error;
    QVERIFY2(aitrain::TaskId::parse(taskId.toString(), &parsed, &error), qPrintable(error));
    QVERIFY(parsed == taskId);

    QVERIFY(!aitrain::TaskId::parse(QStringLiteral("not-a-uuid"), &parsed, &error));
    QVERIFY(!aitrain::TaskId::parse(QString(), &parsed, &error));
}

void DomainTests::taskStateMachineRejectsInvalidTransitions()
{
    using aitrain::TaskState;
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Created, TaskState::Queued));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Queued, TaskState::Starting));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Starting, TaskState::Running));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Running, TaskState::Succeeded));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Running, TaskState::CancelRequested));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::CancelRequested, TaskState::Canceled));

    QVERIFY(!aitrain::isValidTaskStateTransition(TaskState::Created, TaskState::Running));
    QVERIFY(!aitrain::isValidTaskStateTransition(TaskState::Queued, TaskState::Succeeded));
    QVERIFY(!aitrain::isValidTaskStateTransition(TaskState::Succeeded, TaskState::Running));
    QVERIFY(!aitrain::isValidTaskStateTransition(TaskState::Canceled, TaskState::Failed));
}

void DomainTests::terminalTransitionsAreIdempotent()
{
    using aitrain::TaskState;
    QVERIFY(aitrain::isIdempotentTerminalTransition(TaskState::Succeeded, TaskState::Succeeded));
    QVERIFY(aitrain::isIdempotentTerminalTransition(TaskState::Failed, TaskState::Failed));
    QVERIFY(aitrain::isIdempotentTerminalTransition(TaskState::Canceled, TaskState::Canceled));
    QVERIFY(!aitrain::isIdempotentTerminalTransition(TaskState::Failed, TaskState::Canceled));
    QVERIFY(aitrain::isValidTaskStateTransition(TaskState::Succeeded, TaskState::Succeeded));
}

void DomainTests::failureCodesRoundTrip()
{
    const aitrain::FailureCode expected = aitrain::FailureCode::RuntimeNotImplemented;
    const QString text = aitrain::failureCodeToString(expected);
    QCOMPARE(text, QStringLiteral("runtime_not_implemented"));

    aitrain::FailureCode parsed = aitrain::FailureCode::None;
    QVERIFY(aitrain::failureCodeFromString(text, &parsed));
    QCOMPARE(parsed, expected);
    QVERIFY(!aitrain::failureCodeFromString(QStringLiteral("hardware-blocked"), &parsed));

    aitrain::Failure failure;
    QVERIFY(!failure.isFailure());
    failure.code = aitrain::FailureCode::InvalidDataset;
    QVERIFY(failure.isFailure());
}

QTEST_MAIN(DomainTests)
#include "tst_domain.moc"
