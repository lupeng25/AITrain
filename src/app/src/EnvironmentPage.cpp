#include "EnvironmentPage.h"

#include "InfoPanel.h"
#include "MainWindowSupport.h"

#include <QAbstractItemView>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QLabel>
#include <QPushButton>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

using namespace aitrain_app;

namespace {

QLabel* metricValue(InfoPanel* card)
{
    return card->findChild<QLabel*>(QStringLiteral("CompactMetricValue"));
}

} // namespace

EnvironmentWorkspacePage::EnvironmentWorkspacePage(
    QWidget* deliveryEvidencePage, QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(18, 0, 18, 18);
    layout->setSpacing(16);
    auto* runButton = primaryButton(tr("执行环境自检"));
    connect(runButton, &QPushButton::clicked,
        this, &EnvironmentWorkspacePage::runRequested);

    auto* toolbar = new QFrame;
    toolbar->setObjectName(QStringLiteral("WorkspaceToolbar"));
    auto* toolbarLayout = new QHBoxLayout(toolbar);
    auto* contextLabel = new QLabel(tr("运行时与交付证据"));
    contextLabel->setObjectName(QStringLiteral("WorkspaceToolbarTitle"));
    statusLabel_ = inlineStatusLabel(tr("尚未执行环境自检。"));
    statusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    allowLabelToShrink(statusLabel_);
    toolbarLayout->addWidget(contextLabel);
    toolbarLayout->addWidget(statusLabel_, 1);
    toolbarLayout->addWidget(runButton);

    auto* summary = new QFrame;
    summary->setObjectName(QStringLiteral("ActionStrip"));
    auto* summaryLayout = new QGridLayout(summary);
    auto* okCard = createCompactSummaryCard(
        tr("通过"), QStringLiteral("0"), tr("可用依赖"));
    auto* warningCard = createCompactSummaryCard(
        tr("警告"), QStringLiteral("0"), tr("可继续但需关注"));
    auto* missingCard = createCompactSummaryCard(
        tr("缺失"), QStringLiteral("0"), tr("会阻塞相关能力"));
    auto* uncheckedCard = createCompactSummaryCard(
        tr("未检测"), QStringLiteral("0"), tr("等待 Worker 自检"));
    okLabel_ = metricValue(okCard);
    warningLabel_ = metricValue(warningCard);
    missingLabel_ = metricValue(missingCard);
    uncheckedLabel_ = metricValue(uncheckedCard);
    summaryLayout->addWidget(okCard, 0, 0);
    summaryLayout->addWidget(warningCard, 0, 1);
    summaryLayout->addWidget(missingCard, 0, 2);
    summaryLayout->addWidget(uncheckedCard, 0, 3);

    auto* panel = new InfoPanel(tr("检查明细"));
    table_ = new QTableWidget(0, 3);
    table_->setObjectName(QStringLiteral("EnvironmentTable"));
    table_->setHorizontalHeaderLabels(
        QStringList() << tr("检查项") << tr("状态") << tr("说明"));
    table_->setAlternatingRowColors(true);
    table_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table_->setSelectionBehavior(QAbstractItemView::SelectRows);
    table_->verticalHeader()->setVisible(false);
    table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    table_->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    panel->bodyLayout()->addWidget(mutedLabel(tr(
        "状态来自 Worker 已提交的 Environment Check 报告；未提交外部硬件验收证据不会被视为通过。")));
    panel->bodyLayout()->addWidget(table_);

    auto* runtimeTab = new QWidget;
    auto* runtimeLayout = new QVBoxLayout(runtimeTab);
    runtimeLayout->setContentsMargins(0, 0, 0, 0);
    runtimeLayout->setSpacing(16);
    runtimeLayout->addWidget(summary);
    runtimeLayout->addWidget(panel, 1);
    auto* tabs = new QTabWidget;
    tabs->setObjectName(QStringLiteral("EnvironmentTabs"));
    tabs->addTab(runtimeTab, tr("运行环境"));
    tabs->addTab(deliveryEvidencePage, tr("交付证据"));
    layout->addWidget(toolbar);
    layout->addWidget(tabs, 1);
    renderReport({});
}

void EnvironmentWorkspacePage::renderReport(const QJsonObject& report)
{
    table_->setRowCount(0);
    const auto addRow = [this](const QString& name, const QString& state,
        const QString& message) {
        const int row = table_->rowCount();
        table_->insertRow(row);
        table_->setItem(row, 0, new QTableWidgetItem(name));
        auto* stateItem = new QTableWidgetItem(
            state == QStringLiteral("ok") ? tr("通过")
            : state == QStringLiteral("missing") ? tr("缺失")
            : state == QStringLiteral("hardware-blocked") ? tr("硬件受限")
            : state == QStringLiteral("warning") ? tr("警告") : tr("未检测"));
        stateItem->setData(Qt::UserRole, state);
        table_->setItem(row, 1, stateItem);
        table_->setItem(row, 2,
            new QTableWidgetItem(message.isEmpty() ? tr("未提供说明。") : message));
    };
    if (report.isEmpty()) {
        const QStringList names = {
            QStringLiteral("NVIDIA Driver"), QStringLiteral("CUDA Runtime"),
            QStringLiteral("cuDNN"), QStringLiteral("TensorRT"),
            QStringLiteral("ONNX Runtime"), QStringLiteral("Qt Runtime Modules"),
            tr("内置能力"), QStringLiteral("Worker")};
        for (const QString& name : names) {
            addRow(name, QStringLiteral("unchecked"), tr("点击执行环境自检。"));
        }
    } else {
        for (const QJsonValue& value : report.value(QStringLiteral("checks")).toArray()) {
            const QJsonObject check = value.toObject();
            addRow(check.value(QStringLiteral("name")).toString(),
                check.value(QStringLiteral("status")).toString(),
                check.value(QStringLiteral("message")).toString());
        }
        const QJsonObject profiles = report.value(QStringLiteral("profiles")).toObject();
        for (auto it = profiles.constBegin(); it != profiles.constEnd(); ++it) {
            const QJsonObject profile = it.value().toObject();
            QString message = profile.value(QStringLiteral("message")).toString();
            if (message.isEmpty()) {
                QStringList hints;
                for (const QJsonValue& hint :
                    profile.value(QStringLiteral("repairHints")).toArray()) {
                    hints.append(hint.toString());
                }
                message = hints.join(QStringLiteral("；"));
            }
            addRow(profile.value(QStringLiteral("title")).toString(it.key()),
                profile.value(QStringLiteral("status")).toString(), message);
        }
    }
    updateSummary();
}

void EnvironmentWorkspacePage::setChecking()
{
    for (int row = 0; row < table_->rowCount(); ++row) {
        auto* state = new QTableWidgetItem(tr("检测中"));
        state->setData(Qt::UserRole, QStringLiteral("unchecked"));
        table_->setItem(row, 1, state);
        table_->setItem(row, 2, new QTableWidgetItem(tr("等待 Worker 返回结果。")));
    }
    statusLabel_->setText(tr("环境自检正在执行。"));
    updateSummary();
}

void EnvironmentWorkspacePage::updateSummary()
{
    int ok = 0;
    int warning = 0;
    int blocked = 0;
    int missing = 0;
    int unchecked = 0;
    for (int row = 0; row < table_->rowCount(); ++row) {
        const auto* item = table_->item(row, 1);
        const QString state = item ? item->data(Qt::UserRole).toString() : QString();
        if (state == QStringLiteral("ok")) ++ok;
        else if (state == QStringLiteral("hardware-blocked")) {
            ++warning;
            ++blocked;
        } else if (state == QStringLiteral("warning")) ++warning;
        else if (state == QStringLiteral("missing")) ++missing;
        else ++unchecked;
    }
    okLabel_->setText(QString::number(ok));
    warningLabel_->setText(QString::number(warning));
    missingLabel_->setText(QString::number(missing));
    uncheckedLabel_->setText(QString::number(unchecked));
    if (missing > 0) {
        statusLabel_->setText(tr("发现 %1 项缺失，相关能力会被阻塞。").arg(missing));
    } else if (warning > 0) {
        statusLabel_->setText(blocked > 0
            ? tr("发现 %1 项警告（其中 %2 项硬件受限），可继续但需要关注。")
                .arg(warning).arg(blocked)
            : tr("发现 %1 项警告，可继续但需要关注。").arg(warning));
    } else if (unchecked > 0) {
        statusLabel_->setText(tr("尚有 %1 项未检测。").arg(unchecked));
    } else {
        statusLabel_->setText(tr("环境自检通过。"));
    }
}
