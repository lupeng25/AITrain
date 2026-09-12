#include "WorkbenchTranslation.h"
#include "EnvironmentPage.h"

#include "InfoPanel.h"
#include "WorkbenchWidgets.h"
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
    if (deliveryEvidencePage) { deliveryEvidencePage->setParent(this); deliveryEvidencePage->hide(); }
    auto* layout = new QVBoxLayout(this); layout->setContentsMargins(20, 0, 20, 20); layout->setSpacing(12);
    auto* actions = new QHBoxLayout;
    statusLabel_ = workbenchHint(aitrain_app::workbenchText(QStringLiteral("尚未执行环境自检。"))); statusLabel_->setObjectName(QStringLiteral("WorkspaceToolbarStatus"));
    auto* diagnostics = workbenchButton(aitrain_app::workbenchText(QStringLiteral("诊断包"))); auto* run = workbenchButton(aitrain_app::workbenchText(QStringLiteral("检查环境")), QStringLiteral("EnvironmentRun"), true);
    actions->addWidget(statusLabel_, 1); actions->addWidget(diagnostics); actions->addWidget(run); layout->addLayout(actions);
    connect(run, &QPushButton::clicked, this, &EnvironmentWorkspacePage::runRequested);
    connect(diagnostics, &QPushButton::clicked, this, &EnvironmentWorkspacePage::diagnosticsRequested);
    auto* summary = new QHBoxLayout;
    const auto addSummary = [summary](const QString& caption) { summary->addWidget(new QLabel(caption)); auto* value = new QLabel(aitrain_app::workbenchText(QStringLiteral("待检查"))); summary->addWidget(value); summary->addSpacing(20); return value; };
    okLabel_ = addSummary(aitrain_app::workbenchText(QStringLiteral("通过"))); warningLabel_ = addSummary(aitrain_app::workbenchText(QStringLiteral("警告"))); missingLabel_ = addSummary(aitrain_app::workbenchText(QStringLiteral("缺失"))); uncheckedLabel_ = addSummary(aitrain_app::workbenchText(QStringLiteral("未检查"))); summary->addStretch(); layout->addLayout(summary);
    table_ = workbenchTable({aitrain_app::workbenchText(QStringLiteral("检查项 / 能力")), aitrain_app::workbenchText(QStringLiteral("状态")), aitrain_app::workbenchText(QStringLiteral("说明与处理建议"))}); table_->setObjectName(QStringLiteral("EnvironmentTable"));
    table_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents); table_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    layout->addWidget(table_, 1); layout->addWidget(workbenchHint(aitrain_app::workbenchText(QStringLiteral("结果来自最近一次已提交的环境检查；修改环境配置后请重新检查。"))));
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
