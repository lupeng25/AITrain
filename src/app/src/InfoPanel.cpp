#include "InfoPanel.h"

#include <QLabel>
#include <QVBoxLayout>

InfoPanel::InfoPanel(const QString& title, QWidget* parent)
    : QFrame(parent)
{
    setObjectName(QStringLiteral("Panel"));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(14, 12, 14, 14);
    layout->setSpacing(10);

    auto* titleLabel = new QLabel(title);
    titleLabel->setObjectName(QStringLiteral("PanelTitle"));
    layout->addWidget(titleLabel);

    bodyLayout_ = new QVBoxLayout;
    bodyLayout_->setSpacing(10);
    layout->addLayout(bodyLayout_);
}

QVBoxLayout* InfoPanel::bodyLayout() const
{
    return bodyLayout_;
}

