#include "Sidebar.h"

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

Sidebar::Sidebar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName(QStringLiteral("Sidebar"));
    setFixedWidth(210);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(16, 18, 16, 16);
    layout->setSpacing(14);

    auto* brand = new QLabel(QStringLiteral("AITrain Studio"));
    brand->setObjectName(QStringLiteral("BrandTitle"));
    auto* subtitle = new QLabel(tr("Vision Training Workbench"));
    subtitle->setObjectName(QStringLiteral("BrandSubtitle"));

    layout->addWidget(brand);
    layout->addWidget(subtitle);
    layout->addSpacing(8);

    itemsLayout_ = new QVBoxLayout;
    itemsLayout_->setSpacing(4);
    layout->addLayout(itemsLayout_);
    layout->addStretch();

    buttons_.setExclusive(true);
    connect(&buttons_, QOverload<int>::of(&QButtonGroup::buttonClicked), this, [this](int id) {
        if (auto* button = buttons_.button(id)) {
            emit pageRequested(id, button->text());
        }
    });
}

void Sidebar::addItem(const QString& text, int pageIndex)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("SidebarButton"));
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    buttons_.addButton(button, pageIndex);
    itemsLayout_->addWidget(button);
    if (pageIndex == 0) {
        button->setChecked(true);
    }
}

void Sidebar::addSection(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("SidebarSection"));
    itemsLayout_->addSpacing(10);
    itemsLayout_->addWidget(label);
}

void Sidebar::setCurrentIndex(int pageIndex)
{
    if (auto* button = buttons_.button(pageIndex)) {
        button->setChecked(true);
    }
}
