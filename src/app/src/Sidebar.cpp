#include "Sidebar.h"

#include <QLabel>
#include <QIcon>
#include <QPixmap>
#include <QPushButton>
#include <QSignalBlocker>
#include <QStyle>
#include <QVariant>
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>

Sidebar::Sidebar(QWidget* parent)
    : QFrame(parent)
{
    setObjectName(QStringLiteral("Sidebar"));
    setFixedWidth(210);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(16, 18, 16, 16);
    layout->setSpacing(14);

    auto* brandRow = new QWidget;
    brandRow->setObjectName(QStringLiteral("SidebarBrand"));
    auto* brandLayout = new QHBoxLayout(brandRow);
    brandLayout->setContentsMargins(0, 0, 0, 0);
    brandLayout->setSpacing(9);
    auto* brandIcon = new QLabel;
    brandIcon->setObjectName(QStringLiteral("BrandIcon"));
    brandIcon->setFixedSize(30, 30);
    brandIcon->setPixmap(QIcon(QStringLiteral(":/icons/app.ico")).pixmap(28, 28));
    auto* brandText = new QWidget;
    auto* brandTextLayout = new QVBoxLayout(brandText);
    brandTextLayout->setContentsMargins(0, 0, 0, 0);
    brandTextLayout->setSpacing(0);
    brandTitle_ = new QLabel(QStringLiteral("AITrain Studio"));
    brandTitle_->setObjectName(QStringLiteral("BrandTitle"));
    brandSubtitle_ = new QLabel(tr("本地训练工作台"));
    brandSubtitle_->setObjectName(QStringLiteral("BrandSubtitle"));
    brandTextLayout->addWidget(brandTitle_);
    brandTextLayout->addWidget(brandSubtitle_);
    brandLayout->addWidget(brandIcon);
    brandLayout->addWidget(brandText, 1);
    layout->addWidget(brandRow);
    layout->addSpacing(4);

    itemsLayout_ = new QVBoxLayout;
    itemsLayout_->setSpacing(4);
    layout->addLayout(itemsLayout_);
    layout->addStretch();

    auto* footer = new QFrame;
    footer->setObjectName(QStringLiteral("SidebarFooter"));
    auto* footerLayout = new QHBoxLayout(footer);
    footerLayout->setContentsMargins(0, 8, 0, 0);
    footerLayout->setSpacing(9);
    auto* avatar = new QLabel(QStringLiteral("LM"));
    avatar->setObjectName(QStringLiteral("SidebarAvatar"));
    avatar->setAlignment(Qt::AlignCenter);
    avatar->setFixedSize(30, 30);
    userText_ = new QLabel(QStringLiteral("Local Admin\n工作站 · 在线"));
    userText_->setObjectName(QStringLiteral("SidebarUserText"));
    footerLayout->addWidget(avatar);
    footerLayout->addWidget(userText_, 1);
    layout->addWidget(footer);

    buttons_.setExclusive(true);
}

void Sidebar::addItem(const QString& text, int pageIndex)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("SidebarButton"));
    button->setProperty("fullText", text);
    button->setCheckable(true);
    button->setCursor(Qt::PointingHandCursor);
    static const QStyle::StandardPixmap icons[] = {
        QStyle::SP_DesktopIcon,
        QStyle::SP_DirIcon,
        QStyle::SP_DriveHDIcon,
        QStyle::SP_MediaPlay,
        QStyle::SP_FileDialogDetailedView,
        QStyle::SP_DirLinkIcon,
        QStyle::SP_ArrowForward,
        QStyle::SP_ComputerIcon,
        QStyle::SP_FileDialogContentsView
    };
    if (pageIndex >= 0 && pageIndex < static_cast<int>(sizeof(icons) / sizeof(icons[0]))) {
        button->setIcon(style()->standardIcon(icons[pageIndex]));
        button->setIconSize(QSize(17, 17));
    }
    buttons_.addButton(button, pageIndex);
    connect(button, &QPushButton::toggled, this, [this, button, pageIndex](bool checked) {
        if (checked) {
            const QString fullText = button->property("fullText").toString();
            emit pageRequested(pageIndex, fullText.isEmpty() ? button->text() : fullText);
        }
    });
    itemsLayout_->addWidget(button);
    navigationButtons_.append(button);
    if (pageIndex == 0) {
        button->setChecked(true);
    }
}

void Sidebar::addSection(const QString& text)
{
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("SidebarSection"));
    sectionLabels_.append(label);
    itemsLayout_->addSpacing(10);
    itemsLayout_->addWidget(label);
}

void Sidebar::setCompact(bool compact)
{
    setFixedWidth(compact ? 72 : width() >= 210 ? width() : 200);
    if (brandTitle_) brandTitle_->setVisible(!compact);
    if (brandSubtitle_) brandSubtitle_->setVisible(!compact);
    if (userText_) userText_->setVisible(!compact);
    for (QLabel* label : sectionLabels_) label->setVisible(!compact);
    for (QPushButton* button : navigationButtons_) {
        button->setProperty("compact", compact);
        const QString fullText = button->property("fullText").toString();
        button->setText(compact ? QString() : fullText);
        button->setToolTip(compact ? fullText : QString());
        button->style()->unpolish(button);
        button->style()->polish(button);
    }
    if (auto* root = qobject_cast<QVBoxLayout*>(layout())) {
        root->setContentsMargins(compact ? 10 : 16, 14, compact ? 10 : 16, 12);
    }
}

void Sidebar::setCurrentIndex(int pageIndex)
{
    if (auto* button = buttons_.button(pageIndex)) {
        const QSignalBlocker blocker(button);
        button->setChecked(true);
    }
}
