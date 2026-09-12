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
#include <QResizeEvent>
#include <QTimer>

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
    brandSubtitle_->setProperty("fullText", brandSubtitle_->text());
    brandSubtitle_->setMinimumWidth(0);
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

    toolsLayout_ = new QVBoxLayout;
    toolsLayout_->setSpacing(4);
    layout->addLayout(toolsLayout_);

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

void Sidebar::addToolItem(const QString& text, int pageIndex)
{
    auto* button = new QPushButton(text);
    button->setObjectName(QStringLiteral("SidebarToolButton"));
    button->setProperty("fullText", text);
    button->setToolTip(text);
    button->setProperty("workspaceTool", true);
    button->setCursor(Qt::PointingHandCursor);
    connect(button, &QPushButton::clicked, this, [this, pageIndex, text]() {
        emit pageRequested(pageIndex, text);
    });
    toolsLayout_->addWidget(button);
}

void Sidebar::setCompact(bool compact)
{
    compact_ = compact;
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

void Sidebar::resizeEvent(QResizeEvent* event)
{
    QFrame::resizeEvent(event);
    QTimer::singleShot(0, this, [this]() {
        for (auto* button : findChildren<QPushButton*>()) {
            const QString full = button->property("fullText").toString();
            if (full.isEmpty()) continue;
            const QString visible = compact_ ? (button->icon().isNull() ? full.left(1) : QString())
                : button->fontMetrics().elidedText(full, Qt::ElideRight, qMax(1, button->width() - 28 - (button->icon().isNull() ? 0 : 23)));
            button->setText(visible); button->setToolTip(full); button->setAccessibleName(full);
        }
        if (brandSubtitle_) {
            const QString full = brandSubtitle_->property("fullText").toString();
            brandSubtitle_->setText(brandSubtitle_->fontMetrics().elidedText(full, Qt::ElideRight, brandSubtitle_->width()));
            brandSubtitle_->setToolTip(full);
        }
    });
}

void Sidebar::setCurrentIndex(int pageIndex)
{
    if (auto* button = buttons_.button(pageIndex)) {
        const QSignalBlocker blocker(button);
        button->setChecked(true);
    }
}
