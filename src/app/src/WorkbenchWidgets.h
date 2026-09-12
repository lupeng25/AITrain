#pragma once
#include "WorkbenchTranslation.h"

#include <QAbstractItemView>
#include <QFrame>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QPixmap>
#include <QResizeEvent>
#include <QPainter>
#include <QKeyEvent>
#include <QLineEdit>
#include <QTimer>
#include <functional>

namespace aitrain_app {

inline QLineEdit* catalogSearchField(const QString& name, const QString& hint)
{
    auto* field = new QLineEdit;
    field->setObjectName(name); field->setPlaceholderText(hint);
    field->setClearButtonEnabled(true); field->setMaxLength(200); field->setMinimumWidth(0);
    return field;
}

inline void bindCatalogSearch(QLineEdit* field, QObject* owner, std::function<void(QString)> apply)
{
    auto* debounce = new QTimer(owner); debounce->setSingleShot(true); debounce->setInterval(250);
    QObject::connect(field, &QLineEdit::textChanged, debounce, [debounce]() { debounce->start(); });
    QObject::connect(debounce, &QTimer::timeout, owner, [field, apply]() { apply(field->text().trimmed()); });
    QObject::connect(field, &QLineEdit::returnPressed, owner, [debounce, field, apply]() { debounce->stop(); apply(field->text().trimmed()); });
}

class ElidingLabel : public QLabel {
public:
    using QLabel::QLabel;
protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter painter(this);
        painter.setPen(palette().color(QPalette::WindowText));
        painter.drawText(contentsRect(), Qt::AlignLeft | Qt::AlignVCenter,
            fontMetrics().elidedText(text(), Qt::ElideRight, contentsRect().width()));
    }
};

class ImagePreviewLabel : public QLabel {
public:
    explicit ImagePreviewLabel(QWidget* parent = nullptr) : QLabel(parent)
    {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(0, 0);
        setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        setWordWrap(true);
    }
    void setImage(const QPixmap& image) { image_ = image; refreshImage(); }
    void setText(const QString& text) { image_ = {}; QLabel::setText(text); }
    void clear() { image_ = {}; QLabel::clear(); }
protected:
    void resizeEvent(QResizeEvent* event) override { QLabel::resizeEvent(event); refreshImage(); }
private:
    void refreshImage()
    {
        if (!image_.isNull()) QLabel::setPixmap(image_.scaled(contentsRect().size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    QPixmap image_;
};

// 单层模式切换：目录、对象详情或当前操作只显示一个，不叠加标签和整页滚动。
class WorkspaceViewHost : public QWidget {
public:
    explicit WorkspaceViewHost(QWidget* parent = nullptr) : QWidget(parent)
    {
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(20, 0, 20, 20);
        root->setSpacing(12);
        toolbar = new QHBoxLayout;
        toolbar->setSpacing(10);
        backButton = new QPushButton(aitrain_app::workbenchText(QStringLiteral("返回目录")));
        backButton->setObjectName(QStringLiteral("WorkspaceBackButton"));
        modeTitle = new QLabel;
        modeTitle->setObjectName(QStringLiteral("WorkspaceModeTitle"));
        modeTitle->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        toolbar->addWidget(backButton);
        toolbar->addWidget(modeTitle, 1);
        root->addLayout(toolbar);
        views = new QStackedWidget;
        views->setObjectName(QStringLiteral("WorkspaceModeStack"));
        root->addWidget(views, 1);
        backButton->hide();
        connect(backButton, &QPushButton::clicked, this, [this]() {
            setMode(returnMode_);
        });
    }

    QVBoxLayout* addMode(const QString& title)
    {
        auto* page = new QWidget;
        page->setProperty("modeTitle", title);
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(12);
        views->addWidget(page);
        return layout;
    }

    void setMode(int index, int returnMode = 0)
    {
        if (index < 0 || index >= views->count()) return;
        returnMode_ = returnMode;
        views->setCurrentIndex(index);
        modeTitle->setText(views->widget(index)->property("modeTitle").toString());
        backButton->setVisible(index != 0);
        backButton->setText(returnMode == 0 ? aitrain_app::workbenchText(QStringLiteral("返回目录")) : aitrain_app::workbenchText(QStringLiteral("返回详情")));
    }

    QStackedWidget* views = nullptr;
    QHBoxLayout* toolbar = nullptr;
    QPushButton* backButton = nullptr;
    QLabel* modeTitle = nullptr;
protected:
    void keyPressEvent(QKeyEvent* event) override
    {
        if (event->key() == Qt::Key_Escape && backButton->isVisible()) {
            backButton->click();
            event->accept();
            return;
        }
        QWidget::keyPressEvent(event);
    }
private:
    int returnMode_ = 0;
};

inline QTableWidget* workbenchTable(const QStringList& headers)
{
    auto* table = new QTableWidget(0, headers.size());
    table->setHorizontalHeaderLabels(headers);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setShowGrid(false);
    table->setWordWrap(false);
    table->setMinimumSize(0, 0);
    table->verticalHeader()->hide();
    table->verticalHeader()->setDefaultSectionSize(40);
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    return table;
}

inline QLabel* workbenchHint(const QString& text = {})
{
    auto* label = new QLabel(text);
    label->setTextFormat(Qt::PlainText);
    label->setObjectName(QStringLiteral("MutedText"));
    label->setWordWrap(true);
    label->setMinimumWidth(0);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    return label;
}

inline QPushButton* workbenchButton(const QString& text, const QString& name = {}, bool primary = false)
{
    auto* button = new QPushButton(text);
    button->setObjectName(name.isEmpty() ? (primary ? QStringLiteral("PrimaryButton") : QStringLiteral("WorkbenchButton")) : name);
    button->setProperty("primaryAction", primary);
    button->setCursor(Qt::PointingHandCursor);
    return button;
}

} // namespace aitrain_app
