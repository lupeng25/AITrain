#pragma once

#include <QLabel>

class StatusPill : public QLabel {
    Q_OBJECT

public:
    enum class Tone {
        Neutral,
        Success,
        Warning,
        Error,
        Info
    };

    explicit StatusPill(QWidget* parent = nullptr);
    void setStatus(const QString& text, Tone tone);
};

