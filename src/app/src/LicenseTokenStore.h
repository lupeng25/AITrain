#pragma once

#include <QString>

namespace aitrain_app {

class LicenseTokenStore
{
public:
    QString read() const;
    void write(const QString& token) const;
};

} // namespace aitrain_app
