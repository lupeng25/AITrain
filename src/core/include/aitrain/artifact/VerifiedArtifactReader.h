#pragma once

#include "aitrain/storage/ProjectStore.h"

#include <QByteArray>
#include <QString>
#include <QVector>

namespace aitrain {

enum class ArtifactReadError {
    None,
    InvalidMember,
    IntegrityMismatch,
    TooLarge,
    IoError
};

struct VerifiedArtifactFile final {
    QString relativePath;
    QString absolutePath;
    QString sha256;
    qint64 byteCount = 0;
};

struct ArtifactFilePreview final {
    QString relativePath;
    QString sha256;
    qint64 byteCount = 0;
    QByteArray content;
    bool truncated = false;
};

struct VerifiedArtifactDirectory final {
    ArtifactId artifactId;
    QString absolutePath;
    QVector<VerifiedArtifactFile> files;
};

class VerifiedArtifactReader final {
public:
    explicit VerifiedArtifactReader(QString committedArtifactRoot);

    bool verify(const ArtifactFileSnapshot& expected,
        VerifiedArtifactFile* result,
        ArtifactReadError* readError = nullptr,
        QString* error = nullptr) const;
    bool preview(const ArtifactFileSnapshot& expected,
        qint64 maxBytes,
        ArtifactFilePreview* result,
        ArtifactReadError* readError = nullptr,
        QString* error = nullptr) const;
    bool verifyInventory(const ArtifactSnapshot& artifact,
        VerifiedArtifactDirectory* result,
        ArtifactReadError* readError = nullptr,
        QString* error = nullptr) const;

private:
    QString committedArtifactRoot_;
};

} // namespace aitrain
