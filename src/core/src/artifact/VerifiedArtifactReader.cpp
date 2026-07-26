#include "aitrain/artifact/VerifiedArtifactReader.h"

#include "aitrain/domain/ArtifactMemberPath.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <utility>

namespace aitrain {
namespace {

void fail(ArtifactReadError code, const QString& message,
    ArtifactReadError* readError, QString* error)
{
    if (readError) *readError = code;
    if (error) *error = message;
}

bool isWithinRoot(const QString& root, const QString& candidate)
{
    const QString canonicalRoot = QDir::cleanPath(QFileInfo(root).absoluteFilePath());
    const QString absoluteCandidate =
        QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
    return absoluteCandidate.startsWith(canonicalRoot + QLatin1Char('/'),
        Qt::CaseInsensitive);
}

} // namespace

VerifiedArtifactReader::VerifiedArtifactReader(QString committedArtifactRoot)
    : committedArtifactRoot_(QDir::cleanPath(std::move(committedArtifactRoot)))
{
}

bool VerifiedArtifactReader::verify(const ArtifactFileSnapshot& expected,
    VerifiedArtifactFile* result, ArtifactReadError* readError, QString* error) const
{
    if (readError) *readError = ArtifactReadError::None;
    if (error) error->clear();
    QString member;
    QString memberError;
    if (committedArtifactRoot_.isEmpty()
        || !normalizeArtifactMemberPath(expected.relativePath, &member, &memberError)) {
        fail(ArtifactReadError::InvalidMember,
            QStringLiteral("已提交 Artifact 相对路径无效：%1").arg(memberError),
            readError, error);
        return false;
    }
    const QString absolutePath =
        QDir(committedArtifactRoot_).filePath(member);
    const QFileInfo info(absolutePath);
    if (!isWithinRoot(committedArtifactRoot_, absolutePath)
        || !info.exists() || !info.isFile() || info.isSymLink()) {
        fail(ArtifactReadError::InvalidMember,
            QStringLiteral("已提交 Artifact 文件不存在、越界或为符号链接：%1")
                .arg(expected.relativePath),
            readError, error);
        return false;
    }
    if (info.size() != expected.byteCount) {
        fail(ArtifactReadError::IntegrityMismatch,
            QStringLiteral("已提交 Artifact 文件大小不匹配：%1").arg(expected.relativePath),
            readError, error);
        return false;
    }
    QFile file(info.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        fail(ArtifactReadError::IoError,
            QStringLiteral("无法读取已提交 Artifact 文件：%1").arg(expected.relativePath),
            readError, error);
        return false;
    }
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray block = file.read(1024 * 1024);
        if (block.isEmpty() && file.error() != QFileDevice::NoError) {
            fail(ArtifactReadError::IoError,
                QStringLiteral("读取已提交 Artifact 文件失败：%1").arg(expected.relativePath),
                readError, error);
            return false;
        }
        hash.addData(block);
    }
    if (QString::fromLatin1(hash.result().toHex()) != expected.sha256) {
        fail(ArtifactReadError::IntegrityMismatch,
            QStringLiteral("已提交 Artifact 文件 SHA-256 不匹配：%1")
                .arg(expected.relativePath),
            readError, error);
        return false;
    }
    if (result) {
        *result = {expected.relativePath, info.absoluteFilePath(),
            expected.sha256, expected.byteCount};
    }
    return true;
}

bool VerifiedArtifactReader::preview(const ArtifactFileSnapshot& expected,
    qint64 maxBytes, ArtifactFilePreview* result,
    ArtifactReadError* readError, QString* error) const
{
    if (maxBytes < 1) {
        fail(ArtifactReadError::TooLarge,
            QStringLiteral("TooLarge：预览大小上限必须为正数。"), readError, error);
        return false;
    }
    VerifiedArtifactFile verified;
    if (!verify(expected, &verified, readError, error)) return false;
    QFile file(verified.absolutePath);
    if (!file.open(QIODevice::ReadOnly)) {
        fail(ArtifactReadError::IoError,
            QStringLiteral("无法读取已验证 Artifact 文件。"), readError, error);
        return false;
    }
    if (result) {
        result->relativePath = expected.relativePath;
        result->sha256 = expected.sha256;
        result->byteCount = expected.byteCount;
        result->content = file.read(maxBytes);
        result->truncated = expected.byteCount > maxBytes;
    }
    return true;
}

} // namespace aitrain
