#include "aitrain/core/Deployment.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QLibrary>
#include <QProcessEnvironment>

#include <algorithm>

namespace aitrain {
namespace {

QString executableName(const QString& baseName)
{
#ifdef Q_OS_WIN
    return QStringLiteral("%1.exe").arg(baseName);
#else
    return baseName;
#endif
}

QString libraryFileName(const QString& libraryName)
{
    if (!QFileInfo(libraryName).suffix().isEmpty()) {
        return libraryName;
    }
#ifdef Q_OS_WIN
    return QStringLiteral("%1.dll").arg(libraryName);
#elif defined(Q_OS_MAC)
    return libraryName.startsWith(QStringLiteral("lib"))
        ? QStringLiteral("%1.dylib").arg(libraryName)
        : QStringLiteral("lib%1.dylib").arg(libraryName);
#else
    return libraryName.startsWith(QStringLiteral("lib"))
        ? QStringLiteral("%1.so").arg(libraryName)
        : QStringLiteral("lib%1.so").arg(libraryName);
#endif
}

bool samePath(const QString& left, const QString& right)
{
#ifdef Q_OS_WIN
    return left.compare(right, Qt::CaseInsensitive) == 0;
#else
    return left == right;
#endif
}

void appendUniquePath(QStringList* paths, const QString& path)
{
    if (!paths || path.trimmed().isEmpty()) {
        return;
    }
    const QString cleanPath = QDir::cleanPath(path.trimmed());
    const auto found = std::find_if(paths->cbegin(), paths->cend(), [&cleanPath](const QString& existing) {
        return samePath(existing, cleanPath);
    });
    if (found == paths->cend()) {
        paths->append(cleanPath);
    }
}

void appendRootCandidates(QStringList* paths, const QString& root)
{
    if (root.trimmed().isEmpty()) {
        return;
    }
    appendUniquePath(paths, root);
    appendUniquePath(paths, QDir(root).filePath(QStringLiteral("bin")));
    appendUniquePath(paths, QDir(root).filePath(QStringLiteral("lib")));
    appendUniquePath(paths, QDir(root).filePath(QStringLiteral("lib/x64")));
}

QStringList splitPathEnvironment(const QString& value)
{
#ifdef Q_OS_WIN
    return value.split(QLatin1Char(';'),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
        QString::SkipEmptyParts
#else
        Qt::SkipEmptyParts
#endif
    );
#else
    return value.split(QLatin1Char(':'),
#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
        QString::SkipEmptyParts
#else
        Qt::SkipEmptyParts
#endif
    );
#endif
}

QString findRuntimeLibraryFile(const QStringList& libraryNames, const QStringList& searchPaths)
{
    for (const QString& path : searchPaths) {
        const QDir dir(path);
        for (const QString& libraryName : libraryNames) {
            const QString candidate = dir.filePath(libraryFileName(libraryName));
            if (QFileInfo::exists(candidate) && QFileInfo(candidate).isFile()) {
                return QDir::cleanPath(QFileInfo(candidate).absoluteFilePath());
            }
        }
    }
    return {};
}

void prependRuntimeDirectoryForLoad(const QString& libraryPathOrName)
{
    const QFileInfo info(libraryPathOrName);
    if (!info.isAbsolute()) {
        return;
    }

    const QByteArray directory = QFile::encodeName(QDir::toNativeSeparators(info.absolutePath()));
    const QByteArray path = qgetenv("PATH");
#ifdef Q_OS_WIN
    const char separator = ';';
#else
    const char separator = ':';
#endif
    if (!path.split(separator).contains(directory)) {
        qputenv("PATH", directory + QByteArray(1, separator) + path);
    }
}

bool tryLoadLibrary(const QString& libraryPathOrName, QString* error)
{
    prependRuntimeDirectoryForLoad(libraryPathOrName);
    QLibrary library(libraryPathOrName);
    const bool loaded = library.load();
    if (loaded) {
        library.unload();
        return true;
    }
    if (error) {
        *error = library.errorString();
    }
    return false;
}

RuntimeDependencyCheck checkCudaRuntimeDependency(const QString& applicationDir)
{
    RuntimeDependencyCheck check = checkRuntimeDependency(
        QStringLiteral("CUDA Runtime"),
        QStringList() << QStringLiteral("cudart64_13") << QStringLiteral("cudart64_12") << QStringLiteral("cudart64_120") << QStringLiteral("cudart64_110"),
        QStringLiteral("TensorRT engine 和真实 CUDA 训练需要 CUDA Runtime DLL；可放入 runtimes/tensorrt、应用目录或 PATH。"),
        applicationDir);
    if (check.status != QStringLiteral("ok")) {
        return check;
    }

    prependRuntimeDirectoryForLoad(check.resolvedPath);
    QLibrary library(check.resolvedPath);
    if (!library.load()) {
        check.status = QStringLiteral("warning");
        check.message = QStringLiteral("CUDA Runtime DLL 可找到，但无法加载；请检查驱动和依赖 DLL。");
        check.details.insert(QStringLiteral("loadError"), library.errorString());
        return check;
    }

    using CudaFreeFn = int (*)(void*);
    using CudaGetErrorStringFn = const char* (*)(int);
    const auto cudaFree = reinterpret_cast<CudaFreeFn>(library.resolve("cudaFree"));
    const auto cudaGetErrorString = reinterpret_cast<CudaGetErrorStringFn>(library.resolve("cudaGetErrorString"));
    if (!cudaFree || !cudaGetErrorString) {
        check.status = QStringLiteral("warning");
        check.message = QStringLiteral("CUDA Runtime DLL 可加载，但缺少基础 CUDA runtime symbol。");
        return check;
    }

    const int cudaStatus = cudaFree(nullptr);
    if (cudaStatus != 0) {
        const char* cudaMessage = cudaGetErrorString(cudaStatus);
        check.status = QStringLiteral("warning");
        check.message = QStringLiteral("CUDA Runtime DLL 可加载，但 CUDA 初始化失败：%1。请检查 NVIDIA 驱动是否支持当前 CUDA runtime。")
            .arg(cudaMessage ? QString::fromUtf8(cudaMessage) : QStringLiteral("error %1").arg(cudaStatus));
        check.details.insert(QStringLiteral("cudaErrorCode"), cudaStatus);
        return check;
    }
    return check;
}

QJsonArray stringArray(const QStringList& values)
{
    QJsonArray array;
    for (const QString& value : values) {
        array.append(value);
    }
    return array;
}

} // namespace

QJsonObject PackagingLayout::toJson() const
{
    return QJsonObject{
        {QStringLiteral("rootPath"), rootPath},
        {QStringLiteral("appExecutablePath"), appExecutablePath},
        {QStringLiteral("workerExecutablePath"), workerExecutablePath},
        {QStringLiteral("pluginModelsDirectory"), pluginModelsDirectory},
        {QStringLiteral("runtimesDirectory"), runtimesDirectory},
        {QStringLiteral("onnxRuntimeDirectory"), onnxRuntimeDirectory},
        {QStringLiteral("tensorRtRuntimeDirectory"), tensorRtRuntimeDirectory},
        {QStringLiteral("examplesDirectory"), examplesDirectory},
        {QStringLiteral("docsDirectory"), docsDirectory}
    };
}

QJsonObject RuntimeDependencyCheck::toJson() const
{
    QJsonObject object;
    object.insert(QStringLiteral("name"), name);
    object.insert(QStringLiteral("status"), status);
    object.insert(QStringLiteral("message"), message);
    QJsonObject detailsObject = details;
    detailsObject.insert(QStringLiteral("libraryNames"), stringArray(libraryNames));
    detailsObject.insert(QStringLiteral("searchPaths"), stringArray(searchPaths));
    if (!resolvedPath.isEmpty()) {
        detailsObject.insert(QStringLiteral("resolvedPath"), resolvedPath);
    }
    object.insert(QStringLiteral("details"), detailsObject);
    return object;
}

PackagingLayout packagingLayoutForRoot(const QString& rootPath)
{
    const QString root = QDir::cleanPath(rootPath);
    const QDir rootDir(root);
    PackagingLayout layout;
    layout.rootPath = root;
    layout.appExecutablePath = rootDir.filePath(executableName(QStringLiteral("AITrainStudio")));
    layout.workerExecutablePath = rootDir.filePath(executableName(QStringLiteral("aitrain_worker")));
    layout.pluginModelsDirectory = rootDir.filePath(QStringLiteral("plugins/models"));
    layout.runtimesDirectory = rootDir.filePath(QStringLiteral("runtimes"));
    layout.onnxRuntimeDirectory = rootDir.filePath(QStringLiteral("runtimes/onnxruntime"));
    layout.tensorRtRuntimeDirectory = rootDir.filePath(QStringLiteral("runtimes/tensorrt"));
    layout.examplesDirectory = rootDir.filePath(QStringLiteral("examples"));
    layout.docsDirectory = rootDir.filePath(QStringLiteral("docs"));
    return layout;
}

QStringList runtimeSearchPaths(const QString& applicationDir)
{
    QString resolvedApplicationDir = applicationDir;
    if (resolvedApplicationDir.isEmpty() && QCoreApplication::instance()) {
        resolvedApplicationDir = QCoreApplication::applicationDirPath();
    }

    QStringList paths;
    if (!resolvedApplicationDir.isEmpty()) {
        appendUniquePath(&paths, resolvedApplicationDir);
        appendUniquePath(&paths, QDir(resolvedApplicationDir).filePath(QStringLiteral("runtimes/onnxruntime")));
        appendUniquePath(&paths, QDir(resolvedApplicationDir).filePath(QStringLiteral("runtimes/tensorrt")));
        appendUniquePath(&paths, QDir(resolvedApplicationDir).filePath(QStringLiteral("../runtimes/onnxruntime")));
        appendUniquePath(&paths, QDir(resolvedApplicationDir).filePath(QStringLiteral("../runtimes/tensorrt")));
    }

    const QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    const QStringList rootVariables = {
        QStringLiteral("CUDA_PATH"),
        QStringLiteral("CUDNN_ROOT"),
        QStringLiteral("TENSORRT_ROOT"),
        QStringLiteral("TRT_ROOT"),
        QStringLiteral("ONNXRUNTIME_ROOT"),
        QStringLiteral("ORT_ROOT"),
        QStringLiteral("LIBTORCH_ROOT")
    };
    for (const QString& variable : rootVariables) {
        appendRootCandidates(&paths, environment.value(variable));
    }

    for (const QString& path : splitPathEnvironment(environment.value(QStringLiteral("PATH")))) {
        appendUniquePath(&paths, path);
    }
    return paths;
}

RuntimeDependencyCheck checkRuntimeDependency(
    const QString& name,
    const QStringList& libraryNames,
    const QString& missingMessage,
    const QString& applicationDir)
{
    RuntimeDependencyCheck check;
    check.name = name;
    check.libraryNames = libraryNames;
    check.searchPaths = runtimeSearchPaths(applicationDir);

    const QString resolvedPath = findRuntimeLibraryFile(libraryNames, check.searchPaths);
    QString loadError;
    if (!resolvedPath.isEmpty()) {
        check.resolvedPath = resolvedPath;
        if (tryLoadLibrary(resolvedPath, &loadError)) {
            check.status = QStringLiteral("ok");
            check.message = QStringLiteral("可加载 %1：%2。").arg(name, QDir::toNativeSeparators(resolvedPath));
            return check;
        }

        check.status = QStringLiteral("warning");
        check.message = QStringLiteral("找到 %1，但无法加载；请检查依赖 DLL。").arg(name);
        check.details.insert(QStringLiteral("loadError"), loadError);
        return check;
    }

    for (const QString& libraryName : libraryNames) {
        if (tryLoadLibrary(libraryName, &loadError)) {
            check.status = QStringLiteral("ok");
            check.resolvedPath = libraryName;
            check.message = QStringLiteral("可通过系统路径加载 %1（%2）。").arg(name, libraryFileName(libraryName));
            return check;
        }
    }

    check.status = QStringLiteral("missing");
    check.message = QStringLiteral("无法加载 %1；%2").arg(name, missingMessage);
    if (!loadError.isEmpty()) {
        check.details.insert(QStringLiteral("loadError"), loadError);
    }
    return check;
}

QVector<RuntimeDependencyCheck> defaultRuntimeDependencyChecks(const QString& applicationDir)
{
    return {
        checkRuntimeDependency(
            QStringLiteral("CUDA Driver"),
            QStringList() << QStringLiteral("nvcuda"),
            QStringLiteral("请安装 NVIDIA 驱动，或确认驱动 DLL 在系统路径中。"),
            applicationDir),
        checkCudaRuntimeDependency(applicationDir),
        checkRuntimeDependency(
            QStringLiteral("cuDNN"),
            QStringList() << QStringLiteral("cudnn64_9") << QStringLiteral("cudnn64_8"),
            QStringLiteral("后续真实训练需要 cuDNN DLL；可放入 runtimes/tensorrt、应用目录或 PATH。"),
            applicationDir),
        checkRuntimeDependency(
            QStringLiteral("TensorRT"),
            QStringList() << QStringLiteral("nvinfer") << QStringLiteral("nvinfer_10") << QStringLiteral("nvinfer_8"),
            QStringLiteral("TensorRT engine 导出和推理仍是占位；真实接入前请配置 TensorRT bin 目录。"),
            applicationDir),
        checkRuntimeDependency(
            QStringLiteral("TensorRT Plugin"),
            QStringList() << QStringLiteral("nvinfer_plugin") << QStringLiteral("nvinfer_plugin_10") << QStringLiteral("nvinfer_plugin_8"),
            QStringLiteral("TensorRT plugin DLL 缺失；真实 engine 构建/加载前需要配置 TensorRT plugin 运行库。"),
            applicationDir),
        checkRuntimeDependency(
            QStringLiteral("TensorRT ONNX Parser"),
            QStringList() << QStringLiteral("nvonnxparser") << QStringLiteral("nvonnxparser_10") << QStringLiteral("nvonnxparser_8"),
            QStringLiteral("ONNX 转 TensorRT engine 需要 nvonnxparser DLL。"),
            applicationDir),
        checkRuntimeDependency(
            QStringLiteral("ONNX Runtime"),
            QStringList() << QStringLiteral("onnxruntime"),
            QStringLiteral("ONNX 推理需要 onnxruntime.dll；打包时应放入应用目录和 runtimes/onnxruntime。"),
            applicationDir),
        checkRuntimeDependency(
            QStringLiteral("LibTorch"),
            QStringList() << QStringLiteral("torch") << QStringLiteral("torch_cpu"),
            QStringLiteral("真实 LibTorch/CUDA 训练暂未接入；后续需要配置 LibTorch DLL。"),
            applicationDir)
    };
}

} // namespace aitrain
