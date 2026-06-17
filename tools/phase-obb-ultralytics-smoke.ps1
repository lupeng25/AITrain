param(
    [string]$WorkDir = ".deps\phase-obb-ultralytics-smoke",
    [string]$Python = "",
    [string]$WorkerExe = "build-vscode\bin\aitrain_worker.exe",
    [string]$DatasetPath = "",
    [string]$DatasetYaml = "DOTA8.yaml",
    [string]$Model = "yolo11n-obb.pt",
    [string]$Device = "cpu",
    [int]$Epochs = 1,
    [int]$ImageSize = 640,
    [int]$BatchSize = 2,
    [switch]$SkipDownload,
    [switch]$RequirePublicDataset,
    [switch]$SkipProductRuntime
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Resolve-Python {
    if ($Python) {
        $resolved = Resolve-RepoPath $Python
        if (!(Test-Path -LiteralPath $resolved)) {
            throw "Python executable was not found: $resolved"
        }
        return $resolved
    }
    $candidates = @(
        (Resolve-RepoPath ".deps\envs\yolo-cuda\Scripts\python.exe"),
        (Resolve-RepoPath ".deps\envs\yolo26\Scripts\python.exe"),
        (Resolve-RepoPath ".deps\envs\smp-gpu\Scripts\python.exe"),
        "python"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -eq "python" -or (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    throw "Python was not found. Pass -Python."
}

function Resolve-Worker {
    if ([string]::IsNullOrWhiteSpace($WorkerExe)) {
        return ""
    }
    $resolved = Resolve-RepoPath $WorkerExe
    if (!(Test-Path -LiteralPath $resolved)) {
        throw "aitrain_worker.exe was not found: $resolved"
    }
    return $resolved
}

function Write-Json {
    param([string]$Path, [object]$Value)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-Json {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Encoding UTF8 -Raw | ConvertFrom-Json
}

function Read-LastJsonLine {
    param([string]$Text)
    $lines = @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    for ($index = $lines.Count - 1; $index -ge 0; --$index) {
        $line = $lines[$index].Trim()
        if ($line.StartsWith("{") -and $line.EndsWith("}")) {
            try {
                return ($line | ConvertFrom-Json)
            } catch {
            }
        }
    }
    return $null
}

function Invoke-Logged {
    param(
        [string]$File,
        [string[]]$Arguments,
        [string]$LogPath,
        [switch]$AllowFailure
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $File @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $text = (@($output | ForEach-Object { [string]$_ }) -join [Environment]::NewLine)
    Set-Content -LiteralPath $LogPath -Value $text -Encoding UTF8
    foreach ($line in @($output)) {
        Write-Host $line
    }
    if ($exitCode -ne 0 -and !$AllowFailure) {
        throw "Command failed with exit code $exitCode`: $File $($Arguments -join ' ')"
    }
    return [pscustomobject]@{
        exitCode = $exitCode
        text = $text
        json = Read-LastJsonLine -Text $text
        logPath = $LogPath
    }
}

function New-GeneratedObbDataset {
    param(
        [string]$PythonExe,
        [string]$OutputPath,
        [string]$ScriptPath
    )
    $code = @'
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except Exception as exc:
    raise SystemExit(f"Pillow is required to generate OBB smoke data: {exc}")


def rotated_box(cx: float, cy: float, w: float, h: float, angle: float) -> list[tuple[float, float]]:
    ca, sa = math.cos(angle), math.sin(angle)
    points = []
    for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
        points.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return points


def write_split(root: Path, split: str, count: int, image_size: int) -> dict[str, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    instances = 0
    for index in range(count):
        image = Image.new("RGB", (image_size, image_size), (245, 245, 240))
        draw = ImageDraw.Draw(image)
        cx = image_size * (0.35 + 0.10 * (index % 3))
        cy = image_size * (0.38 + 0.08 * (index % 4))
        w = image_size * (0.28 + 0.02 * (index % 2))
        h = image_size * 0.12
        angle = math.radians((index * 17) % 120)
        class_id = index % 2
        points = rotated_box(cx, cy, w, h, angle)
        fill = (70, 130, 180) if class_id == 0 else (190, 90, 70)
        draw.polygon(points, fill=fill, outline=(20, 20, 20))
        draw.line(points + [points[0]], fill=(10, 10, 10), width=max(1, image_size // 128))
        image_path = image_dir / f"{split}_{index:03d}.jpg"
        image.save(image_path, quality=95)
        normalized = []
        for x, y in points:
            normalized.extend([max(0.0, min(1.0, x / image_size)), max(0.0, min(1.0, y / image_size))])
        label = " ".join([str(class_id), *[f"{value:.6f}" for value in normalized]])
        (label_dir / f"{split}_{index:03d}.txt").write_text(label + "\n", encoding="utf-8")
        instances += 1
    return {"images": count, "instances": instances}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        import shutil
        shutil.rmtree(root)
    stats = {
        "train": write_split(root, "train", 12, args.image_size),
        "val": write_split(root, "val", 4, args.image_size),
        "test": write_split(root, "test", 4, args.image_size),
    }
    (root / "data.yaml").write_text(
        "\n".join([
            f"path: \"{root.as_posix()}\"",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "task: obb",
            "nc: 2",
            "names:",
            "  0: widget",
            "  1: plate",
            "",
        ]),
        encoding="utf-8",
    )
    report = {"ok": True, "source": "generated_obb_smoke", "path": str(root), "stats": stats}
    (root / "dataset_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@
    Set-Content -LiteralPath $ScriptPath -Value $code -Encoding UTF8
    Invoke-Logged -File $PythonExe -Arguments @($ScriptPath, "--output", $OutputPath, "--image-size", [string]$ImageSize) -LogPath (Join-Path (Split-Path -Parent $ScriptPath) "generate_obb_dataset.log")
}

function Test-ObbDataset {
    param(
        [string]$PythonExe,
        [string]$DatasetRoot,
        [string]$ReportPath
    )
    $code = @'
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

def parse_inline_names(value: str):
    value = value.strip()
    if value.startswith("[") and "]" in value:
        inner = value[1:value.index("]")]
        return [item.strip().strip("\"'") for item in inner.split(",") if item.strip()]
    if value.startswith("{") and "}" in value:
        inner = value[1:value.index("}")]
        return [item for item in inner.split(",") if ":" in item]
    return []

def load_class_count(root: Path, issues):
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        issues.append({"severity": "error", "code": "missing_data_yaml", "path": str(data_yaml)})
        return None

    text = data_yaml.read_text(encoding="utf-8")
    nc_match = re.search(r"(?m)^\s*nc\s*:\s*(\d+)\s*(?:#.*)?$", text)
    if nc_match:
        return int(nc_match.group(1))

    names_count = 0
    in_names_block = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        names_match = re.match(r"^\s*names\s*:\s*(.*)$", line)
        if names_match:
            inline_names = parse_inline_names(names_match.group(1))
            if inline_names:
                return len(inline_names)
            in_names_block = True
            continue
        if not in_names_block:
            continue
        if re.match(r"^\S", line):
            break
        indexed_name = re.match(r"^\s*(\d+)\s*:", line)
        if indexed_name:
            names_count = max(names_count, int(indexed_name.group(1)) + 1)
            continue
        if re.match(r"^\s*-\s+", line):
            names_count += 1
    if names_count > 0:
        return names_count

    issues.append({"severity": "error", "code": "missing_class_names", "path": str(data_yaml)})
    return None

root = Path(sys.argv[1]).resolve()
issues = []
counts = {"images": 0, "labels": 0, "instances": 0}
class_count = load_class_count(root, issues)
for split in ("train", "val", "test"):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    for image in image_dir.glob("*"):
        if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            counts["images"] += 1
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                issues.append({"severity": "error", "code": "missing_label", "path": str(label)})
    for label in label_dir.glob("*.txt"):
        counts["labels"] += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                issues.append({"severity": "error", "code": "invalid_yolo_obb_row", "path": str(label), "line": line_number})
                continue
            try:
                class_id = int(parts[0])
                coords = [float(item) for item in parts[1:]]
            except Exception:
                issues.append({"severity": "error", "code": "invalid_number", "path": str(label), "line": line_number})
                continue
            if class_id < 0 or (class_count is not None and class_id >= class_count):
                issues.append({
                    "severity": "error",
                    "code": "class_id_out_of_range",
                    "path": str(label),
                    "line": line_number,
                    "classId": class_id,
                    "classCount": class_count,
                })
            if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in coords):
                issues.append({"severity": "error", "code": "coordinate_out_of_range", "path": str(label), "line": line_number})
            points = list(zip(coords[0::2], coords[1::2]))
            area = 0.0
            for index, (x1, y1) in enumerate(points):
                x2, y2 = points[(index + 1) % len(points)]
                area += x1 * y2 - x2 * y1
            if abs(area) * 0.5 <= 1.0e-8:
                issues.append({"severity": "error", "code": "obb_degenerate_quad", "path": str(label), "line": line_number})
            counts["instances"] += 1

ok = not any(item["severity"] == "error" for item in issues)
payload = {"ok": ok, "format": "yolo_obb", "path": str(root), "classCount": class_count, "counts": counts, "issues": issues}
Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
raise SystemExit(0 if ok else 2)
'@
    $scriptPath = Join-Path (Split-Path -Parent $ReportPath) "validate_obb_dataset.py"
    Set-Content -LiteralPath $scriptPath -Value $code -Encoding UTF8
    Invoke-Logged -File $PythonExe -Arguments @($scriptPath, $DatasetRoot, $ReportPath) -LogPath ($ReportPath + ".log")
}

function Repair-ObbDatasetCoordinates {
    param(
        [string]$PythonExe,
        [string]$DatasetRoot,
        [string]$ReportPath
    )
    $code = @'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
changed_files = 0
changed_instances = 0
changed_coordinates = 0
instances = 0
min_before = None
max_before = None

for label in sorted((root / "labels").rglob("*.txt")):
    original_lines = label.read_text(encoding="utf-8").splitlines()
    new_lines = []
    file_changed = False
    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        parts = stripped.split()
        if len(parts) != 9:
            new_lines.append(line)
            continue
        try:
            class_id = int(parts[0])
            coords = [float(item) for item in parts[1:]]
        except Exception:
            new_lines.append(line)
            continue
        if not all(math.isfinite(value) for value in coords):
            new_lines.append(line)
            continue
        instances += 1
        current_min = min(coords)
        current_max = max(coords)
        min_before = current_min if min_before is None else min(min_before, current_min)
        max_before = current_max if max_before is None else max(max_before, current_max)
        clipped = [min(1.0, max(0.0, value)) for value in coords]
        if clipped != coords:
            file_changed = True
            changed_instances += 1
            changed_coordinates += sum(1 for before, after in zip(coords, clipped) if before != after)
        values = [str(class_id), *[f"{value:.8g}" for value in clipped]]
        new_lines.append(" ".join(values))
    if file_changed:
        label.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        changed_files += 1

payload = {
    "ok": True,
    "operation": "clip_obb_coordinates_to_unit_range",
    "path": str(root),
    "instances": instances,
    "changedFiles": changed_files,
    "changedInstances": changed_instances,
    "changedCoordinates": changed_coordinates,
    "minBefore": min_before,
    "maxBefore": max_before,
}
Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
'@
    $scriptPath = Join-Path (Split-Path -Parent $ReportPath) "repair_obb_dataset.py"
    Set-Content -LiteralPath $scriptPath -Value $code -Encoding UTF8
    Invoke-Logged -File $PythonExe -Arguments @($scriptPath, $DatasetRoot, $ReportPath) -LogPath ($ReportPath + ".log")
}

function Convert-ObbWorkerSample {
    param(
        [string]$PythonExe,
        [string]$InputPath,
        [string]$OutputPath,
        [string]$ScriptPath
    )
    $code = @'
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
image.save(target)
payload = {
    "ok": True,
    "sourcePath": str(source),
    "outputPath": str(target),
    "format": "png",
    "width": image.width,
    "height": image.height,
}
print(json.dumps(payload, ensure_ascii=False))
'@
    Set-Content -LiteralPath $ScriptPath -Value $code -Encoding UTF8
    Invoke-Logged -File $PythonExe -Arguments @($ScriptPath, $InputPath, $OutputPath) -LogPath ($OutputPath + ".log")
}

$work = Resolve-RepoPath $WorkDir
New-Item -ItemType Directory -Force -Path $work | Out-Null
$py = Resolve-Python
$summaryPath = Join-Path $work "obb_ultralytics_smoke_summary.json"
$startedAt = [DateTime]::UtcNow
$datasetRoot = if ($DatasetPath) { Resolve-RepoPath $DatasetPath } else { Join-Path $work "dataset" }
$materialization = $null
$sanitization = $null

if ($DatasetPath) {
    if (!(Test-Path -LiteralPath $datasetRoot)) {
        throw "DatasetPath does not exist: $datasetRoot"
    }
    $materialization = [ordered]@{ ok = $true; source = "provided"; path = $datasetRoot }
} elseif (!$SkipDownload) {
    $materializeReport = Join-Path $work "materialize_obb_dataset.json"
    $materialize = Invoke-Logged -File $py -Arguments @(
        (Resolve-RepoPath "tools\materialize-ultralytics-dataset.py"),
        "--yaml", $DatasetYaml,
        "--destination", $datasetRoot,
        "--downloads", (Resolve-RepoPath ".deps\datasets\downloads"),
        "--materialized-root", (Resolve-RepoPath ".deps\datasets\materialized"),
        "--report", $materializeReport
    ) -LogPath (Join-Path $work "materialize_obb_dataset.log") -AllowFailure
    $materialization = Read-Json $materializeReport
    if ($materialize.exitCode -ne 0 -or $null -eq $materialization -or -not $materialization.ok) {
        if ($RequirePublicDataset) {
            throw "Public OBB dataset materialization failed. See $($materialize.logPath)"
        }
        $datasetRoot = Join-Path $work "generated_obb_dataset"
        $materialization = (New-GeneratedObbDataset -PythonExe $py -OutputPath $datasetRoot -ScriptPath (Join-Path $work "generate_obb_dataset.py")).json
    }
} else {
    $datasetRoot = Join-Path $work "generated_obb_dataset"
    $materialization = (New-GeneratedObbDataset -PythonExe $py -OutputPath $datasetRoot -ScriptPath (Join-Path $work "generate_obb_dataset.py")).json
}

if (!$DatasetPath) {
    $sanitizationReport = Join-Path $work "dataset_sanitization_report.json"
    $sanitization = Repair-ObbDatasetCoordinates -PythonExe $py -DatasetRoot $datasetRoot -ReportPath $sanitizationReport
}

$validationReport = Join-Path $work "dataset_validation_report.json"
$validation = Test-ObbDataset -PythonExe $py -DatasetRoot $datasetRoot -ReportPath $validationReport
if ($validation.exitCode -ne 0) {
    throw "OBB dataset validation failed. See $validationReport"
}

$runDir = Join-Path $work "train"
$trainRequestPath = Join-Path $work "train_request.json"
$trainRequest = [ordered]@{
    taskId = "obb-ultralytics-smoke-train"
    datasetPath = $datasetRoot
    outputPath = $runDir
    parameters = [ordered]@{
        trainingBackend = "ultralytics_yolo_obb"
        modelFamily = "yolo_obb"
        model = $Model
        modelPreset = $Model
        epochs = $Epochs
        batchSize = $BatchSize
        imageSize = $ImageSize
        device = $Device
        exportOnnx = $true
        runName = "aitrain-yolo-obb-smoke"
        ultralyticsExportArgs = [ordered]@{
            format = "onnx"
            imgsz = $ImageSize
            batch = 1
            device = $Device
        }
    }
}
Write-Json -Path $trainRequestPath -Value $trainRequest
$train = Invoke-Logged -File $py -Arguments @((Resolve-RepoPath "python_trainers\obb\ultralytics_trainer.py"), "--request", $trainRequestPath) -LogPath (Join-Path $work "train.log")
$trainingReportPath = Join-Path $runDir "ultralytics_training_report.json"
$trainingReport = Read-Json $trainingReportPath
if ($null -eq $trainingReport -or -not $trainingReport.ok) {
    throw "OBB training did not produce an ok training report: $trainingReportPath"
}
$bestPt = [string]$trainingReport.checkpointPath
$bestOnnx = [string]$trainingReport.onnxPath
if (!(Test-Path -LiteralPath $bestPt) -or !(Test-Path -LiteralPath $bestOnnx)) {
    throw "OBB training artifacts are missing. best.pt=$bestPt best.onnx=$bestOnnx"
}

$evalDir = Join-Path $work "official_val"
$evalRequestPath = Join-Path $work "eval_request.json"
$evalRequest = [ordered]@{
    taskId = "obb-ultralytics-smoke-val"
    modelPath = $bestPt
    datasetPath = $datasetRoot
    outputPath = $evalDir
    taskType = "obb_detection"
    options = [ordered]@{
        imageSize = $ImageSize
        batch = $BatchSize
        ultralyticsValArgs = [ordered]@{
            split = "val"
            device = $Device
            imgsz = $ImageSize
            batch = $BatchSize
        }
    }
}
Write-Json -Path $evalRequestPath -Value $evalRequest
$eval = Invoke-Logged -File $py -Arguments @((Resolve-RepoPath "python_trainers\yolo\ultralytics_evaluator.py"), "--request", $evalRequestPath) -LogPath (Join-Path $work "official_val.log")
$evalReportPath = Join-Path $evalDir "evaluation_report.json"
$evalReport = Read-Json $evalReportPath
if ($null -eq $evalReport -or -not $evalReport.ok) {
    throw "OBB official val did not produce an ok evaluation report: $evalReportPath"
}

$workerSmoke = $null
$sampleImage = Get-ChildItem -LiteralPath (Join-Path $datasetRoot "images\test") -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $sampleImage) {
    $sampleImage = Get-ChildItem -LiteralPath (Join-Path $datasetRoot "images\val") -File -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($null -eq $sampleImage) {
    $sampleImage = Get-ChildItem -LiteralPath (Join-Path $datasetRoot "images\train") -File -ErrorAction SilentlyContinue | Select-Object -First 1
}
if ($null -eq $sampleImage) {
    throw "No test/val/train sample image was found under $datasetRoot"
}
if (!$SkipProductRuntime) {
    $worker = Resolve-Worker
    $workerSmokeDir = Join-Path $work "worker_obb_onnx_smoke"
    $convertedSample = Join-Path $work "worker_sample.png"
    $sampleConversion = Convert-ObbWorkerSample -PythonExe $py -InputPath $sampleImage.FullName -OutputPath $convertedSample -ScriptPath (Join-Path $work "convert_obb_worker_sample.py")
    $workerSampleImage = if ($sampleConversion.json -and $sampleConversion.json.outputPath) { [string]$sampleConversion.json.outputPath } else { $convertedSample }
    $workerSmoke = Invoke-Logged -File $worker -Arguments @("--obb-onnx-smoke", $bestOnnx, "--image", $workerSampleImage, "--output", $workerSmokeDir) -LogPath (Join-Path $work "worker_obb_onnx_smoke.log")
}

$summary = [ordered]@{
    ok = ($validation.exitCode -eq 0 -and $train.exitCode -eq 0 -and $eval.exitCode -eq 0 -and ($SkipProductRuntime -or ($workerSmoke -and $workerSmoke.exitCode -eq 0)))
    status = "passed"
    phase = "obb_ultralytics_smoke"
    startedAt = $startedAt.ToString("o")
    finishedAt = ([DateTime]::UtcNow).ToString("o")
    workDir = $work
    dataset = [ordered]@{
        path = $datasetRoot
        materialization = $materialization
        sanitizationReportPath = if ($sanitization) { [string]$sanitizationReport } else { "" }
        sanitization = if ($sanitization) { $sanitization.json } else { $null }
        validationReportPath = $validationReport
    }
    parameters = [ordered]@{
        model = $Model
        epochs = $Epochs
        imageSize = $ImageSize
        batchSize = $BatchSize
        device = $Device
    }
    artifacts = [ordered]@{
        bestPt = $bestPt
        bestOnnx = $bestOnnx
        trainingReportPath = $trainingReportPath
        evaluationReportPath = $evalReportPath
        workerSmokeSummaryPath = if ($workerSmoke -and $workerSmoke.json) { [string]$workerSmoke.json.summaryPath } else { "" }
    }
    metrics = $evalReport.metrics
    productRuntime = if ($SkipProductRuntime) { [ordered]@{ skipped = $true } } else { $workerSmoke.json }
    note = "OBB smoke uses public DOTA materialization when available, otherwise generated OBB workflow data; it is not customer-domain industrial accuracy evidence."
}
if (-not $summary.ok) {
    $summary["status"] = "failed"
}
Write-Json -Path $summaryPath -Value $summary
$summary | ConvertTo-Json -Depth 100
if (-not $summary["ok"]) {
    exit 1
}
