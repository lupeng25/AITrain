#!/usr/bin/env python3
"""Serve SMP Oxford Pets quality-matrix progress for a work directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PRESETS = [
    "smp_unet_resnet34",
    "smp_unetplusplus_resnet34",
    "smp_fpn_resnet34",
    "smp_deeplabv3plus_resnet50",
    "smp_segformer_mit_b0",
]


def read_text(path: Path, limit: int = 30000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace")


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - diagnostic server
        return {"error": str(exc), "path": str(path)}


def is_pid_running(pid: object) -> bool:
    text = str(pid or "").strip()
    if not text.isdigit():
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {text}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return text in result.stdout
    except Exception:
        return False


def gpu_status() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip()}
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 6:
        return {"available": True, "raw": result.stdout.strip()}
    return {
        "available": True,
        "name": parts[0],
        "utilizationGpu": parts[1],
        "memoryUsedMiB": parts[2],
        "memoryTotalMiB": parts[3],
        "temperatureC": parts[4],
        "powerW": parts[5],
    }


def latest_stage(stdout_tail: str) -> dict:
    stage_lines = [
        line.strip()
        for line in stdout_tail.splitlines()
        if line.strip().startswith("SMP Oxford Pets matrix:")
    ]
    if not stage_lines:
        return {}
    line = stage_lines[-1]
    payload = {"line": line}
    for preset in PRESETS:
        if preset in line:
            payload["preset"] = preset
            break
    if "train " in line:
        payload["stage"] = "train"
    elif "evaluate " in line:
        payload["stage"] = "evaluate"
        if "split=val" in line:
            payload["split"] = "val"
        elif "split=test" in line:
            payload["split"] = "test"
    elif "product runtime" in line:
        payload["stage"] = "product_runtime"
    elif "materialize" in line:
        payload["stage"] = "dataset_materialization"
    elif "quality matrix passed" in line:
        payload["stage"] = "completed"
    return payload


def metric(metrics: object, name: str):
    if isinstance(metrics, dict):
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return value
    return None


def row_status(root: Path, preset: str) -> dict:
    run_dir = root / "runs" / preset
    log_dir = root / "logs"
    eval_dir = root / "evaluations" / preset
    product_dir = root / "product-runtime" / preset
    train_report = read_json(run_dir / "smp_training_report.json")
    sidecar = read_json(run_dir / "semantic_segmentation_sidecar.json")
    val_report = read_json(eval_dir / "val" / "evaluation_report.json")
    test_report = read_json(eval_dir / "test" / "evaluation_report.json")
    product_smokes: list[dict] = []
    if product_dir.exists():
        for smoke_path in sorted(product_dir.glob("*/smp_semantic_onnx_smoke_summary.json")):
            smoke = read_json(smoke_path)
            if isinstance(smoke, dict):
                product_smokes.append(
                    {
                        "sample": smoke_path.parent.name,
                        "ok": bool(smoke.get("ok")),
                        "overlayPath": smoke.get("overlayPath") or "",
                        "p95Ms": metric((smoke.get("benchmark") or {}).get("latency"), "p95Ms"),
                    }
                )
    p95_values = [item["p95Ms"] for item in product_smokes if isinstance(item.get("p95Ms"), (int, float))]
    ok_smokes = [item for item in product_smokes if item.get("ok")]
    status = "pending"
    if isinstance(test_report, dict) and test_report.get("ok") and len(ok_smokes) > 0:
        status = "passed"
    elif isinstance(train_report, dict) and train_report.get("ok"):
        status = "trained"
    elif (run_dir / "best.onnx").exists():
        status = "exported"
    elif (run_dir / "best.pt").exists():
        status = "training"
    elif (log_dir / f"{preset}-train.log").exists():
        status = "training_done"

    train_metrics = train_report.get("metrics") if isinstance(train_report, dict) else {}
    val_metrics = (val_report or {}).get("metrics") if isinstance(val_report, dict) else {}
    test_metrics = (test_report or {}).get("metrics") if isinstance(test_report, dict) else {}
    return {
        "preset": preset,
        "status": status,
        "architecture": (sidecar or {}).get("architecture") if isinstance(sidecar, dict) else "",
        "encoder": (sidecar or {}).get("encoderName") if isinstance(sidecar, dict) else "",
        "trainLoss": metric(train_metrics, "loss"),
        "trainMIoU": metric(train_metrics, "mIoU"),
        "valMIoU": metric(val_metrics, "mIoU"),
        "testMIoU": metric(test_metrics, "mIoU"),
        "testMeanDice": metric(test_metrics, "meanDice"),
        "testPixelAccuracy": metric(test_metrics, "pixelAccuracy"),
        "productSmokePassed": len(ok_smokes),
        "productSmokeTotal": len(product_smokes),
        "p95Ms": sum(p95_values) / len(p95_values) if p95_values else None,
        "checkpointExists": (run_dir / "best.pt").exists(),
        "bestOnnxExists": (run_dir / "best.onnx").exists(),
        "trainingReportPath": str(run_dir / "smp_training_report.json"),
        "testEvaluationPath": str(eval_dir / "test" / "evaluation_report.json"),
    }


def status_payload(root: Path, stdout_path: Path, stderr_path: Path, process_meta_path: Path | None) -> dict:
    summary = read_json(root / "smp_oxford_pets_quality_matrix_summary.json")
    dataset_manifest = read_json(root / "semantic_mask_oxford_pets" / "dataset_manifest.json")
    process_meta = read_json(process_meta_path) if process_meta_path else None
    pid = process_meta.get("id") if isinstance(process_meta, dict) else ""
    stdout_tail = read_text(stdout_path)
    stderr_tail = read_text(stderr_path)
    rows = [row_status(root, preset) for preset in PRESETS]
    finished = [row for row in rows if row["status"] in {"passed", "trained", "exported", "training_done"}]
    passed = [row for row in rows if row["status"] == "passed"]
    ranked = sorted(
        [row for row in rows if isinstance(row.get("testMIoU"), (int, float))],
        key=lambda item: (
            item.get("testMIoU") or 0,
            item.get("testMeanDice") or 0,
            item.get("testPixelAccuracy") or 0,
            -1.0 * (item.get("p95Ms") or 1e12),
        ),
        reverse=True,
    )
    return {
        "root": str(root),
        "pid": pid,
        "running": is_pid_running(pid),
        "summary": summary or {},
        "datasetManifest": dataset_manifest or {},
        "gpu": gpu_status(),
        "latestStage": latest_stage(stdout_tail),
        "rowCount": len(rows),
        "activeOrFinishedRows": len(finished),
        "passedRows": len(passed),
        "rows": rows,
        "rankedRows": ranked,
        "stdoutTail": stdout_tail,
        "stderrTail": stderr_tail,
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SMP Oxford Pets Quality Matrix</title>
  <style>
    :root { font-family: "Segoe UI", Arial, sans-serif; color-scheme: dark; }
    body { margin: 0; color: #e9eef5; background: #11161c; }
    header { position: sticky; top: 0; z-index: 2; padding: 16px 20px; background: #17202a; border-bottom: 1px solid #293745; }
    h1 { margin: 0 0 6px; font-size: 21px; letter-spacing: 0; }
    .sub { color: #9fb1c4; font-size: 13px; }
    main { padding: 18px 20px; display: grid; gap: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
    .panel { background: #151d25; border: 1px solid #2b3947; border-radius: 8px; padding: 14px; }
    .label { color: #95a9bd; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { margin-top: 6px; font-size: 24px; font-weight: 650; }
    .small { color: #aebdca; font-size: 12px; margin-top: 4px; }
    .ok { color: #59d98e; } .warn { color: #f4ca64; } .bad { color: #ff7979; }
    .bar { height: 9px; background: #273542; border-radius: 999px; overflow: hidden; margin-top: 10px; }
    .bar > div { height: 100%; background: linear-gradient(90deg, #52d273, #54a7ff); width: 0%; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid #2b3947; text-align: left; font-size: 13px; vertical-align: top; }
    th { color: #95a9bd; font-weight: 600; }
    pre { margin: 8px 0 0; white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto; background: #0c1116; border: 1px solid #2b3947; border-radius: 6px; padding: 12px; }
  </style>
</head>
<body>
<header>
  <h1>SMP Oxford Pets Quality Matrix</h1>
  <div class="sub" id="subtitle">Loading...</div>
</header>
<main>
  <section class="grid">
    <div class="panel"><div class="label">进程</div><div class="value" id="running">-</div><div class="small" id="pid"></div></div>
    <div class="panel"><div class="label">阶段</div><div class="value" id="stage">-</div><div class="small" id="stageLine"></div></div>
    <div class="panel"><div class="label">行进度</div><div class="value" id="rowsDone">-</div><div class="bar"><div id="progressBar"></div></div></div>
    <div class="panel"><div class="label">GPU</div><div class="value" id="gpu">-</div><div class="small" id="gpuDetail"></div></div>
  </section>
  <section class="grid">
    <div class="panel"><div class="label">数据集</div><div class="value" id="dataset">-</div><div class="small" id="datasetDetail"></div></div>
    <div class="panel"><div class="label">当前最佳</div><div class="value" id="best">-</div><div class="small" id="bestDetail"></div></div>
    <div class="panel"><div class="label">Summary</div><div class="value" id="summary">-</div><div class="small" id="summaryDetail"></div></div>
  </section>
  <section class="panel">
    <div class="label">模型行</div>
    <table>
      <thead><tr><th>Preset</th><th>Status</th><th>Encoder</th><th>Train mIoU</th><th>Val mIoU</th><th>Test mIoU</th><th>Dice</th><th>Smoke</th><th>p95Ms</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
  <section class="panel">
    <div class="label">排名</div>
    <table>
      <thead><tr><th>#</th><th>Preset</th><th>Test mIoU</th><th>Mean Dice</th><th>Pixel Acc</th><th>p95Ms</th></tr></thead>
      <tbody id="ranking"></tbody>
    </table>
  </section>
  <section class="panel"><div class="label">stdout tail</div><pre id="stdout"></pre></section>
  <section class="panel"><div class="label">stderr tail</div><pre id="stderr"></pre></section>
</main>
<script>
function fmt(v, digits = 4) {
  return typeof v === 'number' && isFinite(v) ? v.toFixed(digits) : '';
}
function clsForStatus(status) {
  if (status === 'passed') return 'ok';
  if (status === 'pending') return 'warn';
  return '';
}
async function refresh() {
  const res = await fetch('/api/status?ts=' + Date.now());
  const data = await res.json();
  const summaryStatus = data.summary?.status || (data.running ? 'running' : 'pending');
  const dataset = data.datasetManifest || {};
  const stage = data.latestStage || {};
  const gpu = data.gpu || {};
  const rows = data.rows || [];
  const ranked = data.rankedRows || [];
  document.getElementById('subtitle').textContent = `${data.root}`;
  document.getElementById('running').textContent = data.running ? 'running' : 'stopped';
  document.getElementById('running').className = 'value ' + (data.running ? 'ok' : 'warn');
  document.getElementById('pid').textContent = `PID ${data.pid || '-'}`;
  document.getElementById('stage').textContent = stage.stage || '-';
  document.getElementById('stage').className = 'value ' + (stage.stage === 'completed' ? 'ok' : 'warn');
  document.getElementById('stageLine').textContent = stage.line || '';
  document.getElementById('rowsDone').textContent = `${data.passedRows}/${data.rowCount}`;
  document.getElementById('progressBar').style.width = `${Math.round((data.activeOrFinishedRows || 0) * 100 / Math.max(data.rowCount || 1, 1))}%`;
  document.getElementById('gpu').textContent = gpu.available ? `${gpu.utilizationGpu || '-'}%` : '-';
  document.getElementById('gpuDetail').textContent = gpu.available ? `${gpu.name || ''} | ${gpu.memoryUsedMiB || '-'} / ${gpu.memoryTotalMiB || '-'} MiB | ${gpu.temperatureC || '-'} C | ${gpu.powerW || '-'} W` : (gpu.error || '');
  document.getElementById('dataset').textContent = dataset.totalSamples ? `${dataset.totalSamples} samples` : '-';
  document.getElementById('datasetDetail').textContent = dataset.imageSource ? `${dataset.imageSource}; train ${dataset.splitStats?.train?.sampleCount || '-'}, val ${dataset.splitStats?.val?.sampleCount || '-'}, test ${dataset.splitStats?.test?.sampleCount || '-'}` : '';
  document.getElementById('best').textContent = ranked[0]?.preset || data.summary?.bestPreset || '-';
  document.getElementById('bestDetail').textContent = ranked[0] ? `test mIoU ${fmt(ranked[0].testMIoU, 6)}, Dice ${fmt(ranked[0].testMeanDice, 6)}` : '';
  document.getElementById('summary').textContent = summaryStatus;
  document.getElementById('summary').className = 'value ' + (summaryStatus === 'passed' ? 'ok' : summaryStatus === 'failed' ? 'bad' : 'warn');
  document.getElementById('summaryDetail').textContent = data.summary?.reports?.summary || '';
  document.getElementById('rows').innerHTML = rows.map(r => `<tr>
    <td>${r.preset}</td><td class="${clsForStatus(r.status)}">${r.status}</td><td>${r.encoder || ''}</td>
    <td>${fmt(r.trainMIoU)}</td><td>${fmt(r.valMIoU)}</td><td>${fmt(r.testMIoU)}</td>
    <td>${fmt(r.testMeanDice)}</td><td>${r.productSmokePassed}/${r.productSmokeTotal}</td><td>${fmt(r.p95Ms, 2)}</td>
  </tr>`).join('');
  document.getElementById('ranking').innerHTML = ranked.map((r, i) => `<tr><td>${i + 1}</td><td>${r.preset}</td><td>${fmt(r.testMIoU, 6)}</td><td>${fmt(r.testMeanDice, 6)}</td><td>${fmt(r.testPixelAccuracy, 6)}</td><td>${fmt(r.p95Ms, 2)}</td></tr>`).join('');
  document.getElementById('stdout').textContent = data.stdoutTail || '';
  document.getElementById('stderr').textContent = data.stderrTail || '';
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    root = Path.cwd()
    stdout_path = Path("stdout.log")
    stderr_path = Path("stderr.log")
    process_meta_path: Path | None = None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            data = json.dumps(
                status_payload(self.root, self.stdout_path, self.stderr_path, self.process_meta_path),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--process-meta", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AITRAIN_SMP_PROGRESS_PORT", "8776")))
    args = parser.parse_args()
    Handler.root = args.work_dir.resolve()
    Handler.stdout_path = args.stdout.resolve()
    Handler.stderr_path = args.stderr.resolve()
    Handler.process_meta_path = args.process_meta.resolve() if args.process_meta else None
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
