#!/usr/bin/env python3
"""Serve AITrain full model lifecycle progress for a work directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def read_text(path: Path, limit: int = 20000) -> str:
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


def read_jsonl_tail(path: Path, limit: int = 120000) -> list[dict]:
    events: list[dict] = []
    text = read_text(path, limit)
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def is_pid_running(pid: str) -> bool:
    pid = str(pid or "").strip()
    if not pid.isdigit():
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return pid in result.stdout
    except Exception:
        return False


def row_summaries(root: Path) -> list[dict]:
    rows: list[dict] = []
    for folder in ("yolo", "ocr", "ocr-system"):
        group_root = root / folder
        if not group_root.exists():
            continue
        for path in group_root.rglob("row_summary.json"):
            payload = read_json(path)
            if isinstance(payload, dict):
                payload["_path"] = str(path)
                payload["_group"] = folder
                rows.append(payload)
    return rows


def yolo_request_context(row_root: Path) -> dict:
    request = read_json(row_root / "train_request.json")
    if not isinstance(request, dict):
        return {}
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "model": parameters.get("model") or parameters.get("modelPreset") or "",
        "task": request.get("taskType") or "",
        "epochs": parameters.get("epochs"),
        "device": parameters.get("device") or "",
    }


def last_payload_event(events: list[dict], event_type: str) -> dict:
    for event in reversed(events):
        if event.get("type") == event_type and isinstance(event.get("payload"), dict):
            return event
    return {}


def latest_yolo_worker_stage(root: Path) -> dict:
    yolo_root = root / "yolo"
    if not yolo_root.exists():
        return {}
    event_files: list[tuple[float, Path, str]] = []
    stage_files = {
        "train": "train_worker_events.jsonl",
        "onnx_inference": "onnx_inference_worker_events.jsonl",
        "deployment_onnx": "deployment_onnx_worker_events.jsonl",
    }
    for row_root in yolo_root.iterdir():
        if not row_root.is_dir():
            continue
        for stage, filename in stage_files.items():
            path = row_root / filename
            if path.exists():
                try:
                    event_files.append((path.stat().st_mtime, path, stage))
                except OSError:
                    continue
    if not event_files:
        return {}

    _, event_path, stage = max(event_files, key=lambda item: item[0])
    events = read_jsonl_tail(event_path)
    if not events:
        return {}
    last_event = events[-1]
    progress_event = last_payload_event(events, "progress")
    progress = progress_event.get("payload", {}) if progress_event else {}
    if not isinstance(progress, dict):
        progress = {}
    payload = last_event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    row_root = event_path.parent
    context = yolo_request_context(row_root)
    result = {
        "kind": "yolo",
        "row": row_root.name,
        "stage": stage,
        "eventPath": str(event_path),
        "eventType": str(last_event.get("type") or ""),
        "eventUpdatedAt": event_path.stat().st_mtime,
        "model": context.get("model", ""),
        "task": context.get("task", ""),
        "device": context.get("device", ""),
        "message": progress.get("message") or payload.get("message") or "",
    }
    for key in ("percent", "epoch", "epochs", "batch", "batches", "etaSeconds"):
        value = progress.get(key)
        if value is not None:
            result[key] = value
    if "epochs" not in result and context.get("epochs") is not None:
        result["epochs"] = context["epochs"]
    if last_event.get("type") in {"completed", "failed", "canceled"}:
        result["terminal"] = last_event.get("type")
    return result


def status_payload(root: Path) -> dict:
    pid = read_text(root / "full_model_lifecycle.pid", 100).strip()
    summary = read_json(root / "full_model_lifecycle_summary.json") or {}
    current_run = read_json(root / "current_run.json") or {}
    run_id = str(current_run.get("runId") or summary.get("runId") or "")
    all_rows = row_summaries(root)
    current_rows = [row for row in all_rows if run_id and str(row.get("runId") or "") == run_id]
    historical_rows = [row for row in all_rows if not run_id or str(row.get("runId") or "") != run_id]
    rows = current_rows if run_id else all_rows
    by_status = Counter(str(item.get("status", "unknown")) for item in rows)
    historical_by_status = Counter(str(item.get("status", "unknown")) for item in historical_rows)
    by_group: dict[str, Counter] = {}
    for item in rows:
        group = str(item.get("_group", "unknown"))
        by_group.setdefault(group, Counter())
        by_group[group][str(item.get("status", "unknown"))] += 1
    recent_rows = sorted(rows, key=lambda item: item.get("finishedAt", item.get("startedAt", "")), reverse=True)[:30]
    blockers = [
        row for row in recent_rows
        if str(row.get("status", "")) in {"blocked", "failed"} or row.get("reason") or row.get("failure")
    ][:10]
    stdout_tail = read_text(root / "full_model_lifecycle_stdout.log")
    stderr_tail = read_text(root / "full_model_lifecycle_stderr.log")
    current_stage = detect_current_stage(root, stdout_tail)
    controller_failure = read_json(root / "controller_failure.json")
    if isinstance(controller_failure, dict) and run_id and str(controller_failure.get("runId") or "") != run_id:
        controller_failure = None
    return {
        "root": str(root),
        "pid": pid,
        "running": is_pid_running(pid),
        "runId": run_id,
        "currentRun": current_run,
        "summary": summary,
        "datasets": read_json(root / "datasets_manifest.json"),
        "environment": read_json(root / "environment_self_check.json"),
        "rowCount": len(rows),
        "historicalRowCount": len(historical_rows),
        "byStatus": dict(by_status),
        "historicalByStatus": dict(historical_by_status),
        "byGroup": {key: dict(value) for key, value in by_group.items()},
        "recentRows": recent_rows,
        "recentBlockers": blockers,
        "currentStage": current_stage,
        "stdoutTail": stdout_tail,
        "stderrTail": stderr_tail,
        "controllerFailure": controller_failure,
    }


def parse_epoch_from_tail(text: str) -> dict:
    matches = list(re.finditer(r"epoch:\s*\[(\d+)\s*/\s*(\d+)\].*?(?:global_step|iter|batch)[^0-9]*(\d+)", text, re.I | re.S))
    if not matches:
        matches = list(re.finditer(r"epoch:\s*\[(\d+)\s*/\s*(\d+)\]", text, re.I))
    if not matches:
        return {}
    match = matches[-1]
    epoch = int(match.group(1))
    total = int(match.group(2))
    payload = {"epoch": epoch, "totalEpochs": total, "percent": round(epoch * 100.0 / max(total, 1), 3)}
    if len(match.groups()) >= 3 and match.group(3):
        payload["step"] = int(match.group(3))
    return payload


def detect_current_stage(root: Path, stdout_tail: str) -> dict:
    lines = [line.strip() for line in stdout_tail.splitlines() if line.strip().startswith("Full lifecycle:")]
    if not lines:
        return latest_yolo_worker_stage(root)
    current = lines[-1]
    payload = {"controllerLine": current}
    match = re.search(r"Full lifecycle:\s+(ocr-system|ocr|ncnn-smoke|tensorrt-smoke)-([^:]+):", current)
    if match:
        kind = match.group(1)
        row_name = match.group(2)
        payload.update({"kind": kind, "row": row_name})
        if kind == "ocr":
            row_root = root / "ocr" / row_name / "official"
            candidates = [
                ("train", row_root / "official_det_train.log"),
                ("train", row_root / "official_train.log"),
                ("export", row_root / "official_det_export.log"),
                ("export", row_root / "official_export.log"),
                ("predict", row_root / "official_predict.log"),
            ]
        elif kind == "ocr-system":
            row_root = root / "ocr-system" / row_name / "official"
            candidates = [("system_predict", row_root / "official_system_predict.log")]
        else:
            candidates = []
        for stage, path in candidates:
            if path.exists():
                tail = read_text(path, 4000)
                payload.update({"stage": stage, "logPath": str(path), "logTail": tail})
                payload.update(parse_epoch_from_tail(tail))
                break
    elif "worker train/export" in current or "worker inference" in current:
        payload.update(latest_yolo_worker_stage(root))
    elif not payload.get("stage"):
        yolo_stage = latest_yolo_worker_stage(root)
        if yolo_stage:
            payload.update(yolo_stage)
    return payload


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AITrain Full Lifecycle Progress</title>
  <style>
    :root { color-scheme: light dark; font-family: "Segoe UI", Arial, sans-serif; }
    body { margin: 0; background: #101418; color: #e8edf2; }
    header { padding: 16px 20px; background: #17202a; border-bottom: 1px solid #2b3947; position: sticky; top: 0; }
    h1 { margin: 0 0 4px; font-size: 20px; }
    main { padding: 18px 20px; display: grid; gap: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .panel { background: #151c23; border: 1px solid #283542; border-radius: 8px; padding: 14px; }
    .label { color: #91a4b7; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 24px; font-weight: 650; margin-top: 6px; }
    .ok { color: #58d68d; } .warn { color: #f4d03f; } .bad { color: #ff7675; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0c1116; border: 1px solid #283542; padding: 12px; border-radius: 6px; max-height: 380px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #283542; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }
    th { color: #91a4b7; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <h1>AITrain Full Lifecycle Progress</h1>
    <div id="sub">Loading...</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="label">进程</div><div class="value" id="running">-</div></div>
      <div class="panel"><div class="label">总状态</div><div class="value" id="status">-</div></div>
      <div class="panel"><div class="label">当前 Rows</div><div class="value" id="rows">-</div></div>
      <div class="panel"><div class="label">历史残留</div><div class="value" id="historical">-</div></div>
    </section>
    <section class="grid">
      <div class="panel"><div class="label">当前 run 状态分布</div><pre id="byStatus"></pre></div>
      <div class="panel"><div class="label">历史状态分布</div><pre id="historicalByStatus"></pre></div>
      <div class="panel"><div class="label">最近 blocker / failed</div><pre id="blockers"></pre></div>
      <div class="panel"><div class="label">当前阶段</div><pre id="currentStage"></pre></div>
    </section>
    <section class="panel">
      <div class="label">最近完成/更新的当前 Row</div>
      <table><thead><tr><th>Group</th><th>Name</th><th>Status</th><th>Finished</th><th>Note</th></tr></thead><tbody id="recent"></tbody></table>
    </section>
    <section class="panel"><div class="label">stdout tail</div><pre id="stdout"></pre></section>
    <section class="panel"><div class="label">stderr tail</div><pre id="stderr"></pre></section>
  </main>
<script>
async function refresh() {
  const res = await fetch('/api/status?ts=' + Date.now());
  const data = await res.json();
  const status = data.currentRun?.status || data.summary?.status || '-';
  document.getElementById('sub').textContent = `${data.root} | runId ${data.runId || '-'} | PID ${data.pid || '-'}`;
  document.getElementById('running').textContent = data.running ? 'running' : 'stopped';
  document.getElementById('running').className = 'value ' + (data.running ? 'ok' : 'warn');
  document.getElementById('status').textContent = status;
  document.getElementById('status').className = 'value ' + (status === 'failed' || status === 'controller_failed' ? 'bad' : status === 'blocked' ? 'warn' : 'ok');
  document.getElementById('rows').textContent = data.rowCount;
  document.getElementById('historical').textContent = data.historicalRowCount;
  document.getElementById('historical').className = 'value ' + (data.historicalRowCount ? 'warn' : 'ok');
  document.getElementById('byStatus').textContent = JSON.stringify(data.byStatus, null, 2);
  document.getElementById('historicalByStatus').textContent = JSON.stringify(data.historicalByStatus, null, 2);
  document.getElementById('blockers').textContent = JSON.stringify(data.recentBlockers, null, 2);
  document.getElementById('currentStage').textContent = JSON.stringify(data.currentStage || {}, null, 2);
  document.getElementById('stdout').textContent = data.stdoutTail || '';
  document.getElementById('stderr').textContent = data.stderrTail || '';
  const rows = data.recentRows || [];
  document.getElementById('recent').innerHTML = rows.map(r => `<tr><td>${r._group || ''}</td><td>${r.name || ''}</td><td>${r.status || ''}</td><td>${r.finishedAt || ''}</td><td>${r.reason || r.failure || (r.findings || []).join(', ') || ''}</td></tr>`).join('');
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    root = Path.cwd()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            data = json.dumps(status_payload(self.root), ensure_ascii=False).encode("utf-8")
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
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AITRAIN_PROGRESS_PORT", "8765")))
    args = parser.parse_args()
    Handler.root = args.work_dir.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
