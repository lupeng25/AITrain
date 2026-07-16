# ADR-005：V2 Worker/Python 进程树所有权

日期：2026-07-15  
状态：已接受

## 决策

所有训练和适配器进程由 Worker Host 的统一 Process Supervisor 托管。Windows 使用 Job Object 绑定后代进程；取消按“请求取消、优雅终止、超时强杀、确认退出”状态机执行。

## 后果

- GUI 不直接启动或终止 Python 子进程。
- Python stdout/stderr 不承担协议事件通道；结构化事件通过 Protocol V2 传递。
- 进程树清理和资源限制成为可自动化验证的 Worker 职责。
