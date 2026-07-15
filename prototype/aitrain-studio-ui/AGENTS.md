# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

# User Notes

- 方案 1 是唯一视觉基准：深海军蓝侧栏、浅色原生桌面工作面、克制边框与紧凑信息密度。
- 原始设计基准保存在 `references/option-1.png`，默认画面为“训练实验”的运行监控状态。
- 原型覆盖 9 个一级工作区和全部嵌套页签，使用 Hash 路由与集中 mock 状态，不连接真实 Worker、SQLite、训练或导出后端。
- 仅面向 1280–1920px 桌面视口；1280–1365px 导航收为 72px 图标栏，右侧检查器可折叠。
- 后端名称和产品边界必须遵循当前 AITrain Studio 能力注册表，不展示已移除的诊断或 scaffold 后端。
