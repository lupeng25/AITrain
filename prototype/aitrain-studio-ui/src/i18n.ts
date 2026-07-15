import type { Language, WorkspaceRoute } from "./types";

type Dictionary = Record<string, string>;

const zh: Dictionary = {
  appName: "AITrain Studio", appTagline: "本地训练工作台", overview: "总览", projects: "项目", datasets: "数据集",
  training: "训练实验", tasks: "任务与产物", models: "模型库", deployment: "部署验证", environment: "环境", settings: "系统设置",
  connected: "已连接", healthy: "正常", warning: "警告", blocked: "阻塞", running: "运行中", completed: "已完成", idle: "待运行",
  project: "项目", worker: "Worker", gpu: "GPU", inspect: "检查器", collapse: "收起", expand: "展开", language: "语言",
  refresh: "刷新", search: "搜索", filter: "筛选", all: "全部", details: "详情", status: "状态", actions: "操作", next: "下一步",
};

const en: Dictionary = {
  appName: "AITrain Studio", appTagline: "Local Training Workbench", overview: "Overview", projects: "Projects", datasets: "Datasets",
  training: "Training", tasks: "Tasks & Artifacts", models: "Model Registry", deployment: "Deployment", environment: "Environment", settings: "Settings",
  connected: "Connected", healthy: "Healthy", warning: "Warning", blocked: "Blocked", running: "Running", completed: "Completed", idle: "Idle",
  project: "Project", worker: "Worker", gpu: "GPU", inspect: "Inspector", collapse: "Collapse", expand: "Expand", language: "Language",
  refresh: "Refresh", search: "Search", filter: "Filter", all: "All", details: "Details", status: "Status", actions: "Actions", next: "Next step",
};

export const dictionaries: Record<Language, Dictionary> = { zh, en };

export const routeTitles: Record<WorkspaceRoute, { zh: string; en: string }> = {
  overview: { zh: "总览", en: "Overview" }, projects: { zh: "项目", en: "Projects" }, datasets: { zh: "数据集", en: "Datasets" },
  training: { zh: "训练实验", en: "Training experiments" }, tasks: { zh: "任务与产物", en: "Tasks & artifacts" }, models: { zh: "模型库", en: "Model registry" },
  deployment: { zh: "部署验证", en: "Deployment validation" }, environment: { zh: "环境", en: "Environment" }, settings: { zh: "系统设置", en: "System settings" },
};

export function translateStatus(status: string, language: Language) {
  return dictionaries[language][status] ?? status;
}

const englishUi: Record<string, string> = {
  "项目已就绪": "Project ready", "检查器": "Inspector", "当前运行": "Current run", "当前项目": "Current project", "当前数据集": "Current dataset", "选中任务": "Selected task", "候选模型": "Candidate model", "部署目标": "Deployment target", "当前工作站": "Current workstation", "应用配置": "Application settings",
  "数据准备": "Data preparation", "训练实验": "Training experiment", "评估与模型": "Evaluation & models", "部署验证": "Deployment validation", "已通过": "Passed", "等待当前训练": "Waiting for current run", "待运行": "Pending",
  "编辑配置": "Edit configuration", "停止训练": "Stop training", "启动训练": "Start training", "训练配置": "Training configuration", "实验参数快照": "Experiment parameter snapshot", "训练进度": "Training progress", "指标曲线": "Metric curves", "训练日志": "Training logs", "Checkpoint 与产物": "Checkpoints & artifacts", "使用官方 Ultralytics 训练入口": "Official Ultralytics training entry",
  "总览": "Overview", "项目": "Projects", "数据集": "Datasets", "任务与产物": "Tasks & artifacts", "模型库": "Model registry", "环境": "Environment", "系统设置": "System settings",
  "当前工作上下文": "Current context", "所有状态围绕当前项目组织": "All states are organized around this project", "任务类型": "Task type", "训练后端": "Training backend", "数据准备程度": "Data readiness", "查看质量问题": "Review quality issues", "打开运行监控": "Open run monitor", "最近任务": "Recent tasks", "查看全部": "View all", "推荐下一步": "Recommended next step", "等待训练完成并执行评估": "Wait for training and run evaluation", "查看已有模型基线": "Review model baseline", "补充 TensorRT 交付证据": "Add TensorRT delivery evidence",
  "项目列表": "Project list", "个本地项目": "local projects", "新建项目": "New project", "基本信息": "Basic information", "项目级配置与元数据状态": "Project configuration and metadata", "编辑": "Edit", "项目名称": "Project name", "最近更新": "Last updated", "项目根目录": "Project root", "标准目录结构": "Standard directory structure", "元数据状态": "Metadata status", "SQLite 元数据一致": "SQLite metadata consistent", "重新检查": "Check again", "打开目录": "Open folder",
  "数据集准备": "Dataset preparation", "质量与复核": "Quality & review", "数据集库": "Dataset library", "样本预览": "Sample preview", "导入数据": "Import data", "校验数据集": "Validate dataset", "正在校验…": "Validating…", "所选数据集详情": "Selected dataset details", "样本": "Samples", "已标注": "Annotated", "待复核": "To review", "类别": "Classes", "类别分布": "Class distribution", "数据集可用于训练": "Dataset ready for training", "质量概览": "Quality overview", "完整率": "Completeness", "重复样本": "Duplicates", "低清晰度": "Low clarity", "类别偏差": "Class skew", "轻微": "Minor", "总体质量良好": "Overall quality is good", "复核队列": "Review queue", "全部问题": "All issues", "缺失标注": "Missing annotation", "疑似重复": "Possible duplicate", "框偏移": "Box offset", "样本浏览": "Sample browser", "标注信息": "Annotation details", "文件": "File", "数据划分": "Split", "图像尺寸": "Image size", "标注框": "Boxes",
  "全部": "All", "运行中": "Running", "已完成": "Completed", "警告": "Warning", "刷新": "Refresh", "任务历史": "Task history", "任务检查器": "Task inspector", "产物": "Artifacts", "指标": "Metrics", "导出": "Exports", "预览": "Preview", "当前运行尚无导出": "No export for this run", "最近验证样本": "Latest validation sample",
  "模型版本": "Model versions", "评估报告": "Evaluation report", "模型对比": "Model comparison", "流水线记录": "Pipeline records", "已注册版本": "registered versions", "进入部署验证": "Open deployment validation", "已注册模型版本": "Registered model version", "模型大小": "Model size", "来源数据集": "Source dataset", "来源运行": "Source run", "创建时间": "Created", "固定测试集": "Fixed test set", "报告完整": "Report complete", "选择对比版本": "Select versions", "最多选择 3 个版本": "Select up to 3 versions", "关键指标对比": "Key metric comparison",
  "模型导出": "Model export", "推理验证": "Inference validation", "导出设置": "Export settings", "目标格式": "Target format", "输入尺寸": "Input size", "动态批次": "Dynamic batch", "输出目录": "Output folder", "开始导出": "Start export", "正在导出…": "Exporting…", "格式矩阵": "Format matrix", "支持": "Supported", "需证据": "Evidence required", "等待导出任务": "Waiting for export", "验证输入": "Validation input", "选择图像": "Choose image", "运行时": "Runtime", "置信度阈值": "Confidence threshold", "运行推理验证": "Run inference validation", "正在推理…": "Running inference…", "Overlay 预览": "Overlay preview", "结果摘要": "Result summary", "检测数": "Detections", "最高置信度": "Top confidence", "端到端延迟": "End-to-end latency", "验证链路": "Validation chain", "暂无结果": "No results",
  "运行环境": "Runtime environment", "交付证据": "Delivery evidence", "运行全部检查": "Run all checks", "正在检查…": "Checking…", "环境检查": "Environment checks", "本地训练与部署依赖": "Local training and deployment dependencies", "工作站摘要": "Workstation summary", "在线": "Online", "GPU 显存": "GPU memory", "GPU 温度": "GPU temperature", "项目磁盘": "Project disk", "Worker 延迟": "Worker latency", "证据": "Evidence", "范围": "Scope", "生成时间": "Generated", "状态": "Status",
  "内置能力": "Built-in capabilities", "应用设置": "Application settings", "内置能力矩阵": "Built-in capability matrix", "能力": "Capability", "产品入口": "Product entry", "部署范围": "Deployment scope", "能力声明遵循当前注册表": "Capability claims follow the current registry", "常规": "General", "界面语言": "Interface language", "默认项目目录": "Default project folder", "浏览": "Browse", "启动时恢复项目": "Restore project on startup", "任务完成通知": "Task completion notifications", "授权与本地路径": "License & local paths", "管理授权": "Manage license", "Worker 可执行文件": "Worker executable", "Python 环境": "Python environment", "模型缓存": "Model cache", "恢复默认": "Restore defaults", "保存设置": "Save settings", "关闭": "Close", "取消": "Cancel", "确认": "Confirm", "保存配置": "Save configuration", "停止当前训练？": "Stop current training?",
  "项目目录": "Project folder", "运行资源": "Runtime resources", "显存": "Memory", "快捷入口": "Shortcuts", "查看任务与产物": "View tasks & artifacts", "检查运行环境": "Check environment", "Worker 已连接": "Worker connected"
};

const originalText = new WeakMap<Node, string>();

export function translatePreviewDom(root: HTMLElement, language: Language) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node: Node | null = walker.nextNode();
  while (node) {
    const current = node.textContent ?? "";
    let source = originalText.get(node) ?? current;
    const sourceTrimmed = source.trim();
    const priorTranslated = englishUi[sourceTrimmed] ? source.replace(sourceTrimmed, englishUi[sourceTrimmed]) : source;
    if (!originalText.has(node) || (current !== source && current !== priorTranslated)) {
      source = current;
      originalText.set(node, source);
    }
    const trimmed = source.trim();
    const translated = language === "en" ? englishUi[trimmed] : undefined;
    const next = language === "en" && translated ? source.replace(trimmed, translated) : source;
    if (node.textContent !== next) node.textContent = next;
    node = walker.nextNode();
  }
  root.querySelectorAll<HTMLInputElement>("input[placeholder]").forEach((element) => {
    const source = element.dataset.i18nPlaceholder ?? element.placeholder;
    element.dataset.i18nPlaceholder = source;
    element.placeholder = language === "en" ? englishUi[source] ?? source : source;
  });
}
