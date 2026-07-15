import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AddRegular,
  ArrowDownloadRegular,
  ArrowSyncRegular,
  BoxRegular,
  BrainCircuitRegular,
  CheckmarkCircleRegular,
  ChevronDownRegular,
  ChevronLeftRegular,
  ChevronRightRegular,
  DatabaseRegular,
  DismissRegular,
  DocumentRegular,
  EditRegular,
  EyeRegular,
  FilterRegular,
  FolderRegular,
  GlobeRegular,
  HomeRegular,
  ImageRegular,
  MoreHorizontalRegular,
  OpenRegular,
  PanelRightRegular,
  PlayRegular,
  PulseRegular,
  RocketRegular,
  SearchRegular,
  SettingsRegular,
  ShieldCheckmarkRegular,
  StopRegular,
  TableRegular,
  TaskListSquareLtrRegular,
  WarningRegular,
} from "@fluentui/react-icons";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  artifacts,
  capabilityRows,
  datasets,
  environmentChecks,
  initialTrainingRun,
  logs,
  metrics,
  models,
  projects,
  tasks,
} from "./mockData";
import { dictionaries, routeTitles, translatePreviewDom, translateStatus } from "./i18n";
import type { HealthStatus, Language, ModelRecord, ProjectState, TaskRecord, WorkspaceRoute } from "./types";

type IconType = typeof HomeRegular;

const navItems: Array<{ route: WorkspaceRoute; icon: IconType }> = [
  { route: "overview", icon: HomeRegular },
  { route: "projects", icon: FolderRegular },
  { route: "datasets", icon: DatabaseRegular },
  { route: "training", icon: BrainCircuitRegular },
  { route: "tasks", icon: TaskListSquareLtrRegular },
  { route: "models", icon: BoxRegular },
  { route: "deployment", icon: RocketRegular },
  { route: "environment", icon: PulseRegular },
  { route: "settings", icon: SettingsRegular },
];

function routeFromHash(): WorkspaceRoute {
  const hash = window.location.hash.replace(/^#\/?/, "") as WorkspaceRoute;
  return navItems.some((item) => item.route === hash) ? hash : "training";
}

function StatusPill({ status, language, children }: { status: HealthStatus; language: Language; children?: ReactNode }) {
  return (
    <span className={`status-pill status-${status}`}>
      <span className="status-dot" />
      {children ?? translateStatus(status, language)}
    </span>
  );
}

function Button({ children, variant = "secondary", icon: Icon, onClick, disabled, type = "button", testId }: {
  children: ReactNode; variant?: "primary" | "secondary" | "danger" | "ghost"; icon?: IconType; onClick?: () => void; disabled?: boolean; type?: "button" | "submit"; testId?: string;
}) {
  return <button data-testid={testId} className={`button button-${variant}`} onClick={onClick} disabled={disabled} type={type}>{Icon && <Icon />}<span>{children}</span></button>;
}

function Panel({ title, subtitle, action, className = "", children }: { title?: string; subtitle?: string; action?: ReactNode; className?: string; children: ReactNode }) {
  return <section className={`panel ${className}`}>
    {(title || action) && <header className="panel-header"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</header>}
    <div className="panel-body">{children}</div>
  </section>;
}

function Tabs({ items, value, onChange, label }: { items: Array<{ id: string; label: string }>; value: string; onChange: (id: string) => void; label?: string }) {
  return <div className="tabs" role="tablist" aria-label={label}>{items.map((item) => <button key={item.id} role="tab" aria-selected={value === item.id} className={value === item.id ? "active" : ""} onClick={() => onChange(item.id)}>{item.label}</button>)}</div>;
}

function Metric({ label, value, hint, tone = "default" }: { label: string; value: string; hint?: string; tone?: "default" | "success" | "warning" }) {
  return <div className={`metric metric-${tone}`}><span>{label}</span><strong>{value}</strong>{hint && <small>{hint}</small>}</div>;
}

function Modal({ title, children, onClose, onConfirm, confirmText = "确认" }: { title: string; children: ReactNode; onClose: () => void; onConfirm: () => void; confirmText?: string }) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}><div className="modal" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.stopPropagation()}>
    <header><h2>{title}</h2><button className="icon-button" onClick={onClose} aria-label="关闭"><DismissRegular /></button></header>
    <div className="modal-content">{children}</div>
    <footer><Button onClick={onClose}>取消</Button><Button variant="primary" onClick={onConfirm}>{confirmText}</Button></footer>
  </div></div>;
}

function App() {
  const [route, setRoute] = useState<WorkspaceRoute>(routeFromHash);
  const [language, setLanguage] = useState<Language>("zh");
  const [project, setProject] = useState<ProjectState>(projects[0]);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [toast, setToast] = useState("");

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    if (!window.location.hash) window.location.hash = "/training";
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const root = document.getElementById("root");
    if (!root) return;
    translatePreviewDom(root, language);
    const observer = new MutationObserver(() => translatePreviewDom(root, language));
    observer.observe(root, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [language, route]);

  const t = dictionaries[language];
  const navigate = (next: WorkspaceRoute) => { window.location.hash = `/${next}`; setRoute(next); };

  return <div className={`app-shell ${inspectorOpen ? "inspector-visible" : "inspector-hidden"}`}>
    <aside className="sidebar">
      <div className="brand"><img src="/assets/app-icon.png" alt="AITrain Studio"/><div><strong>AITrain Studio</strong><span>{t.appTagline}</span></div></div>
      <nav aria-label="主导航">{navItems.map(({ route: itemRoute, icon: Icon }) => <button key={itemRoute} className={route === itemRoute ? "active" : ""} onClick={() => navigate(itemRoute)} title={routeTitles[itemRoute][language]}><Icon/><span>{routeTitles[itemRoute][language]}</span>{route === itemRoute && <ChevronRightRegular className="nav-chevron"/>}</button>)}</nav>
      <div className="sidebar-footer"><div className="user-avatar">LM</div><div><strong>Local Admin</strong><span>工作站 · 在线</span></div><MoreHorizontalRegular/></div>
    </aside>

    <header className="topbar">
      <div className="topbar-project"><span>{t.project}</span><label><select value={project.id} onChange={(event) => setProject(projects.find((item) => item.id === event.target.value) ?? projects[0])}>{projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><ChevronDownRegular /></label></div>
      <div className="runtime-strip">
        <span><span className="online-dot"/>{t.worker} <strong>{t.connected}</strong></span>
        <span className="divider"/>
        <span>{t.gpu} <strong>RTX 4090 D</strong><em>91%</em></span>
      </div>
      <div className="top-actions">
        <button className="language-switch" data-testid="language-switch" onClick={() => setLanguage((value) => value === "zh" ? "en" : "zh")}><GlobeRegular/><span>{language === "zh" ? "EN" : "中"}</span></button>
        <button className={`icon-button ${inspectorOpen ? "active" : ""}`} aria-label={t.inspect} onClick={() => setInspectorOpen((value) => !value)}><PanelRightRegular/></button>
      </div>
    </header>

    <main className="workspace">
      <div className="page-heading"><div><p className="eyebrow">{project.name} / {project.task}</p><h1>{routeTitles[route][language]}</h1></div><div className="page-heading-actions"><StatusPill status="healthy" language={language}>项目已就绪</StatusPill></div></div>
      <Workspace route={route} language={language} project={project} notify={setToast} navigate={navigate}/>
    </main>

    {inspectorOpen && <aside className="inspector"><Inspector route={route} language={language} project={project} onClose={() => setInspectorOpen(false)} navigate={navigate}/></aside>}
    {toast && <div className="toast" role="status"><CheckmarkCircleRegular/>{toast}</div>}
  </div>;
}

function Workspace({ route, language, project, notify, navigate }: { route: WorkspaceRoute; language: Language; project: ProjectState; notify: (message: string) => void; navigate: (route: WorkspaceRoute) => void }) {
  switch (route) {
    case "overview": return <OverviewPage language={language} project={project} navigate={navigate}/>;
    case "projects": return <ProjectsPage language={language} project={project} notify={notify}/>;
    case "datasets": return <DatasetsPage language={language} notify={notify}/>;
    case "training": return <TrainingPage language={language} notify={notify}/>;
    case "tasks": return <TasksPage language={language}/>;
    case "models": return <ModelsPage language={language}/>;
    case "deployment": return <DeploymentPage language={language} notify={notify}/>;
    case "environment": return <EnvironmentPage language={language} notify={notify}/>;
    case "settings": return <SettingsPage language={language} notify={notify}/>;
  }
}

function FlowRail({ active, onNavigate }: { active: number; onNavigate?: (index: number) => void }) {
  const steps = [
    ["1", "数据准备", "bearing-v3 · 已通过"],
    ["2", "训练实验", "Epoch 18 / 50"],
    ["3", "评估与模型", "等待当前训练"],
    ["4", "部署验证", "ONNX · 待运行"],
  ];
  return <div className="flow-rail">{steps.map(([num, name, detail], index) => <button key={name} className={`${index === active ? "active" : ""} ${index < active ? "done" : ""}`} onClick={() => onNavigate?.(index)}><span>{index < active ? "✓" : num}</span><div><strong>{name}</strong><small>{detail}</small></div>{index < steps.length - 1 && <i/>}</button>)}</div>;
}

function OverviewPage({ language, project, navigate }: { language: Language; project: ProjectState; navigate: (route: WorkspaceRoute) => void }) {
  return <div className="page-stack">
    <FlowRail active={1} onNavigate={(index) => navigate((["datasets", "training", "models", "deployment"] as WorkspaceRoute[])[index])}/>
    <div className="overview-grid">
      <Panel title="当前工作上下文" subtitle="所有状态围绕当前项目组织" className="context-panel">
        <div className="context-row"><div><span>项目</span><strong>{project.name}</strong></div><div><span>任务类型</span><strong>{project.task}</strong></div><div><span>训练后端</span><strong>{project.backend}</strong></div></div>
        <div className="readiness"><div className="readiness-ring"><strong>92</strong><span>%</span></div><div><h3>数据准备程度</h3><p>2,816 / 2,840 个样本已标注，24 个样本进入复核队列。</p><button onClick={() => navigate("datasets")}>查看质量问题 <ChevronRightRegular/></button></div></div>
      </Panel>
      <Panel title="当前运行" subtitle="run-20260710-0912" action={<StatusPill status="running" language={language}/>}>
        <div className="run-focus"><div><span>YOLO11n · bearing-v3</span><strong>Epoch 18 / 50</strong></div><b>36%</b></div>
        <div className="progress"><i style={{ width: "36%" }}/></div>
        <div className="compact-metrics"><Metric label="mAP50" value="0.842" tone="success"/><Metric label="Precision" value="0.817"/><Metric label="剩余" value="02:31"/></div>
        <Button variant="primary" icon={EyeRegular} onClick={() => navigate("training")}>打开运行监控</Button>
      </Panel>
    </div>
    <div className="two-column wide-left">
      <Panel title="最近任务" subtitle="按时间排序的跨流程操作" action={<button className="text-button" onClick={() => navigate("tasks")}>查看全部 <ChevronRightRegular/></button>}>
        <TaskTable rows={tasks.slice(0, 4)} language={language} compact/>
      </Panel>
      <Panel title="推荐下一步" subtitle="基于当前项目状态">
        <div className="next-step"><span><ShieldCheckmarkRegular/></span><div><strong>等待训练完成并执行评估</strong><p>当前最佳 checkpoint 已更新。训练完成后，使用固定测试集运行官方 Ultralytics val()。</p></div></div>
        <button className="next-link" onClick={() => navigate("models")}><span>查看已有模型基线</span><ChevronRightRegular/></button>
        <button className="next-link" onClick={() => navigate("environment")}><span>补充 TensorRT 交付证据</span><ChevronRightRegular/></button>
      </Panel>
    </div>
  </div>;
}

function ProjectsPage({ language, project, notify }: { language: Language; project: ProjectState; notify: (message: string) => void }) {
  const [selected, setSelected] = useState(project.id);
  const current = projects.find((item) => item.id === selected) ?? project;
  return <div className="two-column project-layout">
    <Panel title="项目列表" subtitle={`${projects.length} 个本地项目`} action={<Button variant="primary" icon={AddRegular} onClick={() => notify("已打开新建项目向导（预览）")}>新建项目</Button>}>
      <div className="project-list">{projects.map((item) => <button key={item.id} className={selected === item.id ? "active" : ""} onClick={() => setSelected(item.id)}><span className="project-icon"><FolderRegular/></span><div><strong>{item.name}</strong><small>{item.task} · {item.backend}</small><em>{item.updatedAt}</em></div><ChevronRightRegular/></button>)}</div>
    </Panel>
    <div className="page-stack compact-gap">
      <Panel title="基本信息" subtitle="项目级配置与元数据状态" action={<Button icon={EditRegular} onClick={() => notify("项目编辑面板已打开（预览）")}>编辑</Button>}>
        <dl className="detail-grid"><div><dt>项目名称</dt><dd>{current.name}</dd></div><div><dt>任务类型</dt><dd>{current.task}</dd></div><div><dt>训练后端</dt><dd>{current.backend}</dd></div><div><dt>最近更新</dt><dd>{current.updatedAt}</dd></div><div className="span-2"><dt>项目根目录</dt><dd className="mono">{current.root}</dd></div></dl>
      </Panel>
      <Panel title="标准目录结构" subtitle="由 ProjectRepository 管理的项目工作区">
        <div className="folder-tree"><p><ChevronDownRegular/> <FolderRegular/> <strong>{current.name}</strong></p>{["datasets", "experiments", "models", "exports", "reports"].map((name) => <p key={name} className="child"><FolderRegular/> {name}<span>{name === "datasets" ? `${current.datasets} 个数据集` : name === "experiments" ? `${current.experiments} 次实验` : name === "models" ? `${current.models} 个版本` : "已创建"}</span></p>)}</div>
      </Panel>
      <Panel title="元数据状态"><div className="health-row"><StatusPill status="healthy" language={language}>SQLite 元数据一致</StatusPill><span>最后检查 10:41</span><Button icon={ArrowSyncRegular} onClick={() => notify("元数据检查完成，未发现问题")}>重新检查</Button></div></Panel>
    </div>
  </div>;
}

function DatasetsPage({ language, notify }: { language: Language; notify: (message: string) => void }) {
  const [section, setSection] = useState("prepare");
  const [view, setView] = useState("library");
  const [selected, setSelected] = useState(datasets[0].id);
  const [validation, setValidation] = useState<"idle" | "validating" | "success" | "issues">("idle");
  const [qualityFilter, setQualityFilter] = useState("all");
  const current = datasets.find((item) => item.id === selected) ?? datasets[0];
  const validate = () => {
    setValidation("validating");
    window.setTimeout(() => { setValidation(current.id === "bearing-v3" ? "success" : "issues"); notify(current.id === "bearing-v3" ? "数据校验完成：未发现阻塞项" : "数据校验完成：发现 8 个问题"); }, 900);
  };
  return <div className="page-stack">
    <Tabs label="数据集工作区" value={section} onChange={setSection} items={[{ id: "prepare", label: "数据集准备" }, { id: "quality", label: "质量与复核" }]}/>
    {section === "prepare" ? <>
      <div className="toolbar"><Tabs label="数据集视图" value={view} onChange={setView} items={[{ id: "library", label: "数据集库" }, { id: "preview", label: "样本预览" }]}/><div className="toolbar-actions"><Button icon={ArrowDownloadRegular} onClick={() => notify("导入向导已打开（预览）")}>导入数据</Button><Button variant="primary" icon={validation === "validating" ? ArrowSyncRegular : CheckmarkCircleRegular} onClick={validate} disabled={validation === "validating"}>{validation === "validating" ? "正在校验…" : "校验数据集"}</Button></div></div>
      {view === "library" ? <div className="dataset-grid">
        <Panel title="数据集库" subtitle="当前项目中的版本化数据集">
          <div className="dataset-list">{datasets.map((item) => <button key={item.id} className={selected === item.id ? "active" : ""} onClick={() => { setSelected(item.id); setValidation("idle"); }}><DatabaseRegular/><div><strong>{item.name}</strong><span>{item.sampleCount.toLocaleString()} 样本 · {item.split}</span><small>{item.updatedAt}</small></div><StatusPill status={item.status} language={language}/></button>)}</div>
        </Panel>
        <Panel title="所选数据集详情" subtitle={current.path} action={<Button icon={OpenRegular} onClick={() => notify("目录打开动作仅作演示")}>打开目录</Button>}>
          <div className="compact-metrics four"><Metric label="样本" value={current.sampleCount.toLocaleString()}/><Metric label="已标注" value={current.annotatedCount.toLocaleString()} tone="success"/><Metric label="待复核" value={(current.sampleCount - current.annotatedCount).toString()} tone="warning"/><Metric label="类别" value={current.classes.length.toString()}/></div>
          {validation !== "idle" && <div className={`validation-banner ${validation}`}>
            {validation === "validating" ? <ArrowSyncRegular className="spin"/> : validation === "success" ? <CheckmarkCircleRegular/> : <WarningRegular/>}
            <div><strong>{validation === "validating" ? "正在扫描标注、图像和数据划分" : validation === "success" ? "数据集可用于训练" : "发现需要处理的问题"}</strong><span>{validation === "validating" ? "正在检查 2,840 个样本…" : validation === "success" ? "路径、标注和类别映射均一致" : "5 个缺失标注，3 个重复样本"}</span></div>
          </div>}
          <h3 className="section-title">类别分布</h3>
          <div className="chart-short"><ResponsiveContainer width="100%" height="100%"><BarChart data={current.classes} layout="vertical" margin={{ left: 8, right: 16 }}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number" hide/><YAxis type="category" dataKey="name" width={126} tick={{ fontSize: 11 }}/><Tooltip/><Bar dataKey="count" fill="#2563eb" radius={[0, 4, 4, 0]}/></BarChart></ResponsiveContainer></div>
        </Panel>
      </div> : <SamplePreview/>}
    </> : <div className="quality-grid">
      <Panel title="质量概览" subtitle="标注完整性、重复项和样本异常">
        <div className="compact-metrics four"><Metric label="完整率" value="99.2%" tone="success"/><Metric label="重复样本" value="3" tone="warning"/><Metric label="低清晰度" value="11" tone="warning"/><Metric label="类别偏差" value="轻微"/></div>
        <div className="quality-score"><div><strong>94</strong><span>/ 100</span></div><p><b>总体质量良好</b><br/>建议优先复核 24 个低置信度或标注缺失样本。</p></div>
      </Panel>
      <Panel title="复核队列" subtitle="按风险和置信度排序" action={<label className="select-wrap"><FilterRegular/><select value={qualityFilter} onChange={(e) => setQualityFilter(e.target.value)}><option value="all">全部问题</option><option value="missing">缺失标注</option><option value="blur">低清晰度</option></select></label>}>
        <div className="review-list">{[
          ["BRG_02761.jpg", "缺失标注", "P1", "0.96"], ["BRG_01842.jpg", "低清晰度", "P2", "0.72"], ["BRG_00991.jpg", "疑似重复", "P2", "0.89"], ["BRG_02217.jpg", "框偏移", "P2", "0.68"],
        ].filter((row) => qualityFilter === "all" || (qualityFilter === "missing" ? row[1] === "缺失标注" : row[1] === "低清晰度")).map((row, index) => <button key={row[0]}><img src={index % 2 === 0 ? "/assets/bearing-defect.png" : "/assets/bearing-clean.png"} alt="轴承样本"/><div><strong>{row[0]}</strong><span>{row[1]} · 置信度 {row[3]}</span></div><em>{row[2]}</em><ChevronRightRegular/></button>)}</div>
      </Panel>
    </div>}
  </div>;
}

function SamplePreview() {
  const [sample, setSample] = useState(0);
  const items = [
    { src: "/assets/bearing-defect.png", name: "BRG_02761.jpg", label: "outer-race scratch", split: "train" },
    { src: "/assets/bearing-clean.png", name: "BRG_01842.jpg", label: "normal", split: "val" },
    { src: "/assets/bearing-overlay.png", name: "BRG_00991.jpg", label: "outer-race scratch", split: "test" },
  ];
  const current = items[sample];
  return <div className="sample-layout">
    <Panel title="样本浏览" subtitle="轴承缺陷检测 · bearing-v3"><div className="sample-stage"><img src={current.src} alt={current.name}/><span>{sample + 1} / {items.length}</span><button className="sample-prev" onClick={() => setSample((sample - 1 + items.length) % items.length)}><ChevronLeftRegular/></button><button className="sample-next" onClick={() => setSample((sample + 1) % items.length)}><ChevronRightRegular/></button></div></Panel>
    <Panel title="标注信息"><dl className="detail-list"><div><dt>文件</dt><dd>{current.name}</dd></div><div><dt>数据划分</dt><dd>{current.split}</dd></div><div><dt>类别</dt><dd>{current.label}</dd></div><div><dt>图像尺寸</dt><dd>1536 × 1152</dd></div><div><dt>标注框</dt><dd>{current.label === "normal" ? "0" : "1"}</dd></div></dl><div className="thumbnail-row">{items.map((item, index) => <button className={sample === index ? "active" : ""} key={item.name} onClick={() => setSample(index)}><img src={item.src} alt={item.name}/></button>)}</div></Panel>
  </div>;
}

function TrainingPage({ language, notify }: { language: Language; notify: (message: string) => void }) {
  const [tab, setTab] = useState("metrics");
  const [status, setStatus] = useState<HealthStatus>(initialTrainingRun.status);
  const [modal, setModal] = useState<"config" | "stop" | null>(null);
  const running = status === "running";
  const toggleRun = () => {
    if (running) setModal("stop");
    else { setStatus("running"); notify("训练任务已启动（演示状态）"); }
  };
  return <div className="page-stack training-page">
    <FlowRail active={1}/>
    <div className="training-header">
      <div><StatusPill status={status} language={language}/><h2>{initialTrainingRun.name}</h2><p>{initialTrainingRun.backend} · {initialTrainingRun.model} · bearing-v3</p></div>
      <div className="training-actions"><Button icon={EditRegular} onClick={() => setModal("config")}>编辑配置</Button><Button variant={running ? "danger" : "primary"} icon={running ? StopRegular : PlayRegular} onClick={toggleRun} testId="training-toggle">{running ? "停止训练" : "启动训练"}</Button></div>
    </div>
    <div className="training-grid">
      <Panel title="训练配置" subtitle="实验参数快照" className="config-panel">
        <dl className="detail-list"><div><dt>模型</dt><dd>yolo11n.pt</dd></div><div><dt>数据集</dt><dd>bearing-v3</dd></div><div><dt>Epochs</dt><dd>50</dd></div><div><dt>Batch</dt><dd>16</dd></div><div><dt>Image size</dt><dd>640</dd></div><div><dt>Device</dt><dd>CUDA:0</dd></div><div><dt>Optimizer</dt><dd>auto</dd></div><div><dt>Workers</dt><dd>8</dd></div></dl>
        <div className="config-note"><ShieldCheckmarkRegular/><span>使用官方 Ultralytics 训练入口</span></div>
      </Panel>
      <div className="run-main">
        <Panel className="progress-panel">
          <div className="epoch-row"><div><span>训练进度</span><strong>Epoch {initialTrainingRun.epoch} / {initialTrainingRun.totalEpochs}</strong></div><b>{running ? "36%" : "已暂停"}</b></div>
          <div className="progress large"><i className={running ? "" : "paused"} style={{ width: "36%" }}/></div>
          <div className="compact-metrics five"><Metric label="mAP50" value="0.842" hint="+0.016" tone="success"/><Metric label="mAP50-95" value="0.576" hint="+0.011"/><Metric label="Precision" value="0.817" hint="+0.014"/><Metric label="Recall" value="0.791" hint="+0.009"/><Metric label="ETA" value="02:31:18"/></div>
        </Panel>
        <Panel className="training-detail">
          <Tabs label="训练详情" value={tab} onChange={setTab} items={[{ id: "metrics", label: "指标曲线" }, { id: "logs", label: "训练日志" }, { id: "artifacts", label: "Checkpoint 与产物" }]}/>
          {tab === "metrics" && <div className="chart-large"><ResponsiveContainer width="100%" height="100%"><LineChart data={metrics} margin={{ top: 18, right: 18, left: -12, bottom: 4 }}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="epoch"/><YAxis domain={[0.3, 1]}/><Tooltip/><Legend/><Line type="monotone" dataKey="map50" name="mAP50" stroke="#2563eb" strokeWidth={2.5} dot={false}/><Line type="monotone" dataKey="precision" name="Precision" stroke="#16a34a" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="recall" name="Recall" stroke="#d97706" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div>}
          {tab === "logs" && <div className="log-view"><div className="log-toolbar"><span>实时输出</span><label><input type="checkbox" defaultChecked/> 自动滚动</label></div>{logs.map((line, index) => <code key={index}>{line}</code>)}</div>}
          {tab === "artifacts" && <ArtifactTable/>}
        </Panel>
      </div>
    </div>
    {modal === "config" && <Modal title="编辑训练配置" onClose={() => setModal(null)} onConfirm={() => { setModal(null); notify("训练配置已更新（仅预览状态）"); }} confirmText="保存配置"><div className="form-grid"><label>Epochs<input defaultValue="50"/></label><label>Batch size<input defaultValue="16"/></label><label>Image size<select defaultValue="640"><option>640</option><option>1024</option></select></label><label>Device<select defaultValue="CUDA:0"><option>CUDA:0</option><option>CPU</option></select></label></div><p className="modal-hint">更改将在下一次启动训练时生效。</p></Modal>}
    {modal === "stop" && <Modal title="停止当前训练？" onClose={() => setModal(null)} onConfirm={() => { setStatus("idle"); setModal(null); notify("训练任务已停止，checkpoint 已保留"); }} confirmText="停止训练"><p>停止后不会删除已有 checkpoint 和指标记录，可以从当前实验创建新运行。</p></Modal>}
  </div>;
}

function TaskTable({ rows, language, compact = false, selected, onSelect }: { rows: TaskRecord[]; language: Language; compact?: boolean; selected?: string; onSelect?: (id: string) => void }) {
  return <div className={`data-table ${compact ? "compact" : ""}`} role="table"><div className="table-row table-head"><span>任务</span><span>类型</span><span>状态</span><span>进度</span><span>开始时间</span></div>{rows.map((task) => <button key={task.id} className={`table-row ${selected === task.id ? "selected" : ""}`} onClick={() => onSelect?.(task.id)}><span><strong>{task.name}</strong><small>{task.id}</small></span><span>{task.type}</span><span><StatusPill status={task.status} language={language}/></span><span><i className="mini-progress"><b style={{ width: `${task.progress}%` }}/></i>{task.progress}%</span><span>{task.startedAt}</span></button>)}</div>;
}

function ArtifactTable() {
  return <div className="artifact-list">{artifacts.map((item) => <div key={item.id}><span className="file-icon">{item.kind === "Metrics" ? <TableRegular/> : <DocumentRegular/>}</span><div><strong>{item.name}</strong><small>{item.path}</small></div><span>{item.format}</span><span>{item.size}</span><button className="icon-button" aria-label="更多"><MoreHorizontalRegular/></button></div>)}</div>;
}

function TasksPage({ language }: { language: Language }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(tasks[0].id);
  const [tab, setTab] = useState("artifacts");
  const rows = tasks.filter((task) => (filter === "all" || task.status === filter) && task.name.toLowerCase().includes(query.toLowerCase()));
  const current = tasks.find((task) => task.id === selected) ?? tasks[0];
  return <div className="page-stack">
    <div className="toolbar"><div className="search-box"><SearchRegular/><input aria-label="搜索任务" placeholder="搜索任务名称或 ID" value={query} onChange={(e) => setQuery(e.target.value)}/></div><div className="segmented">{[["all", "全部"], ["running", "运行中"], ["completed", "已完成"], ["warning", "警告"]].map(([id, label]) => <button key={id} className={filter === id ? "active" : ""} onClick={() => setFilter(id)}>{label}</button>)}</div><Button icon={ArrowSyncRegular}>刷新</Button></div>
    <div className="tasks-layout">
      <Panel title="任务历史" subtitle={`显示 ${rows.length} / ${tasks.length} 项`}><TaskTable rows={rows} language={language} selected={selected} onSelect={setSelected}/></Panel>
      <Panel title="任务检查器" subtitle={current.id} action={<StatusPill status={current.status} language={language}/>}>
        <div className="task-summary"><h3>{current.name}</h3><p>{current.type} · {current.startedAt}</p><div className="progress"><i style={{ width: `${current.progress}%` }}/></div><span>{current.progress}% · {current.duration}</span></div>
        <Tabs label="任务详情" value={tab} onChange={setTab} items={[{ id: "artifacts", label: "产物" }, { id: "metrics", label: "指标" }, { id: "exports", label: "导出" }, { id: "preview", label: "预览" }]}/>
        {tab === "artifacts" && <ArtifactTable/>}
        {tab === "metrics" && <div className="compact-metrics two"><Metric label="mAP50" value="0.842" tone="success"/><Metric label="mAP50-95" value="0.576"/><Metric label="Precision" value="0.817"/><Metric label="Recall" value="0.791"/></div>}
        {tab === "exports" && <div className="empty-state"><ArrowDownloadRegular/><strong>当前运行尚无导出</strong><span>训练完成并注册模型后可创建 ONNX 导出任务。</span></div>}
        {tab === "preview" && <div className="preview-card"><img src="/assets/bearing-overlay.png" alt="推理预览"/><div><strong>最近验证样本</strong><span>outer-race scratch · 0.94</span></div></div>}
      </Panel>
    </div>
  </div>;
}

function ModelsPage({ language }: { language: Language }) {
  const [tab, setTab] = useState("versions");
  const [selected, setSelected] = useState(models[0].id);
  const [compare, setCompare] = useState<string[]>([models[0].id, models[1].id]);
  const current = models.find((item) => item.id === selected) ?? models[0];
  const toggleCompare = (id: string) => setCompare((items) => items.includes(id) ? items.filter((item) => item !== id) : items.length < 3 ? [...items, id] : items);
  return <div className="page-stack">
    <Tabs label="模型库视图" value={tab} onChange={setTab} items={[{ id: "versions", label: "模型版本" }, { id: "report", label: "评估报告" }, { id: "compare", label: "模型对比" }, { id: "pipeline", label: "流水线记录" }]}/>
    {tab === "versions" && <div className="models-layout"><Panel title="模型版本" subtitle="轴承缺陷检测 · 已注册版本"><div className="model-list">{models.map((model) => <button key={model.id} className={selected === model.id ? "active" : ""} onClick={() => setSelected(model.id)}><div><strong>{model.version}</strong><span>{model.backend}</span><small>{model.createdAt}</small></div><span className="model-score"><b>{model.map50.toFixed(3)}</b><small>mAP50</small></span><StatusPill status={model.status} language={language}/></button>)}</div></Panel><ModelDetail model={current}/></div>}
    {tab === "report" && <Panel title="评估报告" subtitle={`${current.version} · 官方 Ultralytics val() 结果`}><div className="report-header"><div><span>固定测试集</span><strong>bearing-v3 / test · 426 images</strong></div><StatusPill status="healthy" language={language}>报告完整</StatusPill></div><div className="compact-metrics four"><Metric label="mAP50" value={current.map50.toFixed(3)} tone="success"/><Metric label="Precision" value={current.precision.toFixed(3)}/><Metric label="Recall" value={current.recall.toFixed(3)}/><Metric label="Latency" value={`${current.latency} ms`}/></div><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={[{ name: "outer scratch", ap: .88 }, { name: "inner pit", ap: .81 }, { name: "ball corrosion", ap: .79 }]}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name"/><YAxis domain={[0, 1]}/><Tooltip/><Bar dataKey="ap" fill="#2563eb" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div></Panel>}
    {tab === "compare" && <div className="two-column wide-left"><Panel title="选择对比版本" subtitle="最多选择 3 个版本"><div className="compare-select">{models.map((model) => <label key={model.id}><input type="checkbox" checked={compare.includes(model.id)} onChange={() => toggleCompare(model.id)}/><span>{model.version}</span><b>{model.map50.toFixed(3)}</b></label>)}</div></Panel><Panel title="关键指标对比" subtitle={`${compare.length} 个模型`}><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={models.filter((model) => compare.includes(model.id))}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="version"/><YAxis domain={[.6,.9]}/><Tooltip/><Legend/><Bar dataKey="map50" fill="#2563eb"/><Bar dataKey="precision" fill="#16a34a"/><Bar dataKey="recall" fill="#d97706"/></BarChart></ResponsiveContainer></div></Panel></div>}
    {tab === "pipeline" && <Panel title="流水线记录" subtitle="从训练到部署验证的可追踪链路"><div className="timeline">{[
      ["训练完成", "run-20260706-1340", "2026-07-06 16:48", "healthy"], ["官方评估", "val-report-v1.4.0", "2026-07-06 17:02", "healthy"], ["模型注册", "bearing-detector v1.4.0", "2026-07-06 17:18", "healthy"], ["ONNX 导出", "opset 17 · dynamic=false", "2026-07-06 17:24", "healthy"], ["部署验证", "RTX 4090 D · 8.4 ms", "2026-07-06 17:31", "warning"],
    ].map(([title, detail, time, status]) => <div key={title}><span className={`timeline-dot ${status}`}/><div><strong>{title}</strong><p>{detail}</p><small>{time}</small></div></div>)}</div></Panel>}
  </div>;
}

function ModelDetail({ model }: { model: ModelRecord }) {
  return <Panel title={`${model.version} 详情`} subtitle="已注册模型版本" action={<Button variant="primary" icon={RocketRegular}>进入部署验证</Button>}><div className="model-hero"><div className="model-cube"><BoxRegular/></div><div><h3>bearing-detector</h3><p>{model.backend}</p><code>sha256:7e92d4f0a31c…</code></div></div><div className="compact-metrics four"><Metric label="mAP50" value={model.map50.toFixed(3)} tone="success"/><Metric label="Precision" value={model.precision.toFixed(3)}/><Metric label="Recall" value={model.recall.toFixed(3)}/><Metric label="Latency" value={`${model.latency} ms`}/></div><dl className="detail-list"><div><dt>模型大小</dt><dd>{model.size}</dd></div><div><dt>来源数据集</dt><dd>bearing-v3</dd></div><div><dt>来源运行</dt><dd>run-20260706-1340</dd></div><div><dt>创建时间</dt><dd>{model.createdAt}</dd></div></dl></Panel>;
}

function DeploymentPage({ language, notify }: { language: Language; notify: (message: string) => void }) {
  const [tab, setTab] = useState("export");
  const [format, setFormat] = useState("onnx");
  const [exportStatus, setExportStatus] = useState<HealthStatus>("idle");
  const [inferenceStatus, setInferenceStatus] = useState<HealthStatus>("idle");
  const runExport = () => { setExportStatus("running"); window.setTimeout(() => { setExportStatus("completed"); notify("ONNX 导出已完成（演示）"); }, 1000); };
  const runInference = () => { setInferenceStatus("running"); window.setTimeout(() => { setInferenceStatus("completed"); notify("推理验证完成：检测到 1 个缺陷"); }, 900); };
  return <div className="page-stack">
    <FlowRail active={3}/>
    <Tabs label="部署工作区" value={tab} onChange={setTab} items={[{ id: "export", label: "模型导出" }, { id: "inference", label: "推理验证" }]}/>
    {tab === "export" ? <div className="deployment-grid">
      <Panel title="导出设置" subtitle="bearing-detector · v1.4.0">
        <div className="form-stack"><label>目标格式<select value={format} onChange={(e) => setFormat(e.target.value)}><option value="onnx">ONNX</option><option value="tensorrt">TensorRT</option><option value="ncnn">NCNN</option></select></label><label>输入尺寸<div className="input-pair"><input defaultValue="640"/><span>×</span><input defaultValue="640"/></div></label><label><span>动态批次</span><input type="checkbox"/></label><label>输出目录<input defaultValue="exports\\bearing-detector-v1.4.0"/></label></div>
        <div className="export-boundary"><WarningRegular/><p>{format === "onnx" ? "首个 ONNX 由官方 Ultralytics 导出；后续验证由 AITrain C++ 运行时执行。" : format === "tensorrt" ? "TensorRT 需要当前受支持硬件与交付证据。" : "NCNN 仅适用于当前已支持的 YOLO 打包推理边界。"}</p></div>
        <Button variant="primary" icon={ArrowDownloadRegular} onClick={runExport} disabled={exportStatus === "running"}>{exportStatus === "running" ? "正在导出…" : "开始导出"}</Button>
      </Panel>
      <Panel title="格式矩阵" subtitle="当前模型与运行时支持范围"><div className="format-matrix"><div className={format === "onnx" ? "active" : ""}><strong>ONNX</strong><StatusPill status="healthy" language={language}>支持</StatusPill><span>ONNX Runtime · CPU/CUDA</span></div><div className={format === "tensorrt" ? "active" : ""}><strong>TensorRT</strong><StatusPill status="warning" language={language}>需证据</StatusPill><span>NVIDIA GPU · 当前工作站</span></div><div className={format === "ncnn" ? "active" : ""}><strong>NCNN</strong><StatusPill status="healthy" language={language}>支持</StatusPill><span>CPU/Vulkan · 静态输入</span></div></div><div className={`run-result ${exportStatus}`}><span>{exportStatus === "completed" ? <CheckmarkCircleRegular/> : exportStatus === "running" ? <ArrowSyncRegular className="spin"/> : <DocumentRegular/>}</span><div><strong>{exportStatus === "completed" ? "导出完成" : exportStatus === "running" ? "正在运行官方导出适配器" : "等待导出任务"}</strong><p>{exportStatus === "completed" ? "bearing-detector-v1.4.0.onnx · 11.8 MB" : "导出后将生成格式、校验和与来源记录。"}</p></div></div></Panel>
    </div> : <div className="inference-layout">
      <Panel title="验证输入" subtitle="选择样本并使用打包运行时执行"><div className="drop-zone"><ImageRegular/><strong>BRG_02761.jpg</strong><span>1536 × 1152 · 1.8 MB</span><Button icon={OpenRegular}>选择图像</Button></div><dl className="detail-list"><div><dt>运行时</dt><dd>ONNX Runtime CUDA</dd></div><div><dt>模型</dt><dd>bearing-detector-v1.4.0.onnx</dd></div><div><dt>置信度阈值</dt><dd>0.25</dd></div></dl><Button variant="primary" icon={PlayRegular} onClick={runInference} disabled={inferenceStatus === "running"}>{inferenceStatus === "running" ? "正在推理…" : "运行推理验证"}</Button></Panel>
      <Panel title="Overlay 预览" subtitle="AITrain C++ 打包推理输出" action={<StatusPill status={inferenceStatus === "completed" ? "healthy" : inferenceStatus} language={language}/>}><div className="overlay-stage"><img src={inferenceStatus === "completed" ? "/assets/bearing-overlay.png" : "/assets/bearing-defect.png"} alt="轴承推理 overlay"/>{inferenceStatus !== "completed" && <span>运行验证后显示检测 overlay</span>}</div><div className="result-summary"><Metric label="检测数" value={inferenceStatus === "completed" ? "1" : "—"}/><Metric label="最高置信度" value={inferenceStatus === "completed" ? "0.94" : "—"}/><Metric label="端到端延迟" value={inferenceStatus === "completed" ? "8.6 ms" : "—"}/></div></Panel>
      <Panel title="结果摘要" subtitle="结构化推理输出"><div className="result-table"><div className="result-head"><span>类别</span><span>置信度</span><span>位置</span></div>{inferenceStatus === "completed" ? <div><strong>outer-race scratch</strong><span>0.94</span><code>[604, 105, 718, 184]</code></div> : <p>暂无结果</p>}</div><div className="evidence-card"><ShieldCheckmarkRegular/><div><strong>验证链路</strong><span>模型哈希、运行时版本、输入哈希和结果报告将一起记录。</span></div></div></Panel>
    </div>}
  </div>;
}

function EnvironmentPage({ language, notify }: { language: Language; notify: (message: string) => void }) {
  const [tab, setTab] = useState("runtime");
  const [checking, setChecking] = useState(false);
  const runCheck = () => { setChecking(true); window.setTimeout(() => { setChecking(false); notify("环境检查完成：5 项通过，1 项警告"); }, 900); };
  return <div className="page-stack">
    <div className="toolbar"><Tabs label="环境视图" value={tab} onChange={setTab} items={[{ id: "runtime", label: "运行环境" }, { id: "evidence", label: "交付证据" }]}/><Button variant="primary" icon={ArrowSyncRegular} onClick={runCheck} disabled={checking}>{checking ? "正在检查…" : "运行全部检查"}</Button></div>
    {tab === "runtime" ? <div className="environment-grid"><Panel title="环境检查" subtitle="本地训练与部署依赖"><div className="check-list">{environmentChecks.map((check) => <div key={check.id}><span className={`check-icon ${check.status}`}>{check.status === "healthy" ? <CheckmarkCircleRegular/> : <WarningRegular/>}</span><div><strong>{check.name}</strong><span>{check.detail}</span></div><b>{check.version}</b><StatusPill status={check.status} language={language}/></div>)}</div></Panel><Panel title="工作站摘要" subtitle="当前 Worker 报告"><div className="machine-card"><div className="machine-icon"><PulseRegular/></div><div><strong>TRAIN-STATION-01</strong><span>Windows 11 Pro · x64</span></div><StatusPill status="healthy" language={language}>在线</StatusPill></div><div className="compact-metrics two"><Metric label="GPU 显存" value="21.3 / 23.4 GB" tone="warning"/><Metric label="GPU 温度" value="68°C"/><Metric label="项目磁盘" value="462 GB"/><Metric label="Worker 延迟" value="24 ms" tone="success"/></div><div className="boundary-note"><ShieldCheckmarkRegular/><p>环境状态仅描述返回的本机检查证据，不扩展为未验证硬件或清洁 Windows 验收结论。</p></div></Panel></div> : <Panel title="交付证据" subtitle="按运行时与交付包组织的可审计记录"><div className="evidence-table"><div className="evidence-head"><span>证据</span><span>范围</span><span>生成时间</span><span>状态</span></div>{[
      ["ONNX Runtime CUDA 验证报告", "bearing-detector v1.4.0", "2026-07-09 17:52", "healthy"], ["模型导出来源记录", "Ultralytics · opset 17", "2026-07-09 17:34", "healthy"], ["TensorRT package-root rerun", "RTX 4090 D", "尚未生成", "warning"], ["Worker 环境快照", "TRAIN-STATION-01", "2026-07-10 10:41", "healthy"],
    ].map(([name, scope, time, status]) => <div key={name}><span><DocumentRegular/><strong>{name}</strong></span><span>{scope}</span><span>{time}</span><StatusPill status={status as HealthStatus} language={language}/></div>)}</div></Panel>}
  </div>;
}

function SettingsPage({ language, notify }: { language: Language; notify: (message: string) => void }) {
  const [tab, setTab] = useState("capabilities");
  const [appLanguage, setAppLanguage] = useState(language);
  return <div className="page-stack">
    <Tabs label="系统设置视图" value={tab} onChange={setTab} items={[{ id: "capabilities", label: "内置能力" }, { id: "application", label: "应用设置" }]}/>
    {tab === "capabilities" ? <Panel title="内置能力矩阵" subtitle="基于当前产品注册表的后端与运行时边界"><div className="capability-table"><div className="capability-head"><span>能力</span><span>产品入口</span><span>部署范围</span><span>状态</span></div>{capabilityRows.map((row) => <div key={row[0]}>{row.slice(0,3).map((cell) => <span key={cell}>{cell}</span>)}<StatusPill status="healthy" language={language}>{row[3]}</StatusPill></div>)}</div><div className="registry-note"><ShieldCheckmarkRegular/><div><strong>能力声明遵循当前注册表</strong><span>不展示已移除的诊断路径；异常检测与 OCR 保持各自官方运行时和验收边界。</span></div></div></Panel> : <div className="settings-grid"><Panel title="常规" subtitle="语言、项目与启动行为"><div className="settings-form"><label><span><strong>界面语言</strong><small>切换导航、表单和操作文案</small></span><select value={appLanguage} onChange={(e) => setAppLanguage(e.target.value as Language)}><option value="zh">简体中文</option><option value="en">English</option></select></label><label><span><strong>默认项目目录</strong><small>新项目的默认保存位置</small></span><div className="inline-input"><input defaultValue="D:\\AITrain"/><Button icon={FolderRegular}>浏览</Button></div></label><label><span><strong>启动时恢复项目</strong><small>自动打开最近使用的项目</small></span><input type="checkbox" defaultChecked className="switch"/></label><label><span><strong>任务完成通知</strong><small>显示本地桌面通知</small></span><input type="checkbox" defaultChecked className="switch"/></label></div></Panel><Panel title="授权与本地路径" subtitle="仅影响当前工作站"><div className="license-card"><ShieldCheckmarkRegular/><div><strong>AITrain Studio Local</strong><span>授权有效 · 绑定 TRAIN-STATION-01</span></div><Button>管理授权</Button></div><div className="path-list"><label>Worker 可执行文件<input defaultValue="C:\\Program Files\\AITrain\\worker.exe"/></label><label>Python 环境<input defaultValue="D:\\AITrain\\envs\\official-backends"/></label><label>模型缓存<input defaultValue="D:\\AITrain\\cache\\models"/></label></div><div className="settings-actions"><Button onClick={() => notify("设置已恢复为预览默认值")}>恢复默认</Button><Button variant="primary" onClick={() => notify("应用设置已保存（内存状态）")}>保存设置</Button></div></Panel></div>}
  </div>;
}

function Inspector({ route, language, project, onClose, navigate }: { route: WorkspaceRoute; language: Language; project: ProjectState; onClose: () => void; navigate: (route: WorkspaceRoute) => void }) {
  const details: Record<WorkspaceRoute, { title: string; subtitle: string; status: HealthStatus }> = {
    overview: { title: project.name, subtitle: "项目工作上下文", status: "healthy" }, projects: { title: project.name, subtitle: "当前项目", status: "healthy" }, datasets: { title: "bearing-v3", subtitle: "当前数据集", status: "healthy" },
    training: { title: "yolo11n-bearing-v3", subtitle: "当前运行", status: "running" }, tasks: { title: "TR-1048", subtitle: "选中任务", status: "running" }, models: { title: "bearing-detector v1.5.0", subtitle: "候选模型", status: "running" },
    deployment: { title: "ONNX Runtime CUDA", subtitle: "部署目标", status: "healthy" }, environment: { title: "TRAIN-STATION-01", subtitle: "当前工作站", status: "warning" }, settings: { title: "AITrain Studio Local", subtitle: "应用配置", status: "healthy" },
  };
  const current = details[route];
  return <><header className="inspector-header"><div><span>检查器</span><strong>{current.subtitle}</strong></div><button className="icon-button" onClick={onClose} aria-label="关闭检查器"><DismissRegular/></button></header><div className="inspector-content"><div className="inspector-identity"><span className="identity-icon">{route === "datasets" ? <DatabaseRegular/> : route === "training" ? <BrainCircuitRegular/> : route === "environment" ? <PulseRegular/> : <FolderRegular/>}</span><div><h2>{current.title}</h2><StatusPill status={current.status} language={language}/></div></div><dl className="detail-list inspector-list"><div><dt>项目</dt><dd>{project.name}</dd></div><div><dt>任务类型</dt><dd>{project.task}</dd></div><div><dt>训练后端</dt><dd>{project.backend}</dd></div><div><dt>项目目录</dt><dd className="mono">{project.root}</dd></div></dl><div className="inspector-section"><h3>运行资源</h3><div className="resource-row"><span>GPU</span><strong>91%</strong></div><div className="progress"><i style={{ width: "91%" }}/></div><div className="resource-row"><span>显存</span><strong>21.3 / 23.4 GB</strong></div><div className="progress memory"><i style={{ width: "91%" }}/></div></div><div className="inspector-section"><h3>快捷入口</h3><button onClick={() => navigate("tasks")}><TaskListSquareLtrRegular/><span>查看任务与产物</span><ChevronRightRegular/></button><button onClick={() => navigate("environment")}><PulseRegular/><span>检查运行环境</span><ChevronRightRegular/></button></div></div><footer className="inspector-footer"><span className="online-dot"/><div><strong>Worker 已连接</strong><small>localhost:49720 · 24 ms</small></div></footer></>;
}

export default App;
