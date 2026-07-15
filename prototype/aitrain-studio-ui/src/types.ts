export type Language = "zh" | "en";

export type WorkspaceRoute =
  | "overview"
  | "projects"
  | "datasets"
  | "training"
  | "tasks"
  | "models"
  | "deployment"
  | "environment"
  | "settings";

export type HealthStatus = "healthy" | "warning" | "blocked" | "idle" | "running" | "completed";

export interface ProjectState {
  id: string;
  name: string;
  task: string;
  backend: string;
  root: string;
  updatedAt: string;
  datasets: number;
  experiments: number;
  models: number;
}

export interface DatasetState {
  id: string;
  name: string;
  split: string;
  sampleCount: number;
  annotatedCount: number;
  classes: Array<{ name: string; count: number; color: string }>;
  status: HealthStatus;
  updatedAt: string;
  path: string;
}

export interface TrainingRun {
  id: string;
  name: string;
  backend: string;
  model: string;
  status: HealthStatus;
  epoch: number;
  totalEpochs: number;
  map50: number;
  precision: number;
  recall: number;
  elapsed: string;
  eta: string;
  startedAt: string;
}

export interface TaskRecord {
  id: string;
  name: string;
  type: string;
  status: HealthStatus;
  progress: number;
  startedAt: string;
  duration: string;
}

export interface ArtifactRecord {
  id: string;
  name: string;
  kind: string;
  format: string;
  size: string;
  createdAt: string;
  path: string;
}

export interface ModelRecord {
  id: string;
  version: string;
  backend: string;
  map50: number;
  precision: number;
  recall: number;
  latency: number;
  size: string;
  status: HealthStatus;
  createdAt: string;
}

export interface EnvironmentCheck {
  id: string;
  name: string;
  version: string;
  status: HealthStatus;
  detail: string;
}

export interface MetricPoint {
  epoch: number;
  map50: number;
  precision: number;
  recall: number;
  loss: number;
}
