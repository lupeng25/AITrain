import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import App from "./App";

afterEach(() => {
  cleanup();
  window.location.hash = "";
});

function renderAt(route: string) {
  window.location.hash = `/${route}`;
  return render(<App />);
}

describe("AITrain Studio HTML 预览", () => {
  it("可在九个一级工作区之间导航", async () => {
    renderAt("training");
    await userEvent.click(screen.getByRole("button", { name: "数据集" }));
    expect(await screen.findByRole("heading", { name: "数据集", level: 1 })).toBeInTheDocument();
    expect(window.location.hash).toBe("#/datasets");
  });

  it("支持中英文切换", async () => {
    renderAt("training");
    await userEvent.click(screen.getByTestId("language-switch"));
    expect(screen.getByRole("heading", { name: "Training experiments", level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Datasets" })).toBeInTheDocument();
  });

  it("训练状态可通过确认弹窗停止", async () => {
    renderAt("training");
    await userEvent.click(screen.getByTestId("training-toggle"));
    const dialog = screen.getByRole("dialog", { name: "停止当前训练？" });
    expect(dialog).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole("button", { name: "停止训练" }));
    expect(screen.getByTestId("training-toggle")).toHaveTextContent("启动训练");
  });

  it("数据校验展示完成状态", async () => {
    renderAt("datasets");
    await userEvent.click(screen.getByRole("button", { name: "校验数据集" }));
    expect(screen.getByRole("button", { name: "正在校验…" })).toBeDisabled();
    expect(await screen.findByText("数据集可用于训练", {}, { timeout: 1600 })).toBeInTheDocument();
  });

  it("任务筛选会过滤表格", async () => {
    renderAt("tasks");
    await userEvent.click(screen.getByRole("button", { name: "运行中" }));
    expect(screen.getAllByText("yolo11n-bearing-v3").length).toBeGreaterThan(0);
    expect(screen.queryByText("bearing-v3 数据校验")).not.toBeInTheDocument();
  });

  it("嵌套页签和检查器折叠可操作", async () => {
    renderAt("deployment");
    await userEvent.click(screen.getByRole("tab", { name: "推理验证" }));
    expect(screen.getByRole("heading", { name: "Overlay 预览" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "检查器" }));
    await waitFor(() => expect(screen.queryByText("当前工作站")).not.toBeInTheDocument());
  });
});
