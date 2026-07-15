import "@testing-library/jest-dom/vitest";

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(window, "ResizeObserver", { value: ResizeObserverMock });

Object.defineProperty(HTMLElement.prototype, "getBoundingClientRect", {
  configurable: true,
  value: () => ({ width: 800, height: 300, top: 0, left: 0, bottom: 300, right: 800, x: 0, y: 0, toJSON: () => ({}) }),
});
