import { describe, it, expect, beforeEach } from "vitest";
import { useToastStore } from "../../store/toast";

function resetStore() {
  useToastStore.setState({ toasts: [] });
}

beforeEach(resetStore);

describe("useToastStore", () => {
  it("starts with empty toasts", () => {
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it("adds a toast with auto-incrementing id", () => {
    useToastStore.getState().addToast({ type: "success", text: "Done" });
    useToastStore.getState().addToast({ type: "error", text: "Fail" });

    const { toasts } = useToastStore.getState();
    expect(toasts).toHaveLength(2);
    expect(toasts[0].id).toBe(1);
    expect(toasts[0].type).toBe("success");
    expect(toasts[0].text).toBe("Done");
    expect(toasts[1].id).toBe(2);
    expect(toasts[1].type).toBe("error");
  });

  it("dismisses a toast by id", () => {
    const store = useToastStore.getState();
    store.addToast({ type: "info", text: "A" });
    store.addToast({ type: "info", text: "B" });
    store.addToast({ type: "info", text: "C" });

    const ids = useToastStore.getState().toasts.map((t) => t.id);
    useToastStore.getState().dismissToast(ids[1]);

    const remaining = useToastStore.getState().toasts.map((t) => t.id);
    expect(remaining).toHaveLength(2);
    expect(remaining).toEqual([ids[0], ids[2]]);
  });

  it("dismissing non-existent id is a no-op", () => {
    useToastStore.getState().addToast({ type: "info", text: "Only" });
    useToastStore.getState().dismissToast(999);

    expect(useToastStore.getState().toasts).toHaveLength(1);
  });
});
