import { create } from "zustand";
import { EsnpnRuntime } from "./runtime";
import { clearPersisted, freshMind, loadPersisted, savePersisted } from "./persist";
import { toSnapshot } from "./snapshot";
import { getAiStatus, runLanguageCortex } from "./cortex";
import type { RegionId } from "./types";

let runtime: EsnpnRuntime | null = null;
let saveTimer: ReturnType<typeof setTimeout> | null = null;
let tickTimer: ReturnType<typeof setInterval> | null = null;
let hydrated = false;

export function getRuntime(): EsnpnRuntime {
  if (!runtime) runtime = new EsnpnRuntime();
  return runtime;
}

function persistSoon() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const rt = getRuntime();
    savePersisted({
      organism: rt.organism,
      messages: rt.messages,
      thoughts: rt.thoughts,
      woken: rt.woken,
      autonomy: rt.autonomy,
    });
  }, 400);
}

export interface MindStore {
  ready: boolean;
  version: number;
  woken: boolean;
  autonomy: boolean;
  thinking: boolean;
  aiAvailable: boolean | null;
  aiError: string | null;
  selectedRegion: RegionId | null;
  panel: "talk" | "inner" | "work" | "arch";
  hydrate: () => void;
  wake: () => void;
  setAutonomy: (v: boolean) => void;
  selectRegion: (id: RegionId | null) => void;
  setPanel: (p: MindStore["panel"]) => void;
  poke: (id: RegionId) => void;
  send: (text: string) => Promise<void>;
  think: () => Promise<void>;
  reset: () => void;
}

export const useMind = create<MindStore>((set, get) => ({
  ready: false,
  version: 0,
  woken: false,
  autonomy: true,
  thinking: false,
  aiAvailable: null,
  aiError: null,
  selectedRegion: null,
  panel: "talk",

  hydrate: () => {
    if (hydrated) return;
    hydrated = true;
    const saved = loadPersisted();
    runtime = new EsnpnRuntime(saved?.organism);
    if (saved) {
      runtime.messages = saved.messages ?? [];
      runtime.thoughts = saved.thoughts ?? [];
      runtime.woken = saved.woken;
      runtime.autonomy = saved.autonomy ?? true;
    }
    runtime.subscribe(() => {
      set({
        version: get().version + 1,
        woken: runtime!.woken,
        autonomy: runtime!.autonomy,
      });
      persistSoon();
    });
    set({
      ready: true,
      woken: runtime.woken,
      autonomy: runtime.autonomy,
      version: 1,
    });
    if (tickTimer) clearInterval(tickTimer);
    tickTimer = setInterval(() => {
      getRuntime().tick(1);
    }, 80);
    void getAiStatus()
      .then((r) => set({ aiAvailable: r.available }))
      .catch(() => set({ aiAvailable: false }));
  },

  wake: () => {
    getRuntime().wake();
    set({ woken: true, panel: "talk" });
  },

  setAutonomy: (v) => {
    getRuntime().autonomy = v;
    set({ autonomy: v });
    persistSoon();
  },

  selectRegion: (id) => set({ selectedRegion: id }),

  setPanel: (p) => set({ panel: p }),

  poke: (id) => {
    const rt = getRuntime();
    if (!rt.woken) rt.wake();
    rt.pokeRegion(id);
    set({ selectedRegion: id });
  },

  send: async (text) => {
    const trimmed = text.trim();
    if (!trimmed || get().thinking) return;
    const rt = getRuntime();
    if (!rt.woken) rt.wake();
    const task = rt.ingestHuman(trimmed);
    set({ thinking: true, aiError: null, panel: "talk" });
    try {
      if (get().aiAvailable === false) {
        rt.speakNative(trimmed, task);
        return;
      }
      const snap = toSnapshot(rt.organism, rt.thoughts, trimmed, "reply");
      const res = await runLanguageCortex({ data: snap });
      if (!res.ok) {
        set({ aiAvailable: false, aiError: res.error });
        rt.speakNative(trimmed, task);
        return;
      }
      set({ aiAvailable: true });
      rt.applyCortex(res.result);
    } catch (err) {
      set({ aiError: err instanceof Error ? err.message : "Language cortex failed" });
      rt.speakNative(trimmed, task);
    } finally {
      set({ thinking: false });
    }
  },

  think: async () => {
    if (get().thinking) return;
    const rt = getRuntime();
    if (!rt.woken) rt.wake();
    rt.stimulate("think");
    set({ thinking: true, aiError: null });
    try {
      if (get().aiAvailable === false) {
        rt.speakNative(null, "think");
        return;
      }
      const snap = toSnapshot(rt.organism, rt.thoughts, "", "think");
      const res = await runLanguageCortex({ data: snap });
      if (!res.ok) {
        set({ aiAvailable: false, aiError: res.error });
        rt.speakNative(null, "think");
        return;
      }
      set({ aiAvailable: true });
      rt.applyCortex(res.result);
    } catch (err) {
      set({ aiError: err instanceof Error ? err.message : "Language cortex failed" });
      rt.speakNative(null, "think");
    } finally {
      set({ thinking: false });
    }
  },

  reset: () => {
    clearPersisted();
    const fresh = freshMind();
    const rt = getRuntime();
    rt.organism = fresh.organism;
    rt.messages = [];
    rt.thoughts = [];
    rt.woken = false;
    rt.autonomy = true;
    rt.pulses = [];
    rt.emit();
    set({ woken: false, selectedRegion: null, panel: "talk", aiError: null });
  },
}));
