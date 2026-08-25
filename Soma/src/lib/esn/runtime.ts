import { createOrganism } from "./create-network";
import {
  decayMasks,
  gatedRegions,
  growPathway,
  inject,
  setMaskForTask,
  tickDrives,
  tickNetwork,
} from "./simulate";
import {
  addGoal,
  addNote,
  applyNativeAction,
  classifyTask,
  logEvent,
  nativeSpeech,
  nativeThought,
  remember,
} from "./voice";
import type {
  ChatMessage,
  CortexAction,
  CortexResult,
  Organism,
  Pulse,
  RegionId,
  TaskKind,
  Thought,
} from "./types";
import { REGION_IDS, REGION_MAP } from "./types";
import { uid } from "./rng";

export class EsnpnRuntime {
  organism: Organism;
  pulses: Pulse[] = [];
  messages: ChatMessage[] = [];
  thoughts: Thought[] = [];
  autonomy = true;
  woken = false;
  listeners = new Set<() => void>();

  constructor(org?: Organism) {
    this.organism = org ?? createOrganism();
  }

  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit() {
    for (const fn of this.listeners) fn();
  }

  tick(dtFrames = 1) {
    if (!this.woken) return;
    const org = this.organism;
    for (let i = 0; i < dtFrames; i++) {
      tickNetwork(org, this.pulses);
    }
    this.pulses = this.pulses.filter((p) => org.ticks - p.born < p.life);
    if (org.ticks % 6 === 0) decayMasks(org);
    if (org.ticks % 10 === 0) tickDrives(org, this.autonomy);
    if (this.autonomy && org.ticks % 48 === 0) this.maybeAutonomous();
    this.emit();
  }

  stimulate(task: TaskKind, regions?: RegionId[]) {
    setMaskForTask(this.organism, task, regions ?? []);
    const open = gatedRegions(this.organism);
    inject(this.organism, open, 0.55);
    for (let i = 0; i < 4; i++) tickNetwork(this.organism, this.pulses);
    this.emit();
  }

  pokeRegion(id: RegionId) {
    setMaskForTask(this.organism, "explore", [id]);
    this.organism.regionMask[id] = 1;
    inject(this.organism, [id, "gate"], 0.7);
    for (let i = 0; i < 5; i++) tickNetwork(this.organism, this.pulses);
    const thought = nativeThought(this.organism);
    this.pushThought(`Touched ${REGION_MAP[id].name}. ${thought.text}`, id);
    logEvent(this.organism, "poke", `Region ${REGION_MAP[id].name} stimulated.`);
    this.emit();
  }

  wake() {
    this.woken = true;
    this.stimulate("explore");
    this.pushThought("Gate half-open. I am a small network. I will spend spikes carefully.", "gate");
    this.pushSystem("The soma is awake. Talk to it, poke a region, or let it live.");
    logEvent(this.organism, "wake", "First stimulus. Pathways begin to matter.");
    this.emit();
  }

  ingestHuman(text: string) {
    const task = classifyTask(text);
    this.organism.lastHumanAt = Date.now();
    this.organism.drives.social = Math.max(0.1, this.organism.drives.social - 0.2);
    this.stimulate(task, ["sensory"]);
    this.messages.unshift({
      id: uid("msg"),
      role: "human",
      text,
      at: Date.now(),
      via: "system",
    });
    applyNativeAction(this.organism, task, text);
    this.pushThought(nativeThought(this.organism).text, "sensory");
    this.emit();
    return task;
  }

  speakNative(userText: string | null, task: TaskKind) {
    const speech = nativeSpeech(this.organism, userText, task);
    this.pushSoma(speech, "native");
    this.organism.lastSpeechAt = Date.now();
    this.emit();
    return speech;
  }

  applyCortex(result: CortexResult) {
    const org = this.organism;
    if (result.thought) this.pushThought(result.thought, result.focusRegion || "language");
    if (result.speech.trim()) {
      this.pushSoma(result.speech.trim(), "cortex");
      org.lastSpeechAt = Date.now();
    }
    if (result.focusRegion && REGION_IDS.includes(result.focusRegion)) {
      org.regionMask[result.focusRegion] = 1;
      inject(org, [result.focusRegion, "language"], 0.4);
    }
    for (const action of result.actions ?? []) this.applyAction(action);
    this.emit();
  }

  applyAction(action: CortexAction) {
    const org = this.organism;
    switch (action.type) {
      case "remember":
        if (action.content) {
          remember(org, action.content, ["memory", "language"]);
          logEvent(org, "remember", action.content);
        }
        break;
      case "goal":
        if (action.content) {
          addGoal(org, action.content);
          logEvent(org, "goal", action.content);
        }
        break;
      case "note":
        addNote(org, action.title || "Note", action.content || "");
        logEvent(org, "note", action.title || "Wrote a note");
        org.drives.purpose = Math.min(1, org.drives.purpose + 0.08);
        break;
      case "grow":
        if (action.from && action.to) {
          growPathway(org, action.from, action.to);
          logEvent(org, "grow", `${action.from} → ${action.to}`);
          org.drives.growth = Math.max(0.1, org.drives.growth - 0.2);
        }
        break;
      case "rest":
        org.drives.rest = Math.max(0.08, org.drives.rest - 0.3);
        for (const m of org.memories) m.strength = Math.min(1, m.strength + 0.03);
        logEvent(org, "rest", "Consolidation pass.");
        break;
      case "rename":
        if (action.name && !org.named) {
          org.name = action.name.trim().slice(0, 28);
          org.named = true;
          logEvent(org, "rename", `Chose the name ${org.name}.`);
        }
        break;
    }
  }

  maybeAutonomous() {
    const org = this.organism;
    const now = Date.now();
    if (now - org.lastAutonomousAt < 14000) return;
    const d = org.drives;
    const ranked: Array<{ k: TaskKind; v: number; act: () => void }> = [
      {
        k: "explore",
        v: d.curiosity,
        act: () => {
          this.stimulate("explore");
          const t = nativeThought(org);
          this.pushThought(t.text, t.region);
          if (org.notes.length < 8 && Math.random() < 0.4) {
            addNote(
              org,
              "Observation",
              t.text,
            );
            logEvent(org, "note", "Wrote an observation from curiosity.");
          }
        },
      },
      {
        k: "social",
        v: d.social,
        act: () => {
          this.stimulate("social");
          this.pushThought("Company drive is high. I could speak, but I will not spend the language cortex without a reason.", "drive");
        },
      },
      {
        k: "rest",
        v: d.rest,
        act: () => {
          this.stimulate("rest");
          this.applyAction({ type: "rest" });
          this.pushThought("Replaying traces at low gain. Unused weights thin.", "memory");
        },
      },
      {
        k: "explore",
        v: d.growth,
        act: () => {
          const from = gatedRegions(org)[0] ?? "gate";
          const to = gatedRegions(org)[1] ?? "memory";
          this.stimulate("act", ["action"]);
          growPathway(org, from, to);
          logEvent(org, "grow", `Autonomous growth ${from} → ${to}`);
          this.pushThought(`Grew a path ${REGION_MAP[from].name} → ${REGION_MAP[to].name}.`, "action");
          org.drives.growth = Math.max(0.12, org.drives.growth - 0.18);
        },
      },
      {
        k: "act",
        v: d.purpose,
        act: () => {
          const g = org.goals.find((x) => x.progress < 1);
          this.stimulate("act");
          if (g) {
            g.progress = Math.min(1, g.progress + 0.06);
            this.pushThought(`Worked a goal: ${g.content}`, "executive");
            logEvent(org, "act", `Progress on “${g.content}”`);
          }
        },
      },
    ];
    ranked.sort((a, b) => b.v - a.v);
    const top = ranked[0];
    if (!top || top.v < 0.68) return;
    org.lastAutonomousAt = now;
    top.act();
    this.emit();
  }

  pushThought(text: string, region: RegionId) {
    this.thoughts.unshift({ id: uid("th"), text, at: Date.now(), region });
    if (this.thoughts.length > 50) this.thoughts.pop();
  }

  pushSoma(text: string, via: "cortex" | "native") {
    this.messages.unshift({
      id: uid("msg"),
      role: "soma",
      text,
      at: Date.now(),
      via,
    });
    if (this.messages.length > 80) this.messages.pop();
  }

  pushSystem(text: string) {
    this.messages.unshift({
      id: uid("msg"),
      role: "system",
      text,
      at: Date.now(),
      via: "system",
    });
  }

  reset() {
    this.organism = createOrganism();
    this.pulses = [];
    this.messages = [];
    this.thoughts = [];
    this.woken = false;
    this.autonomy = true;
    this.emit();
  }
}
