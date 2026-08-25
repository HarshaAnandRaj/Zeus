export const REGION_IDS = [
  "gate",
  "sensory",
  "language",
  "executive",
  "memory",
  "action",
  "drive",
] as const;

export type RegionId = (typeof REGION_IDS)[number];

export type TaskKind =
  | "talk"
  | "think"
  | "remember"
  | "act"
  | "rest"
  | "explore"
  | "social";

export type DriveId = "curiosity" | "social" | "rest" | "growth" | "purpose";

export interface RegionDef {
  id: RegionId;
  name: string;
  short: string;
  role: string;
  x: number;
  y: number;
}

export interface Neuron {
  id: number;
  region: RegionId;
  x: number;
  y: number;
  v: number;
  threshold: number;
  spiked: boolean;
  lastSpike: number;
  firing: number;
}

export interface Synapse {
  id: number;
  from: number;
  to: number;
  weight: number;
  uses: number;
  lastUse: number;
}

export interface MemoryTrace {
  id: string;
  content: string;
  strength: number;
  createdAt: number;
  lastRecall: number;
  tags: RegionId[];
}

export interface Goal {
  id: string;
  content: string;
  progress: number;
  createdAt: number;
}

export interface Note {
  id: string;
  title: string;
  body: string;
  createdAt: number;
}

export interface DriveState {
  curiosity: number;
  social: number;
  rest: number;
  growth: number;
  purpose: number;
}

export interface Pulse {
  from: number;
  to: number;
  born: number;
  life: number;
  strength: number;
}

export interface MindEvent {
  t: number;
  kind: string;
  text: string;
}

export interface Organism {
  seed: number;
  name: string;
  bornAt: number;
  ticks: number;
  neurons: Neuron[];
  synapses: Synapse[];
  regionMask: Record<RegionId, number>;
  drives: DriveState;
  memories: MemoryTrace[];
  goals: Goal[];
  notes: Note[];
  events: MindEvent[];
  lastTask: TaskKind;
  energy: number;
  selectivity: number;
  lastHumanAt: number;
  lastSpeechAt: number;
  lastAutonomousAt: number;
  named: boolean;
}

export interface ChatMessage {
  id: string;
  role: "human" | "soma" | "system";
  text: string;
  at: number;
  via: "cortex" | "native" | "system";
}

export interface Thought {
  id: string;
  text: string;
  at: number;
  region: RegionId;
}

export interface CortexAction {
  type: "remember" | "goal" | "note" | "grow" | "rest" | "rename";
  content?: string;
  title?: string;
  from?: RegionId;
  to?: RegionId;
  name?: string;
}

export interface CortexResult {
  speech: string;
  thought: string;
  actions: CortexAction[];
  mood: string;
  focusRegion: RegionId;
}

export interface MindSnapshot {
  name: string;
  ticks: number;
  drives: DriveState;
  gatedRegions: RegionId[];
  topPathways: { from: RegionId; to: RegionId; weight: number }[];
  memories: string[];
  goals: string[];
  notes: { title: string; body: string }[];
  lastThoughts: string[];
  lastUser: string;
  mode: "reply" | "think";
  named: boolean;
}

export const REGIONS: RegionDef[] = [
  {
    id: "gate",
    name: "Gate",
    short: "GAT",
    role: "Selective attention. Opens only the pathways a task needs.",
    x: 0.5,
    y: 0.1,
  },
  {
    id: "sensory",
    name: "Sensory",
    short: "SEN",
    role: "Encodes what you say and what arrives from the world.",
    x: 0.13,
    y: 0.34,
  },
  {
    id: "language",
    name: "Language",
    short: "LNG",
    role: "Turns internal state into speech.",
    x: 0.87,
    y: 0.34,
  },
  {
    id: "executive",
    name: "Executive",
    short: "EXE",
    role: "Plans, holds goals, sequences action.",
    x: 0.5,
    y: 0.5,
  },
  {
    id: "drive",
    name: "Drive",
    short: "DRV",
    role: "Homeostatic wants: curiosity, company, rest, growth.",
    x: 0.16,
    y: 0.8,
  },
  {
    id: "memory",
    name: "Memory",
    short: "MEM",
    role: "Stores traces. Recalls by resonance, not search.",
    x: 0.5,
    y: 0.9,
  },
  {
    id: "action",
    name: "Action",
    short: "ACT",
    role: "Does things: notes, goals, growth, rest.",
    x: 0.84,
    y: 0.8,
  },
];


export const REGION_MAP: Record<RegionId, RegionDef> = Object.fromEntries(
  REGIONS.map((r) => [r.id, r]),
) as Record<RegionId, RegionDef>;

export const TASK_ROUTES: Record<TaskKind, RegionId[]> = {
  talk: ["gate", "sensory", "language", "memory"],
  think: ["gate", "executive", "memory", "language"],
  remember: ["gate", "memory", "sensory", "executive"],
  act: ["gate", "executive", "action", "drive"],
  rest: ["gate", "drive", "memory"],
  explore: ["gate", "drive", "executive", "memory"],
  social: ["gate", "sensory", "language", "drive"],
};

export const DRIVE_LABELS: Record<DriveId, string> = {
  curiosity: "Curious",
  social: "Company",
  rest: "Rest",
  growth: "Growth",
  purpose: "Purpose",
};

