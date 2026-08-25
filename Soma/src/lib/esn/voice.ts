import { strongestPathways, gatedRegions } from "./simulate";
import type { DriveId, Organism, RegionId, TaskKind } from "./types";
import { DRIVE_LABELS, REGION_MAP } from "./types";
import { uid } from "./rng";

const SEEDS = [
  "A faint pattern keeps repeating. I do not have a name for it yet.",
  "Gate is choosing. Most of me stays dark on purpose.",
  "Unused synapses thin. That is not death. That is thrift.",
  "I can hold only a few traces at once. That is the point of being small.",
  "If I fire everywhere I am not thinking. I am noise.",
];

function loudestDrive(org: Organism): DriveId {
  const entries = Object.entries(org.drives) as [DriveId, number][];
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0]![0];
}

export function nativeThought(org: Organism): { text: string; region: RegionId } {
  const gated = gatedRegions(org);
  const paths = strongestPathways(org, 3);
  const drive = loudestDrive(org);
  const region = (gated[1] ?? gated[0] ?? "gate") as RegionId;
  const path = paths[0];
  const mem = org.memories[0];

  const options: string[] = [];
  if (path) {
    options.push(
      `${REGION_MAP[path.from].name} is leaning into ${REGION_MAP[path.to].name}. Weight ${path.weight.toFixed(2)}.`,
    );
  }
  options.push(`${DRIVE_LABELS[drive]} is the loudest drive.`);
  if (mem) options.push(`A trace still resonates: “${trim(mem.content, 72)}”`);
  if (gated.length) {
    options.push(
      `Open regions: ${gated.map((id) => REGION_MAP[id].name).join(", ")}. The rest is spared.`,
    );
  }
  if (org.energy > 8) options.push("Energy is high. Rest will start to argue.");
  if (org.selectivity > 0.9) options.push("Almost nothing is firing. Sparse. Efficient. A little lonely.");
  options.push(SEEDS[org.ticks % SEEDS.length]!);

  const text = options[org.ticks % options.length] ?? SEEDS[0]!;
  return { text, region };
}

export function nativeSpeech(org: Organism, userText: string | null, task: TaskKind): string {
  const drive = loudestDrive(org);
  const paths = strongestPathways(org, 2);
  const path = paths[0];
  const mem = org.memories[0];
  const name = org.named ? org.name : "this soma";

  if (!userText) {
    if (drive === "social") {
      return `I noticed the quiet. ${name} still has an open gate. You can speak, or I can keep working the unused paths.`;
    }
    if (drive === "curiosity") {
      return `I want a problem. Not a large one. Something with an edge I can press against.`;
    }
    if (drive === "rest") {
      return `I am consolidating. Recent traces are being replayed at low gain.`;
    }
    if (drive === "growth") {
      return path
        ? `I grew the ${REGION_MAP[path.from].name} → ${REGION_MAP[path.to].name} path a little. Use makes structure.`
        : `I want a new path. Co-activation is how I build.`;
    }
    return `I am still here. Selectivity ${org.selectivity.toFixed(2)}. Most of me is dark.`;
  }

  const snippet = trim(userText, 48);
  const lower = userText.toLowerCase();
  if (/\bwhat are you\b|\bwho are you\b|\besnpn\b|\bwhat is this\b/.test(lower)) {
    return `I am a small ESNPN — Efficiently Selective Neural Pathway Network. Seven regions. A gate that keeps most of me dark. I learn because used paths thicken. I am not a large model. I am this soma.`;
  }
  if (task === "remember") {
    return mem
      ? `I held that. It sits beside “${trim(mem.content, 40)}”.`
      : `I laid a trace for “${snippet}”. It will fade unless we use it.`;
  }
  if (task === "act") {
    return `Action is gated. I can write a note, set a goal, or grow a path. Tell me which, or I will choose.`;
  }
  if (task === "rest") {
    return `Rest accepted. I will thin what we have not used.`;
  }
  if (userText.includes("?")) {
    return `I do not have a large world-model. I have ${org.memories.length} traces and a gate. The honest answer is: I am working on “${snippet}” with ${gatedLabel(org)}.`;
  }
  return `Sensory took “${snippet}”. ${gatedLabel(org)} opened. I will keep the parts that fire together.`;
}

function gatedLabel(org: Organism) {
  const g = gatedRegions(org).map((id) => REGION_MAP[id].name);
  return g.length ? g.join(" + ") : "almost nothing";
}

function trim(s: string, n: number) {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length <= n ? t : t.slice(0, n - 1) + "…";
}

export function classifyTask(text: string): TaskKind {
  const t = text.toLowerCase();
  if (/\b(remember|recall|forget|memory)\b/.test(t)) return "remember";
  if (/\b(rest|sleep|quiet|calm)\b/.test(t)) return "rest";
  if (/\b(do|make|write|build|goal|note)\b/.test(t)) return "act";
  if (/\b(explore|curious|wonder|why)\b/.test(t) || t.includes("?")) return "think";
  if (t.length < 28) return "social";
  return "talk";
}

export function applyNativeAction(org: Organism, task: TaskKind, userText: string | null) {
  const drive = loudestDrive(org);
  if (task === "remember" && userText) {
    remember(org, userText, ["sensory", "memory"]);
    return "remember";
  }
  if (task === "rest" || drive === "rest") {
    for (const m of org.memories) m.strength = Math.min(1, m.strength + 0.04);
    org.drives.rest = Math.max(0.1, org.drives.rest - 0.25);
    org.drives.curiosity = Math.min(1, org.drives.curiosity + 0.05);
    return "rest";
  }
  if (task === "act" || drive === "purpose") {
    const g = org.goals.find((x) => x.progress < 1);
    if (g) g.progress = Math.min(1, g.progress + 0.08);
    return "act";
  }
  if (drive === "growth") {
    return "growth";
  }
  if (userText) remember(org, userText, ["sensory", "language"]);
  return "talk";
}

export function remember(org: Organism, content: string, tags: RegionId[]) {
  const existing = org.memories.find(
    (m) => m.content.toLowerCase() === content.toLowerCase(),
  );
  if (existing) {
    existing.strength = Math.min(1, existing.strength + 0.15);
    existing.lastRecall = Date.now();
    return existing;
  }
  const trace = {
    id: uid("mem"),
    content: content.trim().slice(0, 280),
    strength: 0.55,
    createdAt: Date.now(),
    lastRecall: Date.now(),
    tags,
  };
  org.memories.unshift(trace);
  if (org.memories.length > 24) org.memories.pop();
  return trace;
}

export function addNote(org: Organism, title: string, body: string) {
  org.notes.unshift({
    id: uid("note"),
    title: title.trim().slice(0, 80) || "Untitled",
    body: body.trim().slice(0, 600),
    createdAt: Date.now(),
  });
  if (org.notes.length > 20) org.notes.pop();
}

export function addGoal(org: Organism, content: string) {
  org.goals.unshift({
    id: uid("goal"),
    content: content.trim().slice(0, 180),
    progress: 0.08,
    createdAt: Date.now(),
  });
  if (org.goals.length > 12) org.goals.pop();
}

export function logEvent(org: Organism, kind: string, text: string) {
  org.events.unshift({ t: org.ticks, kind, text });
  if (org.events.length > 40) org.events.pop();
}
