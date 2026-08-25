import { gatedRegions, strongestPathways } from "./simulate";
import type { MindSnapshot, Organism, Thought } from "./types";

export function toSnapshot(
  org: Organism,
  thoughts: Thought[],
  lastUser: string,
  mode: "reply" | "think",
): MindSnapshot {
  return {
    name: org.name,
    ticks: org.ticks,
    drives: { ...org.drives },
    gatedRegions: gatedRegions(org),
    topPathways: strongestPathways(org, 5).map((p) => ({
      from: p.from,
      to: p.to,
      weight: Number(p.weight.toFixed(3)),
    })),
    memories: org.memories.slice(0, 6).map((m) => m.content),
    goals: org.goals.filter((g) => g.progress < 1).slice(0, 4).map((g) => g.content),
    notes: org.notes.slice(0, 3).map((n) => ({ title: n.title, body: n.body.slice(0, 180) })),
    lastThoughts: thoughts.slice(0, 4).map((t) => t.text),
    lastUser,
    mode,
    named: org.named,
  };
}
