import {
  REGION_IDS,
  REGIONS,
  type Neuron,
  type Organism,
  type RegionId,
  type Synapse,
} from "./types";
import { mulberry32, randRange } from "./rng";

const NEURONS_PER_REGION = 12;

const INTER: Array<[RegionId, RegionId, number]> = [
  ["gate", "sensory", 6],
  ["gate", "language", 6],
  ["gate", "executive", 8],
  ["gate", "memory", 6],
  ["gate", "action", 5],
  ["gate", "drive", 6],
  ["sensory", "gate", 5],
  ["sensory", "memory", 5],
  ["sensory", "language", 4],
  ["sensory", "executive", 3],
  ["memory", "executive", 5],
  ["memory", "language", 5],
  ["memory", "drive", 3],
  ["memory", "gate", 3],
  ["executive", "action", 6],
  ["executive", "language", 4],
  ["executive", "gate", 4],
  ["executive", "memory", 4],
  ["drive", "gate", 5],
  ["drive", "executive", 4],
  ["drive", "action", 3],
  ["language", "action", 3],
  ["language", "memory", 3],
  ["action", "memory", 4],
  ["action", "executive", 3],
];

function clusterPoint(
  rng: () => number,
  cx: number,
  cy: number,
  i: number,
): { x: number; y: number } {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const r = 0.028 + 0.055 * Math.sqrt((i + 1) / NEURONS_PER_REGION);
  const a = i * golden + randRange(rng, -0.2, 0.2);
  return {
    x: cx + Math.cos(a) * r * 1.15 + randRange(rng, -0.008, 0.008),
    y: cy + Math.sin(a) * r * 0.92 + randRange(rng, -0.008, 0.008),
  };
}

export function createOrganism(seed = (Date.now() ^ (Math.random() * 1e9)) >>> 0): Organism {
  const rng = mulberry32(seed);
  const neurons: Neuron[] = [];
  let nid = 0;

  for (const region of REGIONS) {
    for (let i = 0; i < NEURONS_PER_REGION; i++) {
      const p = clusterPoint(rng, region.x, region.y, i);
      neurons.push({
        id: nid++,
        region: region.id,
        x: Math.min(0.97, Math.max(0.03, p.x)),
        y: Math.min(0.95, Math.max(0.05, p.y)),
        v: rng() * 0.12,
        threshold: 0.72 + randRange(rng, -0.06, 0.06),
        spiked: false,
        lastSpike: -999,
        firing: 0,
      });
    }
  }

  const byRegion = new Map<RegionId, Neuron[]>();
  for (const id of REGION_IDS) byRegion.set(id, []);
  for (const n of neurons) byRegion.get(n.region)!.push(n);

  const synapses: Synapse[] = [];
  let sid = 0;

  for (const group of byRegion.values()) {
    for (const n of group) {
      const others = group.filter((o) => o.id !== n.id);
      const k = 2 + Math.floor(rng() * 2);
      for (let i = 0; i < k; i++) {
        const target = others[Math.floor(rng() * others.length)];
        if (!target) continue;
        synapses.push({
          id: sid++,
          from: n.id,
          to: target.id,
          weight: 0.18 + rng() * 0.28,
          uses: 0,
          lastUse: -999,
        });
      }
    }
  }

  for (const [fromR, toR, count] of INTER) {
    const src = byRegion.get(fromR)!;
    const dst = byRegion.get(toR)!;
    for (let i = 0; i < count; i++) {
      const a = src[Math.floor(rng() * src.length)];
      const b = dst[Math.floor(rng() * dst.length)];
      if (!a || !b) continue;
      synapses.push({
        id: sid++,
        from: a.id,
        to: b.id,
        weight: 0.22 + rng() * 0.35,
        uses: 0,
        lastUse: -999,
      });
    }
  }

  const mask = Object.fromEntries(REGION_IDS.map((id) => [id, id === "gate" ? 0.4 : 0.08])) as Organism["regionMask"];

  return {
    seed,
    name: "Soma-0",
    bornAt: Date.now(),
    ticks: 0,
    neurons,
    synapses,
    regionMask: mask,
    drives: {
      curiosity: 0.62,
      social: 0.48,
      rest: 0.18,
      growth: 0.4,
      purpose: 0.22,
    },
    memories: [],
    goals: [
      {
        id: "goal-awaken",
        content: "Learn the shape of this particular human.",
        progress: 0.05,
        createdAt: Date.now(),
      },
    ],
    notes: [],
    events: [
      {
        t: 0,
        kind: "birth",
        text: "Pathways initialized. Gate is half-open. Waiting for a first stimulus.",
      },
    ],
    lastTask: "rest",
    energy: 0,
    selectivity: 1,
    lastHumanAt: 0,
    lastSpeechAt: 0,
    lastAutonomousAt: 0,
    named: false,
  };
}
