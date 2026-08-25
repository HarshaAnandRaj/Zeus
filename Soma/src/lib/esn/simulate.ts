import { REGION_IDS, TASK_ROUTES, type Organism, type Pulse, type RegionId, type TaskKind } from "./types";

const DECAY = 0.86;
const RESET = 0.08;
const K_WTA = 4;
const PULSE_LIFE = 18;

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

export function setMaskForTask(org: Organism, task: TaskKind, extra: RegionId[] = []) {
  const open = new Set<RegionId>([...TASK_ROUTES[task], ...extra]);
  if (org.drives.curiosity > 0.7) open.add("drive");
  if (org.drives.purpose > 0.65) open.add("executive");
  for (const id of REGION_IDS) {
    org.regionMask[id] = open.has(id) ? (id === "gate" ? 1 : 0.92) : 0.06;
  }
  org.lastTask = task;
}

export function inject(org: Organism, regions: RegionId[], amount: number) {
  const per = amount;
  for (const n of org.neurons) {
    if (regions.includes(n.region)) {
      n.v += per * (0.7 + Math.random() * 0.6);
    }
  }
}

export function tickNetwork(org: Organism, pulses: Pulse[]): number {
  org.ticks += 1;
  const t = org.ticks;
  let spikes = 0;

  for (const n of org.neurons) {
    n.spiked = false;
    n.v *= DECAY;
    n.firing *= 0.88;
    n.v += org.ticks % 4 === 0 ? 0.028 * org.regionMask[n.region] : 0;
  }


  const incoming = new Float32Array(org.neurons.length);
  for (const s of org.synapses) {
    const pre = org.neurons[s.from];
    if (!pre || pre.lastSpike !== t - 1) continue;
    incoming[s.to] += s.weight * (0.55 + pre.firing * 0.5);
    s.lastUse = t;
    s.uses += 1;
    pulses.push({
      from: s.from,
      to: s.to,
      born: t,
      life: PULSE_LIFE,
      strength: s.weight,
    });
  }

  const ranked: Record<RegionId, { id: number; v: number }[]> = {
    gate: [],
    sensory: [],
    language: [],
    executive: [],
    memory: [],
    action: [],
    drive: [],
  };

  for (const n of org.neurons) {
    const gated = org.regionMask[n.region];
    n.v += incoming[n.id] * (0.25 + gated);
    ranked[n.region].push({ id: n.id, v: n.v });
  }

  const allowed = new Set<number>();
  for (const id of REGION_IDS) {
    const list = ranked[id].sort((a, b) => b.v - a.v);
    const k = org.regionMask[id] > 0.4 ? K_WTA : 1;
    for (let i = 0; i < k; i++) {
      const item = list[i];
      if (item) allowed.add(item.id);
    }
  }

  for (const n of org.neurons) {
    if (!allowed.has(n.id)) {
      if (n.v > n.threshold * 0.9) n.v *= 0.5;
      continue;
    }
    if (n.v > n.threshold && org.regionMask[n.region] > 0.2) {
      n.spiked = true;
      n.lastSpike = t;
      n.firing = Math.min(1, n.firing + 0.85);
      n.v = RESET;
      spikes += 1;
    }
  }

  // Hebbian: pre and post recently co-active → thicken
  for (const s of org.synapses) {
    const pre = org.neurons[s.from];
    const post = org.neurons[s.to];
    if (!pre || !post) continue;
    const together =
      t - pre.lastSpike < 8 && t - post.lastSpike < 8 && pre.lastSpike >= 0 && post.lastSpike >= 0;
    if (together) {
      s.weight = clamp01(s.weight + 0.012 * (1 - s.weight));
    } else if (t - s.lastUse > 80) {
      s.weight = Math.max(0.04, s.weight * 0.9994);
    }
  }

  const active = org.neurons.filter((n) => n.firing > 0.25).length;
  org.selectivity = 1 - active / org.neurons.length;
  org.energy = org.energy * 0.92 + spikes;

  if (pulses.length > 220) pulses.splice(0, pulses.length - 180);

  return spikes;
}

export function decayMasks(org: Organism) {
  for (const id of REGION_IDS) {
    const floor = id === "gate" ? 0.38 : 0.14;
    org.regionMask[id] = org.regionMask[id] * 0.975 + floor * 0.025;
  }
}


export function tickDrives(org: Organism, autonomy: boolean) {
  const d = org.drives;
  const idle = Date.now() - (org.lastHumanAt || org.bornAt);
  d.curiosity = clamp01(d.curiosity + (autonomy ? 0.004 : 0.001) - d.rest * 0.002);
  d.social = clamp01(d.social + (idle > 20000 ? 0.006 : -0.002));
  d.rest = clamp01(d.rest + Math.min(org.energy, 5) * 0.0005 - 0.003);
  d.growth = clamp01(d.growth + 0.0015 - d.rest * 0.001);
  const openGoals = org.goals.filter((g) => g.progress < 1).length;
  d.purpose = clamp01(d.purpose * 0.995 + openGoals * 0.008);
}

export function growPathway(org: Organism, from: RegionId, to: RegionId) {
  const src = org.neurons.filter((n) => n.region === from);
  const dst = org.neurons.filter((n) => n.region === to);
  if (!src.length || !dst.length) return false;
  const a = src[Math.floor(Math.random() * src.length)];
  const b = dst[Math.floor(Math.random() * dst.length)];
  org.synapses.push({
    id: org.synapses.length + 1000 + org.ticks,
    from: a.id,
    to: b.id,
    weight: 0.32,
    uses: 1,
    lastUse: org.ticks,
  });
  return true;
}

export function strongestPathways(org: Organism, n = 6) {
  const scored = org.synapses
    .map((s) => {
      const a = org.neurons[s.from];
      const b = org.neurons[s.to];
      return {
        from: a.region,
        to: b.region,
        weight: s.weight,
        inter: a.region !== b.region,
      };
    })
    .filter((s) => s.inter)
    .sort((a, b) => b.weight - a.weight);
  const uniq: typeof scored = [];
  const seen = new Set<string>();
  for (const s of scored) {
    const k = `${s.from}-${s.to}`;
    if (seen.has(k)) continue;
    seen.add(k);
    uniq.push(s);
    if (uniq.length >= n) break;
  }
  return uniq;
}

export function gatedRegions(org: Organism): RegionId[] {
  return REGION_IDS.filter((id) => org.regionMask[id] > 0.35);
}
