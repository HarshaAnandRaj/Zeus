import { getRuntime, useMind } from "@/lib/esn/store";
import { REGION_MAP, type RegionId } from "@/lib/esn/types";
import { Button } from "@/components/ui/button";

export function RegionCard({ id }: { id: RegionId }) {
  const version = useMind((s) => s.version);
  const poke = useMind((s) => s.poke);
  const selectRegion = useMind((s) => s.selectRegion);
  void version;
  const org = getRuntime().organism;
  const def = REGION_MAP[id];
  const mask = org.regionMask[id];
  const cells = org.neurons.filter((n) => n.region === id);
  const firing = cells.filter((n) => n.firing > 0.25).length;
  const paths = org.synapses.filter((s) => {
    const a = org.neurons[s.from];
    const b = org.neurons[s.to];
    return a.region === id || b.region === id;
  }).length;

  return (
    <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
            Region
          </p>
          <h3 className="font-display mt-1 text-xl text-fg">{def.name}</h3>
        </div>
        <button
          type="button"
          className="text-xs text-faint hover:text-fg"
          onClick={() => selectRegion(null)}
        >
          Close
        </button>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{def.role}</p>
      <dl className="mt-4 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-md bg-raised px-2 py-2">
          <dt className="text-xs text-faint">Gate</dt>
          <dd className="tabular-nums text-sm text-fg">{Math.round(mask * 100)}%</dd>
        </div>
        <div className="rounded-md bg-raised px-2 py-2">
          <dt className="text-xs text-faint">Firing</dt>
          <dd className="tabular-nums text-sm text-fg">
            {firing}/{cells.length}
          </dd>
        </div>
        <div className="rounded-md bg-raised px-2 py-2">
          <dt className="text-xs text-faint">Paths</dt>
          <dd className="tabular-nums text-sm text-fg">{paths}</dd>
        </div>
      </dl>
      <Button className="mt-4 w-full" variant="secondary" onClick={() => poke(id)}>
        Stimulate {def.name}
      </Button>
    </div>
  );
}
