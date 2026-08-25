import { getRuntime, useMind } from "@/lib/esn/store";
import { REGION_MAP } from "@/lib/esn/types";

export function InnerPanel() {
  const version = useMind((s) => s.version);
  void version;
  const thoughts = getRuntime().thoughts;
  const events = getRuntime().organism.events;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto">
      <section>
        <h2 className="text-2xs font-medium tracking-widest text-muted-foreground uppercase">
          Inner voice
        </h2>
        <ul className="mt-3 space-y-3">
          {thoughts.length === 0 ? (
            <li className="text-sm text-muted-foreground">Quiet. Touch a region or ask it to think.</li>
          ) : (
            thoughts.slice(0, 16).map((t) => (
              <li key={t.id} className="text-sm leading-relaxed text-fg">
                <span className="mr-2 text-2xs tracking-wider text-faint uppercase">
                  {REGION_MAP[t.region].short}
                </span>
                {t.text}
              </li>
            ))
          )}
        </ul>
      </section>
      <section>
        <h2 className="text-2xs font-medium tracking-widest text-muted-foreground uppercase">
          Pathway log
        </h2>
        <ul className="mt-3 space-y-2">
          {events.slice(0, 12).map((e, i) => (
            <li key={`${e.t}-${i}`} className="text-xs leading-relaxed text-muted-foreground">
              <span className="tabular-nums text-faint">t{e.t}</span>
              <span className="mx-2 text-faint">·</span>
              {e.text}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
