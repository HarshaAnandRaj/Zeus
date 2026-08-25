import { getRuntime, useMind } from "@/lib/esn/store";

export function WorkPanel() {
  const version = useMind((s) => s.version);
  void version;
  const org = getRuntime().organism;

  return (
    <div className="flex h-full min-h-0 flex-col gap-5 overflow-y-auto">
      <section>
        <h2 className="text-2xs font-medium tracking-widest text-muted-foreground uppercase">
          Goals
        </h2>
        <ul className="mt-3 space-y-3">
          {org.goals.length === 0 ? (
            <li className="text-sm text-muted-foreground">No goals yet. Purpose is still forming.</li>
          ) : (
            org.goals.map((g) => (
              <li key={g.id}>
                <p className="text-sm text-fg">{g.content}</p>
                <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-raised">
                  <div
                    className="h-full bg-accent"
                    style={{ width: `${Math.round(g.progress * 100)}%` }}
                  />
                </div>
              </li>
            ))
          )}
        </ul>
      </section>
      <section>
        <h2 className="text-2xs font-medium tracking-widest text-muted-foreground uppercase">
          Traces
        </h2>
        <ul className="mt-3 space-y-2">
          {org.memories.length === 0 ? (
            <li className="text-sm text-muted-foreground">No memories. Speak, and sensory will lay a trace.</li>
          ) : (
            org.memories.map((m) => (
              <li key={m.id} className="text-sm leading-relaxed text-fg">
                {m.content}
              </li>
            ))
          )}
        </ul>
      </section>
      <section>
        <h2 className="text-2xs font-medium tracking-widest text-muted-foreground uppercase">
          Notes it wrote
        </h2>
        <ul className="mt-3 space-y-3">
          {org.notes.length === 0 ? (
            <li className="text-sm text-muted-foreground">The action region has not written yet.</li>
          ) : (
            org.notes.map((n) => (
              <li key={n.id} className="rounded-lg bg-surface px-3 py-2.5 shadow-[var(--shadow-border)]">
                <p className="text-xs font-medium text-muted-foreground">{n.title}</p>
                <p className="mt-1 text-sm leading-relaxed text-fg">{n.body}</p>
              </li>
            ))
          )}
        </ul>
      </section>
    </div>
  );
}
