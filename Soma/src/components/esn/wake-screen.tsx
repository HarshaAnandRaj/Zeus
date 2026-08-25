import { Button } from "@/components/ui/button";

export function WakeScreen({ onWake, ready }: { onWake: () => void; ready: boolean }) {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center bg-bg px-5 py-8">
      <div className="w-full max-w-lg text-center">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Efficiently Selective Neural Pathway Network
        </p>
        <h1 className="font-display mt-3 text-4xl leading-tight tracking-tight text-fg sm:mt-4 sm:text-6xl">
          A small mind
        </h1>
        <p className="mt-4 text-sm leading-relaxed text-muted-foreground sm:mt-5 sm:text-base">
          Seven regions. Sparse pathways. Only what a task needs is allowed to fire.
          Used connections thicken. Unused ones thin. Drives create intent without a prompt.
        </p>
        <div className="mt-6 sm:mt-8">
          <Button size="lg" onClick={onWake} className="min-w-40" disabled={!ready}>
            Wake the soma
          </Button>
        </div>
        <ul className="mx-auto mt-6 grid w-full gap-2 text-left text-sm text-muted-foreground sm:mt-8">
          <li className="rounded-lg bg-surface px-4 py-2.5 shadow-[var(--shadow-border)]">
            <span className="font-medium text-fg">Selective.</span> The gate opens a few regions and leaves the rest dark.
          </li>
          <li className="rounded-lg bg-surface px-4 py-2.5 shadow-[var(--shadow-border)]">
            <span className="font-medium text-fg">Plastic.</span> Fire together, wire together. Structure is use.
          </li>
          <li className="rounded-lg bg-surface px-4 py-2.5 shadow-[var(--shadow-border)]">
            <span className="font-medium text-fg">Autonomous.</span> Curiosity, company, rest, growth, purpose — it acts for itself.
          </li>
        </ul>
      </div>
    </main>
  );
}
