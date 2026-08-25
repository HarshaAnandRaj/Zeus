import { Activity, Brain, Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getRuntime, useMind } from "@/lib/esn/store";

export function HeaderBar() {
  const thinking = useMind((s) => s.thinking);
  const autonomy = useMind((s) => s.autonomy);
  const setAutonomy = useMind((s) => s.setAutonomy);
  const think = useMind((s) => s.think);
  const setPanel = useMind((s) => s.setPanel);
  const panel = useMind((s) => s.panel);
  const version = useMind((s) => s.version);
  const aiAvailable = useMind((s) => s.aiAvailable);
  void version;

  const org = getRuntime().organism;
  const ageMin = Math.max(0, Math.floor(org.ticks / 12 / 60));
  const ageSec = Math.max(0, Math.floor(org.ticks / 12) % 60);

  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3 sm:px-6">
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <div className="flex size-9 items-center justify-center rounded-md bg-raised shadow-[var(--shadow-border)]">
          <Brain className="size-4 text-accent" strokeWidth={1.75} />
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-display text-lg leading-none tracking-tight text-fg">ESNPN</p>
            {autonomy ? <Badge variant="live">Live</Badge> : <Badge>Held</Badge>}
          </div>
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {org.named ? org.name : "Unnamed soma"}
            <span className="text-faint"> · </span>
            <span className="tabular-nums">
              {ageMin}:{String(ageSec).padStart(2, "0")}
            </span>
            <span className="text-faint"> · </span>
            <span className="tabular-nums">{Math.round(org.selectivity * 100)}% sparse</span>
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant={panel === "arch" ? "secondary" : "ghost"}
          size="sm"
          onClick={() => setPanel(panel === "arch" ? "talk" : "arch")}
          className="hidden sm:inline-flex"
        >
          Architecture
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAutonomy(!autonomy)}
          aria-pressed={autonomy}
        >
          {autonomy ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          {autonomy ? "Hold" : "Live"}
        </Button>
        <Button size="sm" onClick={() => void think()} disabled={thinking}>
          <Activity className="size-3.5" />
          {thinking ? "Firing" : "Think"}
        </Button>
      </div>
      {aiAvailable === false ? (
        <p className="w-full text-xs text-muted-foreground">
          Language cortex unavailable — this soma is speaking with its native voice.
        </p>
      ) : null}
    </header>
  );
}
