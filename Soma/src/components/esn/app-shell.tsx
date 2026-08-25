import { useEffect } from "react";
import { AudioLines, MessageSquare, NotebookPen } from "lucide-react";
import { HeaderBar } from "./header-bar";
import { BrainCanvas } from "./brain-canvas";
import { ChatPanel } from "./chat-panel";
import { InnerPanel } from "./inner-panel";
import { WorkPanel } from "./work-panel";
import { ArchitecturePanel } from "./architecture-panel";
import { DriveMeters } from "./drive-meters";
import { RegionCard } from "./region-card";
import { WakeScreen } from "./wake-screen";
import { useMind } from "@/lib/esn/store";
import { cn } from "@/lib/utils";

const TABS = [
  { id: "talk" as const, label: "Talk", icon: MessageSquare },
  { id: "inner" as const, label: "Inner", icon: AudioLines },
  { id: "work" as const, label: "Work", icon: NotebookPen },
];

export function AppShell() {
  const hydrate = useMind((s) => s.hydrate);
  const ready = useMind((s) => s.ready);
  const woken = useMind((s) => s.woken);
  const wake = useMind((s) => s.wake);
  const panel = useMind((s) => s.panel);
  const setPanel = useMind((s) => s.setPanel);
  const selected = useMind((s) => s.selectedRegion);
  const selectRegion = useMind((s) => s.selectRegion);
  const poke = useMind((s) => s.poke);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  if (!woken) {
    return <WakeScreen onWake={wake} ready={ready} />;
  }

  return (
    <div className="flex min-h-dvh flex-col bg-bg text-fg">
      <HeaderBar />
      <div className="relative mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col gap-4 p-3 sm:p-5 lg:grid lg:grid-cols-12 lg:gap-5">
        <section className="relative flex min-h-0 flex-col gap-3 lg:col-span-7">
          <div className="relative h-80 overflow-hidden rounded-xl bg-surface p-2 shadow-[var(--shadow-border)] sm:h-96 lg:h-auto lg:min-h-96 lg:flex-1">
            <BrainCanvas
              selected={selected}
              onSelect={(id) => {
                if (id) poke(id);
                else selectRegion(null);
              }}
              woken={woken}
            />
          </div>
          <div className="rounded-xl bg-surface px-4 py-3 shadow-[var(--shadow-border)]">
            <DriveMeters />
          </div>
          {selected ? (
            <div className="lg:hidden">
              <RegionCard id={selected} />
            </div>
          ) : null}
        </section>

        <aside className="flex min-h-0 flex-col rounded-xl bg-surface p-4 shadow-[var(--shadow-border)] lg:col-span-5 lg:min-h-96">
          <div className="mb-3 flex gap-1 rounded-lg bg-raised p-1">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setPanel(tab.id)}
                  className={cn(
                    "flex h-11 flex-1 items-center justify-center gap-1.5 rounded-md text-xs font-medium transition-colors duration-150",
                    panel === tab.id
                      ? "bg-surface text-fg shadow-[var(--shadow-border)]"
                      : "text-muted-foreground hover:text-fg",
                  )}
                >
                  <Icon className="size-3.5" />
                  {tab.label}
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => setPanel("arch")}
              className={cn(
                "flex h-11 items-center justify-center rounded-md px-3 text-xs font-medium sm:hidden",
                panel === "arch" ? "bg-surface text-fg" : "text-muted-foreground",
              )}
            >
              Arch
            </button>
          </div>
          <div className="min-h-72 flex-1 lg:min-h-0">
            {panel === "talk" ? <ChatPanel /> : null}
            {panel === "inner" ? <InnerPanel /> : null}
            {panel === "work" ? <WorkPanel /> : null}
            {panel === "arch" ? <ArchitecturePanel /> : null}
          </div>
          {selected && panel !== "arch" ? (
            <div className="mt-4 hidden lg:block">
              <RegionCard id={selected} />
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
