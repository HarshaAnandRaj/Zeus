import { getRuntime, useMind } from "@/lib/esn/store";
import { DRIVE_LABELS, type DriveId } from "@/lib/esn/types";

const ORDER: DriveId[] = ["curiosity", "social", "rest", "growth", "purpose"];

export function DriveMeters() {
  const version = useMind((s) => s.version);
  void version;
  const drives = getRuntime().organism.drives;

  return (
    <div className="grid grid-cols-5 gap-2">
      {ORDER.map((id) => {
        const v = drives[id];
        return (
          <div key={id} className="min-w-0">
            <div className="flex items-baseline justify-between gap-1">
              <span className="truncate text-2xs font-medium tracking-wide text-muted-foreground uppercase">
                {DRIVE_LABELS[id]}
              </span>
              <span className="tabular-nums text-2xs text-faint">{Math.round(v * 100)}</span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded-full bg-raised">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
                style={{ width: `${Math.round(v * 100)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
