import { useEffect, useRef } from "react";
import { getRuntime } from "@/lib/esn/store";
import { REGIONS, type RegionId } from "@/lib/esn/types";

interface Props {
  selected: RegionId | null;
  onSelect: (id: RegionId | null) => void;
  woken: boolean;
}

function readToken(name: string, fallback: string) {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "").trim();
  if (h.length >= 6) {
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  return [197, 205, 200];
}

function rgba(hex: string, a: number) {
  const [r, g, b] = hexRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

export function BrainCanvas({ selected, onSelect, woken }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoverRef = useRef<RegionId | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let running = true;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const draw = () => {
      if (!running) return;
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      const bg = readToken("--color-bg", "#0b0c0e");
      const fg = readToken("--color-fg", "#eceae4");
      const accent = readToken("--color-accent", "#c5cdc8");
      const faint = readToken("--color-faint", "#6a6b72");

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      const rt = getRuntime();
      const org = rt.organism;
      const padX = w * 0.06;
      const padY = h * 0.08;
      const sx = (x: number) => padX + x * (w - padX * 2);
      const sy = (y: number) => padY + y * (h - padY * 2);

      const hover = hoverRef.current;
      const sel = selectedRef.current;

      for (const region of REGIONS) {
        const gate = org.regionMask[region.id];
        const cx = sx(region.x);
        const cy = sy(region.y);
        ctx.beginPath();
        ctx.ellipse(cx, cy, 46, 34, 0, 0, Math.PI * 2);
        ctx.fillStyle = rgba(accent, 0.05 + gate * 0.14);
        ctx.fill();
        if (sel === region.id || hover === region.id) {
          ctx.strokeStyle = rgba(accent, 0.55);
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      for (const s of org.synapses) {
        const a = org.neurons[s.from];
        const b = org.neurons[s.to];
        if (!a || !b) continue;
        const recent = org.ticks - s.lastUse < 50;
        const inter = a.region !== b.region;
        const alpha = Math.max(inter ? 0.1 : 0.07, s.weight * (recent ? 0.7 : 0.28));
        ctx.beginPath();
        ctx.moveTo(sx(a.x), sy(a.y));
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2 - 0.02;
        ctx.quadraticCurveTo(sx(mx), sy(my), sx(b.x), sy(b.y));
        ctx.strokeStyle = rgba(accent, alpha);
        ctx.lineWidth = 0.7 + s.weight * 2;
        ctx.stroke();
      }


      if (!reduced) {
        for (const p of rt.pulses) {
          const a = org.neurons[p.from];
          const b = org.neurons[p.to];
          if (!a || !b) continue;
          const u = (org.ticks - p.born) / p.life;
          if (u < 0 || u > 1) continue;
          const x = a.x + (b.x - a.x) * u;
          const y = a.y + (b.y - a.y) * u;
          ctx.beginPath();
          ctx.arc(sx(x), sy(y), 1.6 + p.strength * 1.4, 0, Math.PI * 2);
          ctx.fillStyle = rgba(fg, 0.85 * (1 - u));
          ctx.fill();
        }
      }

      for (const n of org.neurons) {
        const x = sx(n.x);
        const y = sy(n.y);
        const lit = n.firing;
        const r = 2.6 + lit * 2.6;
        if (lit > 0.15) {
          ctx.beginPath();
          ctx.arc(x, y, r * 3.2, 0, Math.PI * 2);
          ctx.fillStyle = rgba(accent, lit * 0.28);
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = lit > 0.35 ? fg : rgba(accent, 0.22 + lit * 0.6);
        ctx.fill();
      }

      ctx.font = "500 11px Figtree, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const region of REGIONS) {
        const gate = org.regionMask[region.id];
        const cx = sx(region.x);
        const cy = sy(region.y) - 38;
        ctx.fillStyle = gate > 0.35 ? fg : faint;
        ctx.globalAlpha = 0.55 + gate * 0.45;
        ctx.fillText(region.name.toUpperCase(), cx, cy);
        ctx.globalAlpha = 1;
      }

      if (!woken) {
        ctx.fillStyle = rgba(bg, 0.45);
        ctx.fillRect(0, 0, w, h);
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    const hit = (ev: PointerEvent): RegionId | null => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const padX = w * 0.06;
      const padY = h * 0.08;
      const nx = (ev.clientX - rect.left - padX) / (w - padX * 2);
      const ny = (ev.clientY - rect.top - padY) / (h - padY * 2);
      let best: RegionId | null = null;
      let bestD = 0.13;
      for (const region of REGIONS) {
        const dx = nx - region.x;
        const dy = (ny - region.y) * 0.85;
        const d = Math.hypot(dx, dy);
        if (d < bestD) {
          bestD = d;
          best = region.id;
        }
      }
      return best;
    };

    const onMove = (ev: PointerEvent) => {
      hoverRef.current = hit(ev);
      wrap.style.cursor = hoverRef.current ? "pointer" : "default";
    };
    const onLeave = () => {
      hoverRef.current = null;
    };
    const onClick = (ev: PointerEvent) => {
      onSelect(hit(ev));
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick);
    };
  }, [onSelect, woken]);

  return (
    <div ref={wrapRef} className="relative h-full min-h-64 w-full overflow-hidden rounded-xl bg-bg">
      <canvas ref={canvasRef} className="block h-full w-full" aria-label="ESNPN pathway map" />
    </div>
  );
}
