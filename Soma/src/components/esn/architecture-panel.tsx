import { Button } from "@/components/ui/button";
import { useMind } from "@/lib/esn/store";

export function ArchitecturePanel() {
  const reset = useMind((s) => s.reset);
  const setPanel = useMind((s) => s.setPanel);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto">
      <h2 className="font-display text-2xl tracking-tight text-fg">ESNPN</h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Efficiently Selective Neural Pathway Networks
      </p>
      <div className="mt-5 space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          A small mind should not fire like a dense transformer. Biology does not. Cortex is
          specialized, gated, and plastic. ESNPN copies three facts:
        </p>
        <p>
          <span className="font-medium text-fg">Selective routing.</span> A gate scores the
          current task against seven regions and opens only the few that are needed. Winner-take-all
          inside a region keeps spikes sparse. Efficiency is the fraction of the network that stays
          dark.
        </p>
        <p>
          <span className="font-medium text-fg">Neuron model.</span> Each unit is a leaky
          integrator. Potential decays, synapses add current, a threshold emits a spike, then reset.
          Pulses you see on the map are those spikes travelling a pathway.
        </p>
        <p>
          <span className="font-medium text-fg">Neuroplasticity.</span> If two neurons fire in a
          short window, their synapse thickens (Hebb). Unused weights thin. Co-active regions can
          grow a new path. Structure is a record of use.
        </p>
        <p>
          <span className="font-medium text-fg">Autonomy.</span> Drives — curiosity, company, rest,
          growth, purpose — rise and fall on their own. When one crosses a threshold the soma acts:
          it writes a note, replays a memory, grows a path, or asks the language cortex to speak.
        </p>
        <p>
          The map is the model. Language, when available, is one specialized region — not the whole
          mind.
        </p>
      </div>
      <div className="mt-6 flex flex-wrap gap-2">
        <Button variant="outline" size="sm" onClick={() => setPanel("talk")}>
          Close
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            reset();
            setPanel("talk");
          }}
        >
          Reset soma
        </Button>
      </div>
    </div>
  );
}
