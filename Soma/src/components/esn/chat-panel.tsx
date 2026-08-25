import { useState, type FormEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getRuntime, useMind } from "@/lib/esn/store";
import { cn } from "@/lib/utils";

export function ChatPanel() {
  const version = useMind((s) => s.version);
  const thinking = useMind((s) => s.thinking);
  const send = useMind((s) => s.send);
  const aiError = useMind((s) => s.aiError);
  void version;
  const messages = getRuntime().messages;
  const [draft, setDraft] = useState("");

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const t = draft;
    setDraft("");
    void send(t);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-1 py-2">
        {thinking ? (
          <p className="text-sm text-muted-foreground">
            <span className="shimmer">Gate routing through language…</span>
          </p>
        ) : null}
        {aiError ? <p className="text-xs text-muted-foreground">{aiError}</p> : null}
        {messages.length === 0 && !thinking ? (
          <p className="text-sm leading-relaxed text-muted-foreground">
            Speak to the soma. Ask what it is. Give it a problem. Watch which regions open.
          </p>
        ) : null}
        {messages.map((m) => (
          <article
            key={m.id}
            className={cn(
              "rounded-lg px-3 py-2.5 text-sm leading-relaxed",
              m.role === "human" && "bg-raised text-fg",
              m.role === "soma" && "bg-surface text-fg shadow-[var(--shadow-border)]",
              m.role === "system" && "text-muted-foreground",
            )}
          >
            {m.role !== "system" ? (
              <p className="mb-1 text-2xs font-medium tracking-wider text-faint uppercase">
                {m.role === "human" ? "You" : m.via === "native" ? "Soma · native" : "Soma · cortex"}
              </p>
            ) : null}
            <p>{m.text}</p>
          </article>
        ))}
      </div>
      <form onSubmit={onSubmit} className="flex gap-2 pt-3">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Talk to it"
          aria-label="Message the soma"
          disabled={thinking}
          maxLength={500}
        />
        <Button type="submit" size="icon" disabled={thinking || !draft.trim()} aria-label="Send">
          <ArrowUp />
        </Button>
      </form>
    </div>
  );
}
