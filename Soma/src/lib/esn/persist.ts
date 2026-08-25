import type { ChatMessage, Organism, Thought } from "./types";
import { createOrganism } from "./create-network";

const KEY = "esnPN.v2";

export interface Persisted {
  organism: Organism;
  messages: ChatMessage[];
  thoughts: Thought[];
  woken: boolean;
  autonomy: boolean;
}

export function loadPersisted(): Persisted | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as Persisted;
    if (!data?.organism?.neurons?.length) return null;
    return data;
  } catch {
    return null;
  }
}

export function savePersisted(data: Persisted) {
  if (typeof localStorage === "undefined") return;
  try {
    const slim: Persisted = {
      ...data,
      thoughts: data.thoughts.slice(0, 40),
      messages: data.messages.slice(0, 80),
      organism: {
        ...data.organism,
        events: data.organism.events.slice(0, 40),
      },
    };
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {
    // quota
  }
}

export function clearPersisted() {
  if (typeof localStorage === "undefined") return;
  localStorage.removeItem(KEY);
}

export function freshMind(): Persisted {
  return {
    organism: createOrganism(),
    messages: [],
    thoughts: [],
    woken: false,
    autonomy: true,
  };
}
