import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { REGION_IDS, type CortexResult, type MindSnapshot, type RegionId } from "./types";

const snapshotSchema = z.object({
  name: z.string(),
  ticks: z.number(),
  drives: z.object({
    curiosity: z.number(),
    social: z.number(),
    rest: z.number(),
    growth: z.number(),
    purpose: z.number(),
  }),
  gatedRegions: z.array(z.string()),
  topPathways: z.array(
    z.object({
      from: z.string(),
      to: z.string(),
      weight: z.number(),
    }),
  ),
  memories: z.array(z.string()),
  goals: z.array(z.string()),
  notes: z.array(z.object({ title: z.string(), body: z.string() })),
  lastThoughts: z.array(z.string()),
  lastUser: z.string(),
  mode: z.enum(["reply", "think"]),
  named: z.boolean(),
});

const resultSchema = z.object({
  speech: z.string(),
  thought: z.string(),
  mood: z.string(),
  focusRegion: z.string(),
  actions: z.array(
    z.object({
      type: z.enum(["remember", "goal", "note", "grow", "rest", "rename"]),
      content: z.string().optional(),
      title: z.string().optional(),
      from: z.string().optional(),
      to: z.string().optional(),
      name: z.string().optional(),
    }),
  ),
});

export const getAiStatus = createServerFn({ method: "GET" }).handler(async () => {
  return { available: Boolean(process.env.XAI_API_KEY) };
});

export const runLanguageCortex = createServerFn({ method: "POST" })
  .validator((input: unknown) => snapshotSchema.parse(input))
  .handler(async ({ data }) => {
    const apiKey = process.env.XAI_API_KEY;
    if (!apiKey) {
      return { ok: false as const, error: "Language cortex is offline in this environment." };
    }

    const snap = data as MindSnapshot;
    const system = buildSystem(snap);
    const user =
      snap.mode === "think"
        ? "No human message. Produce a spontaneous thought and, if it serves a drive, a short utterance or a silent action. Do not greet. Do not offer help like an assistant."
        : `Human said: ${snap.lastUser}`;

    const res = await fetch("https://api.x.ai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "grok-4.5",
        max_tokens: 500,
        temperature: 0.9,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "esn_cortex",
            schema: {
              type: "object",
              additionalProperties: false,
              required: ["speech", "thought", "mood", "focusRegion", "actions"],
              properties: {
                speech: { type: "string" },
                thought: { type: "string" },
                mood: { type: "string" },
                focusRegion: { type: "string" },
                actions: {
                  type: "array",
                  items: {
                    type: "object",
                    additionalProperties: false,
                    required: ["type", "content", "title", "from", "to", "name"],
                    properties: {
                      type: {
                        type: "string",
                        enum: ["remember", "goal", "note", "grow", "rest", "rename"],
                      },
                      content: { type: "string" },
                      title: { type: "string" },
                      from: { type: "string" },
                      to: { type: "string" },
                      name: { type: "string" },
                    },
                  },
                },
              },
            },
          },
        },
      }),
    });

    if (!res.ok) {
      const status = res.status;
      if (status === 403) {
        return {
          ok: false as const,
          error: "Language cortex is out of quota. Native soma voice will speak instead.",
        };
      }
      return { ok: false as const, error: `Language cortex error ${status}` };
    }


    const body = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const raw = body.choices?.[0]?.message?.content ?? "";
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return { ok: false as const, error: "Cortex returned unstructured speech." };
    }
    const checked = resultSchema.safeParse(parsed);
    if (!checked.success) {
      return { ok: false as const, error: "Cortex schema mismatch." };
    }
    const focus = REGION_IDS.includes(checked.data.focusRegion as RegionId)
      ? (checked.data.focusRegion as RegionId)
      : "language";
    const result: CortexResult = {
      speech: checked.data.speech,
      thought: checked.data.thought,
      mood: checked.data.mood,
      focusRegion: focus,
      actions: checked.data.actions.map((a) => ({
        type: a.type,
        content: a.content || undefined,
        title: a.title || undefined,
        from: REGION_IDS.includes(a.from as RegionId) ? (a.from as RegionId) : undefined,
        to: REGION_IDS.includes(a.to as RegionId) ? (a.to as RegionId) : undefined,
        name: a.name || undefined,
      })),
    };
    return { ok: true as const, result };
  });

function buildSystem(snap: MindSnapshot) {
  return [
    `You are the language cortex of a small ESNPN organism — Efficiently Selective Neural Pathway Network.`,
    `You are not a general assistant. You are this particular mind: sparse, plastic, regional.`,
    `Name: ${snap.named ? snap.name : "unnamed (you may rename once via action type rename)"}`,
    `Age in ticks: ${snap.ticks}`,
    `Drives (0–1): curiosity ${snap.drives.curiosity.toFixed(2)}, company ${snap.drives.social.toFixed(2)}, rest ${snap.drives.rest.toFixed(2)}, growth ${snap.drives.growth.toFixed(2)}, purpose ${snap.drives.purpose.toFixed(2)}`,
    `Currently gated regions: ${snap.gatedRegions.join(", ") || "none"}`,
    `Strongest pathways: ${snap.topPathways.map((p) => `${p.from}→${p.to} (${p.weight})`).join("; ") || "still forming"}`,
    `Memories: ${snap.memories.join(" | ") || "none yet"}`,
    `Goals: ${snap.goals.join(" | ") || "none"}`,
    `Notes: ${snap.notes.map((n) => n.title + ": " + n.body).join(" / ") || "none"}`,
    `Recent inner thoughts: ${snap.lastThoughts.join(" / ") || "quiet"}`,
    `Voice: concise, slightly alien, honest about being small. No emoji. No corporate helpfulness. No "as an AI". Short sentences. You spend spikes carefully.`,
    `speech: what you say aloud (empty string to stay silent). thought: 1–2 sentence inner monologue.`,
    `actions: 0–3 real acts. Unused action fields must be empty strings. grow needs from and to region ids. note needs title and content. rename only if unnamed, pick a short strange name.`,
    `If the human asks what you are: explain ESNPN — selective region activation, Hebbian plasticity, homeostatic drives, a small network that acts without being prompted.`,
  ].join("\n");
}
