// Scenario model: pilots (behaviors), presets, and translation to ScenarioConfig JSON.

/** Built-in pilots. `rl` pilots are `external` agents driven by a browser-side policy. */
export const BUILTIN_PILOTS = [
  { id: "rl_random", label: "RL slot, random policy", team: { behavior: "external" }, policy: "random", rl: true },
  { id: "rl_idle", label: "RL slot, straight and level", team: { behavior: "external" }, policy: "idle", rl: true },
  { id: "rl_manual", label: "RL slot, you fly it", team: { behavior: "external" }, policy: "manual", rl: true, liveOnly: true },
  { id: "ref_aggressive", label: "Reference: aggressive", reference: "aggressive" },
  { id: "ref_balanced", label: "Reference: balanced", reference: "balanced" },
  { id: "ref_conservative", label: "Reference: conservative", reference: "conservative" },
  { id: "fsm_b1", label: "FSM-B1 (paper thresholds)", team: { behavior: "baseline1", beh_config: { d_shot: [1.04], l_crank: [1.06], l_break: [1.05] } } },
  { id: "aggressive", label: "Searched: aggressive", profile: "aggressive" },
  { id: "cautious", label: "Searched: cautious", profile: "cautious" },
  { id: "fire_once", label: "Fire once", team: { behavior: "fire_once" } },
  { id: "duck", label: "Duck (flies to target)", team: { behavior: "duck" } },
];

const CUSTOM_KEY = "bace-lab-custom-pilots";

export class PilotBook {
  constructor(baselines, reference) {
    this.baselines = baselines || {};
    this.reference = reference || {};
    this.custom = [];
    try { this.custom = JSON.parse(localStorage.getItem(CUSTOM_KEY) || "[]"); } catch { this.custom = []; }
  }
  all({ live = true } = {}) {
    return [...BUILTIN_PILOTS, ...this.custom].filter((p) => live || !p.liveOnly);
  }
  get(id) {
    return this.all().find((p) => p.id === id) || BUILTIN_PILOTS[4];
  }
  /** Team-config patch for a pilot. */
  team(id) {
    const p = this.get(id);
    if (p.reference) {
      const b = this.reference[p.reference] || {};
      return { behavior: "baseline1", beh_config: b.beh_config };
    }
    if (p.profile) {
      const b = this.baselines[p.profile] || {};
      return { behavior: "baseline1", beh_config: b.beh_config };
    }
    return structuredClone(p.team);
  }
  addCustom(label, genome) {
    const id = `custom_${Date.now().toString(36)}`;
    this.custom.push({
      id, label, custom: true,
      team: { behavior: "baseline1", beh_config: { d_shot: [genome.d_shot], l_crank: [genome.l_crank], l_break: [genome.l_break] } },
    });
    localStorage.setItem(CUSTOM_KEY, JSON.stringify(this.custom));
    return id;
  }
}

const side = (o) => ({
  pilot: "ref_balanced", num_agents: 1, mission: "dca", share_tracks: true,
  init_x: 0, init_z: 30, init_y: 25000, init_hdg: 0, rnd_xz: 4, beh_noise: 0.1, ...o,
});

/** Lab presets. Spawn: blue south of the bullseye heading north, red north heading south. */
export const PRESETS = [
  {
    id: "b1_vs_striker", label: "1v1, stop a striker",
    note: "Red flies straight for blue’s base. Blue (reference balanced profile) must find it, shoot from inside its envelope, and support the missile until it goes active.",
    blue: side({ pilot: "ref_balanced" }),
    red: side({ pilot: "duck", mission: "striker", init_z: -30, init_hdg: 180 }),
    env: { seed: 1, max_cycles: 600 },
  },
  {
    id: "ref_duel", label: "1v1, aggressive vs conservative",
    note: "A long shot against a short one. Watch the range chart: who shoots first, who has to break, and the F-pole when the missile arrives.",
    blue: side({ pilot: "ref_aggressive" }),
    red: side({ pilot: "ref_conservative", init_z: -30, init_hdg: 180 }),
    env: { seed: 3, max_cycles: 700 },
  },
  {
    id: "high_low", label: "1v1, high vs low",
    note: "Same pilots, blue at 40,000 ft and red at 15,000 ft. Altitude stretches the missile envelope: the WEZ arcs show the asymmetry.",
    blue: side({ pilot: "ref_balanced", init_y: 40000 }),
    red: side({ pilot: "ref_balanced", init_z: -30, init_hdg: 180, init_y: 15000 }),
    env: { seed: 2, max_cycles: 700 },
  },
  {
    id: "rl_vs_b1", label: "1v1, RL slot vs balanced",
    note: "Blue is an external agent, here driven by a random policy. Select it to see exactly what a learning agent observes.",
    blue: side({ pilot: "rl_random" }),
    red: side({ pilot: "ref_balanced", init_z: -30, init_hdg: 180 }),
    env: { seed: 1, max_cycles: 400 },
  },
  {
    id: "manual", label: "1v1, fly it yourself",
    note: "You control blue with the arrow keys and Space. Red is a striker heading for your base.",
    blue: side({ pilot: "rl_manual" }),
    red: side({ pilot: "duck", mission: "striker", init_z: -30, init_hdg: 180 }),
    env: { seed: 1, max_cycles: 900 },
  },
  {
    id: "2v2", label: "2v2, balanced vs conservative",
    note: "Two-ship elements with shared datalink tracks. Each aircraft picks its own highest-priority target, and a wingman’s radar keeps a missile supported.",
    blue: side({ pilot: "ref_balanced", num_agents: 2 }),
    red: side({ pilot: "ref_conservative", num_agents: 2, init_z: -30, init_hdg: 180 }),
    env: { seed: 2, max_cycles: 700 },
  },
  {
    id: "4v4", label: "4v4, aggressive vs balanced",
    note: "The largest supported force. Four-ship boxes at 4 NM spacing.",
    blue: side({ pilot: "ref_aggressive", num_agents: 4 }),
    red: side({ pilot: "ref_balanced", num_agents: 4, init_z: -30, init_hdg: 180 }),
    env: { seed: 1, max_cycles: 700 },
  },
  {
    id: "selfplay", label: "2v2, self-play slots",
    note: "Both sides are external agents (random policy here), the setup used for self-play training.",
    blue: side({ pilot: "rl_random", num_agents: 2 }),
    red: side({ pilot: "rl_random", num_agents: 2, init_z: -30, init_hdg: 180 }),
    env: { seed: 1, max_cycles: 400 },
  },
];

/** Translate the Lab form state into a ScenarioConfig JSON (partial, merged over defaults). */
export function buildScenario(state, book) {
  const team = (s) => {
    const t = {
      ...book.team(s.pilot),
      num_agents: Number(s.num_agents),
      mission: s.mission,
      share_tracks: s.share_tracks === true || s.share_tracks === "true",
      init_position: { x: Number(s.init_x), y: Number(s.init_y), z: Number(s.init_z) },
      init_hdg: Number(s.init_hdg),
      rnd_offset_range: { x: Number(s.rnd_xz), y: 0, z: Number(s.rnd_xz) / 2 },
      beh_noise: Number(s.beh_noise),
    };
    return t;
  };
  const blue = team(state.blue);
  const red = team(state.red);
  // Targets: each side's target is the other side's spawn area.
  blue.target_position = { x: Number(state.red.init_x), y: Number(state.blue.init_y), z: Number(state.red.init_z) };
  red.target_position = { x: Number(state.blue.init_x), y: Number(state.red.init_y), z: Number(state.blue.init_z) };
  const env = {
    seed: Number(state.env.seed),
    max_cycles: Number(state.env.max_cycles),
    missile_support: state.env.missile_support || "track",
    missile_model: state.env.missile_model || "generic_mraam",
  };
  if (state.env.rewards) env.rewards = { ...state.env.rewards };
  return { env, blue, red };
}

export function pythonSnippet(cfg, state, book) {
  const pol = ["blue", "red"].map((s) => book.get(state[s].pilot)).filter((p) => p.rl);
  const json = JSON.stringify(cfg, null, 4)
    .replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None");
  return `from bace import BaceEnv

config = ${json}

env = BaceEnv(config)
obs, infos = env.reset(seed=config["env"]["seed"])
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}  # your policy here
    obs, rewards, terms, truncs, infos = env.step(actions)
print(env.outcome()["end"])
${pol.length ? "" : "# No RL slots: every aircraft is scripted, so env.agents still lists blue for observation.\n"}`;
}
