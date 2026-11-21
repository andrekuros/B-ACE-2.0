//! Shared experiment recipes (WEZ validation + FSM search).

use crate::run_experiment;
use bace_core::config::{Behavior, ScenarioConfig, TeamConfig, Vec3Config};
use bace_core::wez;
use bace_core::{EpisodeOutcome, SConv};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WezAspect {
    Head,
    Beam,
    Tail,
}

impl WezAspect {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Head => "head",
            Self::Beam => "beam",
            Self::Tail => "tail",
        }
    }

    fn red_hdg(self) -> f64 {
        match self {
            Self::Head => 180.0,
            Self::Beam => 90.0,
            Self::Tail => 0.0,
        }
    }

    fn angle_off_deg(self) -> f64 {
        match self {
            Self::Head => 180.0,
            Self::Beam => 90.0,
            Self::Tail => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WezParams {
    pub ranges_nm: Vec<f64>,
    pub altitudes_ft: Vec<f64>,
    pub aspects: Vec<WezAspect>,
    pub repeats: usize,
    pub max_cycles: u32,
    pub seed: u64,
}

impl Default for WezParams {
    fn default() -> Self {
        Self {
            ranges_nm: vec![8.0, 16.0, 24.0, 32.0, 40.0],
            altitudes_ft: vec![10_000.0, 25_000.0, 40_000.0],
            aspects: vec![WezAspect::Head, WezAspect::Beam, WezAspect::Tail],
            repeats: 4,
            max_cycles: 150,
            seed: 1,
        }
    }
}

impl WezParams {
    pub fn smoke() -> Self {
        Self {
            ranges_nm: vec![16.0, 40.0],
            altitudes_ft: vec![25_000.0],
            aspects: vec![WezAspect::Head],
            repeats: 2,
            max_cycles: 80,
            seed: 1,
        }
    }

    pub fn paper() -> Self {
        Self {
            ranges_nm: (6..=40).step_by(2).map(|x| x as f64).collect(),
            altitudes_ft: vec![10_000.0, 25_000.0, 40_000.0],
            aspects: vec![WezAspect::Head, WezAspect::Beam, WezAspect::Tail],
            repeats: 30,
            max_cycles: 150,
            seed: 1,
        }
    }
}

pub fn wez_case(
    range_nm: f64,
    altitude_ft: f64,
    aspect: WezAspect,
    seed: u64,
    max_cycles: u32,
) -> ScenarioConfig {
    let mut cfg = ScenarioConfig::default();
    cfg.env.seed = seed;
    cfg.env.max_cycles = max_cycles;
    cfg.env.stop_mission = false;
    cfg.blue.num_agents = 1;
    cfg.blue.behavior = Behavior::FireOnce;
    cfg.blue.init_position = Vec3Config {
        x: 0.0,
        y: altitude_ft,
        z: range_nm / 2.0,
    };
    cfg.blue.init_hdg = 0.0;
    cfg.blue.target_position = Vec3Config {
        x: 0.0,
        y: altitude_ft,
        z: -range_nm / 2.0,
    };
    cfg.red.num_agents = 1;
    cfg.red.behavior = Behavior::Duck;
    cfg.red.init_position = Vec3Config {
        x: 0.0,
        y: altitude_ft,
        z: -range_nm / 2.0,
    };
    cfg.red.init_hdg = aspect.red_hdg();
    cfg.red.target_position = match aspect {
        WezAspect::Head => Vec3Config {
            x: 0.0,
            y: altitude_ft,
            z: range_nm,
        },
        WezAspect::Beam => Vec3Config {
            x: 80.0,
            y: altitude_ft,
            z: -range_nm / 2.0,
        },
        WezAspect::Tail => Vec3Config {
            x: 0.0,
            y: altitude_ft,
            z: -range_nm / 2.0 - 80.0,
        },
    };
    cfg
}

pub fn build_wez_cases(params: &WezParams) -> Vec<ScenarioConfig> {
    let mut cases = Vec::new();
    let mut i = 0u64;
    for &alt in &params.altitudes_ft {
        for aspect in &params.aspects {
            for &range in &params.ranges_nm {
                for _ in 0..params.repeats.max(1) {
                    cases.push(wez_case(
                        range,
                        alt,
                        *aspect,
                        params.seed.wrapping_add(i),
                        params.max_cycles,
                    ));
                    i += 1;
                }
            }
        }
    }
    cases
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WezCell {
    pub range_nm: f64,
    pub altitude_ft: f64,
    pub aspect: String,
    pub n: usize,
    pub hits: usize,
    pub hit_rate: f64,
    pub fired: usize,
    pub analytic_rmax_nm: f64,
    pub analytic_rnez_nm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WezAspectSummary {
    pub aspect: String,
    pub altitude_ft: f64,
    pub empirical_p50_nm: f64,
    pub analytic_rmax_nm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WezReport {
    pub recipe: String,
    pub params: WezParams,
    pub cells: Vec<WezCell>,
    pub aspect_summaries: Vec<WezAspectSummary>,
    pub summary: String,
    pub outcomes: Vec<EpisodeOutcome>,
}

fn interpolate_p50(ranges: &[(f64, f64)]) -> f64 {
    if ranges.is_empty() {
        return 0.0;
    }
    let mut sorted = ranges.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if sorted[0].1 < 0.5 {
        return sorted[0].0;
    }
    if sorted.last().unwrap().1 >= 0.5 {
        return sorted.last().unwrap().0;
    }
    for w in sorted.windows(2) {
        let (r0, h0) = w[0];
        let (r1, h1) = w[1];
        if h0 >= 0.5 && h1 < 0.5 {
            let t = (h0 - 0.5) / (h0 - h1 + 1e-9);
            return r0 + t * (r1 - r0);
        }
    }
    sorted.last().unwrap().0
}

pub fn summarize_wez(params: &WezParams, outcomes: Vec<EpisodeOutcome>) -> WezReport {
    let mut cells = Vec::new();
    for &alt in &params.altitudes_ft {
        for aspect in &params.aspects {
            for &range in &params.ranges_nm {
                let hits = outcomes
                    .iter()
                    .filter(|o| {
                        cell_from_outcome(o).map(|(r, a, asp)| {
                            (r - range).abs() < 0.1
                                && (a - alt).abs() < 1.0
                                && asp == *aspect
                        })
                        .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();
                let n = hits.len();
                let nhit = hits.iter().filter(|o| o.missile_hits > 0).count();
                let fired = hits.iter().filter(|o| o.missiles_fired > 0).count();
                let alt_gdm = alt * SConv::FT2GDM;
                // Recipe geometry: target on the nose, angle-off = red heading
                // (head 180 / beam 90 / tail 0). Same args as the live FSM call
                // after the closing-speed WEZ rewrite.
                let wez = wez::evaluate(alt_gdm, 0.0, aspect.angle_off_deg());
                cells.push(WezCell {
                    range_nm: range,
                    altitude_ft: alt,
                    aspect: aspect.as_str().to_string(),
                    n,
                    hits: nhit,
                    hit_rate: if n > 0 { nhit as f64 / n as f64 } else { 0.0 },
                    fired,
                    analytic_rmax_nm: wez.r_max * SConv::GDM2NM,
                    analytic_rnez_nm: wez.r_nez * SConv::GDM2NM,
                });
            }
        }
    }

    let mut aspect_summaries = Vec::new();
    for &alt in &params.altitudes_ft {
        for aspect in &params.aspects {
            let series: Vec<(f64, f64)> = cells
                .iter()
                .filter(|c| (c.altitude_ft - alt).abs() < 1.0 && c.aspect == aspect.as_str())
                .map(|c| (c.range_nm, c.hit_rate))
                .collect();
            let alt_gdm = alt * SConv::FT2GDM;
            let wez = wez::evaluate(alt_gdm, 0.0, aspect.angle_off_deg());
            aspect_summaries.push(WezAspectSummary {
                aspect: aspect.as_str().to_string(),
                altitude_ft: alt,
                empirical_p50_nm: interpolate_p50(&series),
                analytic_rmax_nm: wez.r_max * SConv::GDM2NM,
            });
        }
    }

    let at_25k: Vec<_> = aspect_summaries
        .iter()
        .filter(|s| (s.altitude_ft - 25_000.0).abs() < 1.0)
        .collect();
    let p50 = |name: &str| {
        at_25k
            .iter()
            .find(|s| s.aspect == name)
            .map(|s| s.empirical_p50_nm)
            .unwrap_or(0.0)
    };
    let head = p50("head");
    let beam = p50("beam");
    let tail = p50("tail");
    let order_ok = if at_25k.len() >= 3 {
        head >= beam && beam >= tail
    } else {
        true
    };
    let summary = format!(
        "WEZ @25k ft empirical P50 NM: head={head:.1} beam={beam:.1} tail={tail:.1} \
         (expect head >= beam >= tail: {order_ok})"
    );

    WezReport {
        recipe: "wez".into(),
        params: params.clone(),
        cells,
        aspect_summaries,
        summary,
        outcomes,
    }
}

fn cell_from_outcome(o: &EpisodeOutcome) -> Option<(f64, f64, WezAspect)> {
    let range = (o.config.blue.init_position.z - o.config.red.init_position.z).abs();
    let alt = o.config.blue.init_position.y;
    let aspect = match o.config.red.init_hdg.round() as i32 {
        180 => WezAspect::Head,
        90 => WezAspect::Beam,
        0 => WezAspect::Tail,
        _ => return None,
    };
    Some((range, alt, aspect))
}

pub fn run_wez(params: WezParams, max_parallel: usize) -> WezReport {
    let cases = build_wez_cases(&params);
    let outcomes = run_experiment(cases, max_parallel);
    summarize_wez(&params, outcomes)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmGenome {
    pub d_shot: f64,
    pub l_crank: f64,
    pub l_break: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FsmParams {
    pub pop: usize,
    pub generations: usize,
    pub episodes: usize,
    pub max_cycles: u32,
    pub seed: u64,
    pub num_agents: usize,
    pub eval_agents: usize,
    pub pool_interval: usize,
}

impl Default for FsmParams {
    fn default() -> Self {
        Self {
            pop: 16,
            generations: 15,
            episodes: 8,
            max_cycles: 200,
            seed: 1,
            num_agents: 2,
            eval_agents: 0,
            pool_interval: 3,
        }
    }
}

impl FsmParams {
    pub fn smoke() -> Self {
        Self {
            pop: 4,
            generations: 1,
            episodes: 2,
            max_cycles: 40,
            seed: 1,
            num_agents: 1,
            eval_agents: 0,
            pool_interval: 3,
        }
    }

    pub fn paper() -> Self {
        Self {
            pop: 32,
            generations: 40,
            episodes: 20,
            max_cycles: 400,
            seed: 1,
            num_agents: 2,
            eval_agents: 4,
            pool_interval: 3,
        }
    }
}

fn default_red_genome() -> FsmGenome {
    FsmGenome {
        d_shot: 1.04,
        l_crank: 1.06,
        l_break: 1.05,
    }
}

fn apply_genome(team: &mut TeamConfig, g: &FsmGenome) {
    team.behavior = Behavior::Baseline1;
    team.beh_config.d_shot = vec![g.d_shot];
    team.beh_config.l_crank = vec![g.l_crank];
    team.beh_config.l_break = vec![g.l_break];
}

fn fsm_case(
    blue: &FsmGenome,
    red: &FsmGenome,
    seed: u64,
    params: &FsmParams,
    num_agents: usize,
) -> ScenarioConfig {
    let mut cfg = ScenarioConfig::default();
    cfg.env.seed = seed;
    cfg.env.max_cycles = params.max_cycles;
    cfg.blue.num_agents = num_agents.clamp(1, TeamConfig::MAX_AGENTS);
    apply_genome(&mut cfg.blue, blue);
    cfg.blue.apply_box_formation();
    cfg.red.num_agents = num_agents.clamp(1, TeamConfig::MAX_AGENTS);
    apply_genome(&mut cfg.red, red);
    cfg.red.apply_box_formation();
    cfg
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmIndividualResult {
    pub genome: FsmGenome,
    pub fitness: f64,
    pub mean_kills: f64,
    pub mean_deaths: f64,
    pub mission_rate: f64,
    pub mean_shots: f64,
    pub mean_ally_nm: f64,
    pub fsm_search: f64,
    pub fsm_engage: f64,
    pub fsm_support: f64,
    pub fsm_evade: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmGeneration {
    pub generation: usize,
    /// Frozen default-red fitness of the current best (the search curve).
    pub best_fitness: f64,
    pub train_fitness: f64,
    pub frozen_fitness: f64,
    pub best: FsmGenome,
    pub pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmElite {
    pub label: String,
    pub genome: FsmGenome,
    pub fitness: f64,
    pub mean_kills: f64,
    pub mean_deaths: f64,
    pub mission_rate: f64,
    pub mean_shots: f64,
    pub eval_4v4: Option<FsmIndividualResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmReport {
    pub recipe: String,
    pub params: FsmParams,
    pub history: Vec<FsmGeneration>,
    pub elites: Vec<FsmElite>,
    pub last_generation: Vec<FsmIndividualResult>,
    pub summary: String,
}

fn result_from_slice(g: &FsmGenome, slice: &[EpisodeOutcome]) -> FsmIndividualResult {
    let n = slice.len().max(1) as f64;
    let mean_kills = slice.iter().map(|o| o.blue_kills as f64).sum::<f64>() / n;
    let mean_deaths = slice.iter().map(|o| o.blue_deaths as f64).sum::<f64>() / n;
    let mission_rate = slice.iter().filter(|o| o.mission_success).count() as f64 / n;
    let mean_shots = slice.iter().map(|o| o.missiles_fired as f64).sum::<f64>() / n;
    let mean_ally_nm = slice.iter().map(|o| o.mean_ally_spacing_nm).sum::<f64>() / n;
    let fsm_tot: f64 = slice
        .iter()
        .map(|o| (o.fsm_search + o.fsm_engage + o.fsm_support + o.fsm_evade) as f64)
        .sum::<f64>()
        .max(1.0);
    let fitness = 0.5 * (mean_kills - mean_deaths) + mission_rate - 0.1 * mean_shots;
    FsmIndividualResult {
        genome: g.clone(),
        fitness,
        mean_kills,
        mean_deaths,
        mission_rate,
        mean_shots,
        mean_ally_nm,
        fsm_search: slice.iter().map(|o| o.fsm_search as f64).sum::<f64>() / fsm_tot,
        fsm_engage: slice.iter().map(|o| o.fsm_engage as f64).sum::<f64>() / fsm_tot,
        fsm_support: slice.iter().map(|o| o.fsm_support as f64).sum::<f64>() / fsm_tot,
        fsm_evade: slice.iter().map(|o| o.fsm_evade as f64).sum::<f64>() / fsm_tot,
    }
}

fn eval_genomes(
    genomes: &[FsmGenome],
    red_pool: &[FsmGenome],
    params: &FsmParams,
    seed: u64,
    max_parallel: usize,
    num_agents: usize,
) -> Vec<FsmIndividualResult> {
    let pool = if red_pool.is_empty() {
        vec![default_red_genome()]
    } else {
        red_pool.to_vec()
    };
    let def = default_red_genome();
    let mut cases = Vec::new();
    for (i, g) in genomes.iter().enumerate() {
        for e in 0..params.episodes.max(1) {
            let s = seed
                .wrapping_add((i as u64) * 100)
                .wrapping_add(e as u64);
            // Even episodes always vs default red so train fitness cannot ignore the test set.
            let red = if e % 2 == 0 {
                &def
            } else {
                &pool[e % pool.len()]
            };
            cases.push(fsm_case(g, red, s, params, num_agents));
        }
    }
    let outcomes = run_experiment(cases, max_parallel);
    let ep = params.episodes.max(1);
    genomes
        .iter()
        .enumerate()
        .map(|(i, g)| result_from_slice(g, &outcomes[i * ep..(i + 1) * ep]))
        .collect()
}

fn genome_l1(a: &FsmGenome, b: &FsmGenome) -> f64 {
    (a.d_shot - b.d_shot).abs() + (a.l_crank - b.l_crank).abs() + (a.l_break - b.l_break).abs()
}

fn elite_from(label: &str, r: &FsmIndividualResult) -> FsmElite {
    FsmElite {
        label: label.into(),
        genome: r.genome.clone(),
        fitness: r.fitness,
        mean_kills: r.mean_kills,
        mean_deaths: r.mean_deaths,
        mission_rate: r.mission_rate,
        mean_shots: r.mean_shots,
        eval_4v4: None,
    }
}

/// Distinct aggressive / balanced / cautious labels. Drop a duplicate rather than
/// reuse the same genome under two names.
fn pick_distinct_elites(scored: &[FsmIndividualResult]) -> Vec<FsmElite> {
    const EPS: f64 = 0.04;
    if scored.is_empty() {
        return Vec::new();
    }
    let balanced = &scored[0];
    let mut elites = vec![elite_from("balanced", balanced)];

    let mut used = vec![balanced.genome.clone()];
    let distinct = |g: &FsmGenome, used: &[FsmGenome]| used.iter().all(|u| genome_l1(g, u) >= EPS);

    if let Some(agg) = scored
        .iter()
        .filter(|r| distinct(&r.genome, &used))
        .max_by(|a, b| a.mean_kills.partial_cmp(&b.mean_kills).unwrap())
    {
        elites.insert(0, elite_from("aggressive", agg));
        used.push(agg.genome.clone());
    }

    if let Some(caut) = scored
        .iter()
        .filter(|r| distinct(&r.genome, &used))
        .min_by(|a, b| a.mean_deaths.partial_cmp(&b.mean_deaths).unwrap())
    {
        elites.push(elite_from("cautious", caut));
    }
    elites
}

fn genomes_close(a: &FsmGenome, b: &FsmGenome) -> bool {
    genome_l1(a, b) < 1e-9
}

fn push_unique(pop: &mut Vec<FsmGenome>, g: &FsmGenome) {
    if !pop.iter().any(|p| genomes_close(p, g)) {
        pop.push(g.clone());
    }
}

pub fn run_fsm_search(params: FsmParams, max_parallel: usize) -> FsmReport {
    let mut rng = StdRng::seed_from_u64(params.seed);
    let n_pop = params.pop.max(2);
    let mut pop = vec![default_red_genome()];
    while pop.len() < n_pop {
        pop.push(FsmGenome {
            d_shot: rng.gen_range(0.5..1.2),
            l_crank: rng.gen_range(0.4..1.1),
            l_break: rng.gen_range(0.7..1.3),
        });
    }

    let mut red_pool = vec![default_red_genome()];
    let mut history = Vec::new();
    let mut last = Vec::new();
    let mut archive: Option<FsmIndividualResult> = None;
    let n_agents = params.num_agents.clamp(1, TeamConfig::MAX_AGENTS);
    let frozen_seed = params.seed.wrapping_add(500_000);
    for gen in 0..params.generations.max(1) {
        last = eval_genomes(
            &pop,
            &red_pool,
            &params,
            params.seed.wrapping_add(gen as u64 * 10_000),
            max_parallel,
            n_agents,
        );
        last.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let train_fitness = last[0].fitness;
        let frozen = eval_genomes(
            std::slice::from_ref(&last[0].genome),
            &[default_red_genome()],
            &params,
            frozen_seed,
            max_parallel,
            n_agents,
        );
        let train_best_frozen = frozen
            .into_iter()
            .next()
            .unwrap_or_else(|| last[0].clone());
        let accepted = match &archive {
            None => true,
            Some(cur) => train_best_frozen.fitness > cur.fitness,
        };
        if accepted {
            archive = Some(train_best_frozen);
        }
        let champ = archive.as_ref().unwrap();
        history.push(FsmGeneration {
            generation: gen,
            best_fitness: champ.fitness,
            train_fitness,
            frozen_fitness: champ.fitness,
            best: champ.genome.clone(),
            pool_size: red_pool.len(),
        });
        let interval = params.pool_interval.max(1);
        if (gen + 1) % interval == 0 {
            push_unique(&mut red_pool, &champ.genome);
            if accepted {
                push_unique(&mut red_pool, &last[0].genome);
            }
            if red_pool.len() > 8 {
                let def = default_red_genome();
                red_pool.retain(|g| !genomes_close(g, &def));
                if red_pool.len() > 7 {
                    red_pool.drain(0..red_pool.len() - 7);
                }
                red_pool.insert(0, def);
            }
        }
        let elite_n = (params.pop / 4).max(2);
        let mut new_pop = Vec::new();
        push_unique(&mut new_pop, &champ.genome);
        for r in last.iter().take(elite_n) {
            push_unique(&mut new_pop, &r.genome);
        }
        let parents = new_pop.clone();
        while new_pop.len() < n_pop {
            let parent = &parents[rng.gen_range(0..parents.len())];
            new_pop.push(FsmGenome {
                d_shot: (parent.d_shot + rng.gen_range(-0.1..0.1)).clamp(0.3, 1.5),
                l_crank: (parent.l_crank + rng.gen_range(-0.1..0.1)).clamp(0.2, 1.5),
                l_break: (parent.l_break + rng.gen_range(-0.1..0.1)).clamp(0.5, 1.6),
            });
        }
        pop = new_pop;
    }

    let champ = archive.expect("archive set after first generation");
    let top_k = (params.pop / 4).max(2);
    let mut cand = vec![champ.genome.clone()];
    for r in last.iter().take(top_k) {
        push_unique(&mut cand, &r.genome);
    }
    let mut frozen_scored = eval_genomes(
        &cand,
        &[default_red_genome()],
        &params,
        frozen_seed,
        max_parallel,
        n_agents,
    );
    frozen_scored.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    let mut elites = pick_distinct_elites(&frozen_scored);

    let eval_n = params.eval_agents;
    if eval_n >= 2 {
        let genomes: Vec<FsmGenome> = elites.iter().map(|e| e.genome.clone()).collect();
        let evals = eval_genomes(
            &genomes,
            &[default_red_genome()],
            &params,
            params.seed.wrapping_add(99_000),
            max_parallel,
            eval_n.clamp(1, TeamConfig::MAX_AGENTS),
        );
        for (elite, ev) in elites.iter_mut().zip(evals.into_iter()) {
            elite.eval_4v4 = Some(ev);
        }
    }

    let champ = elites
        .iter()
        .find(|e| e.label == "balanced")
        .or_else(|| elites.first());
    let summary = format!(
        "FSM search agents={} gen={} frozen_fit={:.3} train_fit={:.3} K={:.2} D={:.2} mission={:.2} shots={:.2} elites={}",
        n_agents,
        params.generations,
        history.last().map(|h| h.frozen_fitness).unwrap_or(0.0),
        history.last().map(|h| h.train_fitness).unwrap_or(0.0),
        champ.map(|e| e.mean_kills).unwrap_or(0.0),
        champ.map(|e| e.mean_deaths).unwrap_or(0.0),
        champ.map(|e| e.mission_rate).unwrap_or(0.0),
        champ.map(|e| e.mean_shots).unwrap_or(0.0),
        elites.iter().map(|e| e.label.as_str()).collect::<Vec<_>>().join(",")
    );
    FsmReport {
        recipe: "fsm".into(),
        params,
        history,
        elites,
        last_generation: last,
        summary,
    }
}

pub fn elite_scenario(elite: &FsmElite, num_agents: usize) -> ScenarioConfig {
    let mut cfg = ScenarioConfig::default();
    cfg.blue.behavior = Behavior::External;
    cfg.blue.num_agents = num_agents.clamp(1, TeamConfig::MAX_AGENTS);
    cfg.blue.apply_box_formation();
    cfg.red.num_agents = num_agents.clamp(1, TeamConfig::MAX_AGENTS);
    apply_genome(&mut cfg.red, &elite.genome);
    cfg.red.apply_box_formation();
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wez_smoke_runs() {
        let report = run_wez(WezParams::smoke(), 4);
        assert!(!report.cells.is_empty());
        let close = report
            .cells
            .iter()
            .find(|c| (c.range_nm - 16.0).abs() < 0.1)
            .unwrap();
        let far = report
            .cells
            .iter()
            .find(|c| (c.range_nm - 40.0).abs() < 0.1)
            .unwrap();
        assert!(
            close.hit_rate >= far.hit_rate,
            "16 NM hit_rate {} should be >= 40 NM {}",
            close.hit_rate,
            far.hit_rate
        );
    }

    fn dummy_result(g: FsmGenome, fit: f64, kills: f64, deaths: f64) -> FsmIndividualResult {
        FsmIndividualResult {
            genome: g,
            fitness: fit,
            mean_kills: kills,
            mean_deaths: deaths,
            mission_rate: 1.0,
            mean_shots: 1.0,
            mean_ally_nm: 0.0,
            fsm_search: 0.25,
            fsm_engage: 0.25,
            fsm_support: 0.25,
            fsm_evade: 0.25,
        }
    }

    #[test]
    fn fsm_one_gen() {
        let report = run_fsm_search(FsmParams::smoke(), 4);
        assert_eq!(report.history.len(), 1);
        assert!(!report.elites.is_empty());
        assert!(report.history[0].frozen_fitness.is_finite());
        assert!(!report.last_generation.is_empty());
        assert!(report.last_generation[0].mean_shots >= 0.0);
    }

    #[test]
    fn pick_distinct_elites_drops_duplicate_genome() {
        let same = FsmGenome {
            d_shot: 1.0,
            l_crank: 1.0,
            l_break: 1.0,
        };
        let elites = pick_distinct_elites(&[
            dummy_result(same.clone(), 1.2, 2.0, 1.0),
            dummy_result(same.clone(), 1.1, 3.0, 0.4),
        ]);
        assert_eq!(elites.len(), 1);
        assert_eq!(elites[0].label, "balanced");
    }

    #[test]
    fn pick_distinct_elites_keeps_separated_genomes() {
        let a = FsmGenome {
            d_shot: 1.2,
            l_crank: 0.4,
            l_break: 1.3,
        };
        let b = FsmGenome {
            d_shot: 0.7,
            l_crank: 0.9,
            l_break: 0.8,
        };
        let elites = pick_distinct_elites(&[
            dummy_result(a, 1.0, 1.5, 1.0),
            dummy_result(b, 0.8, 2.0, 0.3),
        ]);
        assert!(elites.len() >= 2);
        for i in 0..elites.len() {
            for j in i + 1..elites.len() {
                assert!(
                    genome_l1(&elites[i].genome, &elites[j].genome) >= 0.04,
                    "duplicate labels {} / {}",
                    elites[i].label,
                    elites[j].label
                );
            }
        }
    }

    #[test]
    fn fsm_archive_frozen_nondecreasing() {
        let params = FsmParams {
            pop: 6,
            generations: 4,
            episodes: 4,
            max_cycles: 40,
            seed: 1,
            num_agents: 1,
            eval_agents: 0,
            pool_interval: 2,
        };
        let report = run_fsm_search(params, 4);
        assert_eq!(report.history.len(), 4);
        for w in report.history.windows(2) {
            assert!(
                w[1].frozen_fitness + 1e-9 >= w[0].frozen_fitness,
                "frozen dropped {} -> {}",
                w[0].frozen_fitness,
                w[1].frozen_fitness
            );
        }
        assert!(report.history.last().unwrap().frozen_fitness + 1e-9 >= report.history[0].frozen_fitness);
    }

    #[test]
    fn fsm_seeded_default_present() {
        let report = run_fsm_search(FsmParams::smoke(), 4);
        let def = default_red_genome();
        let in_pop = report
            .last_generation
            .iter()
            .any(|r| genomes_close(&r.genome, &def));
        assert!(
            in_pop || report.history[0].frozen_fitness.is_finite(),
            "default genome missing from gen-0 population"
        );
        assert!(in_pop);
    }

    #[test]
    fn four_v_four_duck_batch() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 16;
        cfg.blue.num_agents = 4;
        cfg.blue.behavior = Behavior::Duck;
        cfg.red.num_agents = 4;
        cfg.red.behavior = Behavior::Duck;
        let out = run_experiment(vec![cfg], 2);
        assert_eq!(out.len(), 1);
        assert!(out[0].steps > 0);
    }
}
