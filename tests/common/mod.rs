//! exact laws of configurations and a goodness-of-fit check against them,
//! shared by the integration tests
#![allow(dead_code)]

use std::collections::BTreeMap;

use ewens_sampler::samplers::Sampler;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// probability of each configuration, keyed by its sorted counts
pub type Law = BTreeMap<Vec<u16>, f64>;

/// number of configurations drawn by `assert_fits`
pub const N_SAMPLES: usize = 100_000;

/// all partitions of n into exactly k parts, as ascending vectors
pub fn partitions(n: usize, k: usize) -> Vec<Vec<u16>> {
    // parts in descending order, none larger than max
    fn go(n: usize, k: usize, max: usize, out: &mut Vec<Vec<u16>>, prefix: &mut Vec<u16>) {
        if k == 0 {
            if n == 0 {
                out.push(prefix.iter().rev().copied().collect());
            }
            return;
        }
        for part in (1..=max.min(n)).rev() {
            prefix.push(part as u16);
            go(n - part, k - 1, part, out, prefix);
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    go(n, k, n, &mut out, &mut Vec::new());
    out
}

pub fn factorial(m: usize) -> f64 {
    (1..=m).map(|i| i as f64).product()
}

fn rising(x: f64, m: usize) -> f64 {
    (0..m).map(|i| x + i as f64).product()
}

// prod_j a_j! where a_j is the number of alleles with j copies
pub fn multiplicity_factorials(c: &[u16]) -> f64 {
    c.chunk_by(|a, b| a == b)
        .map(|g| factorial(g.len()))
        .product()
}

fn normalize(law: impl IntoIterator<Item = (Vec<u16>, f64)>) -> Law {
    let mut law: Law = law.into_iter().collect();
    let z: f64 = law.values().sum();
    law.values_mut().for_each(|p| *p /= z);
    law
}

/// closed-form pitman-yor law of the configuration given n and k (theta cancels):
///   P(c | n, k) ∝ prod_b (1-alpha)_(c_b - 1) / c_b!  /  prod_j a_j!
/// alpha = 0 is the ewens sampling formula, P(c | n, k) ∝ 1 / (prod_b c_b prod_j a_j!).
/// the limits are alpha = 1, with all the mass on (1, ..., 1, n - k + 1), and alpha = -inf, where
/// (1-alpha)_(c-1) ~ |alpha|^(c-1) and prod_b |alpha|^(c_b - 1) = |alpha|^(n-k) cancels
pub fn py_law(n: usize, k: usize, alpha: f64) -> Law {
    if alpha == 1.0 {
        let mut c = vec![1; k];
        c[k - 1] = (n - k + 1) as u16;
        return Law::from([(c, 1.0)]);
    }
    normalize(partitions(n, k).into_iter().map(|c| {
        let w = py_weight(&c, alpha);
        (c, w)
    }))
}

// prod_b (1-alpha)_(c_b - 1) / c_b!  /  prod_j a_j!, which at alpha = -inf leaves out the |alpha|^(n-k)
fn py_weight(c: &[u16], alpha: f64) -> f64 {
    c.iter()
        .map(|&m| {
            let m = m as usize;
            let rise = if alpha == f64::NEG_INFINITY {
                1.0
            } else {
                rising(1.0 - alpha, m - 1)
            };
            rise / factorial(m)
        })
        .product::<f64>()
        / multiplicity_factorials(c)
}

/// pitman-yor law without conditioning on k (pitman's sampling formula):
///   P(c | n) ∝ prod_(i=1)^(k-1) (theta + i alpha) prod_b (1-alpha)_(c_b - 1) / c_b!  /  prod_j a_j!
/// for alpha < 0 theta has to be m |alpha|, and then the first product rules out more than m alleles.
/// impossible configurations are left out, so the law has exactly the possible ones
pub fn py_free_law(n: usize, theta: f64, alpha: f64) -> Law {
    normalize((1..=n).flat_map(|k| partitions(n, k)).filter_map(|c| {
        let w = (1..c.len())
            .map(|i| theta + i as f64 * alpha)
            .product::<f64>()
            * py_weight(&c, alpha);
        (w > 0.0).then_some((c, w))
    }))
}

/// ewens sampling formula without conditioning on k
pub fn esf_law(n: usize, theta: f64) -> Law {
    normalize((1..=n).flat_map(|k| partitions(n, k)).map(|c| {
        let w = theta.powi(c.len() as i32)
            / c.iter().map(|&m| m as f64).product::<f64>()
            / multiplicity_factorials(&c);
        (c, w)
    }))
}

/// exact probability law of the configuration of a chinese restaurant after n customers,
/// starting from tables `init`, computed by enumerating every seating.
/// with j customers seated at x tables, the next one  opens a new table w.p. new_table(x, j),
/// and otherwise joins a table of size c w.p. ∝ join(c).
/// moves with probability 0 are left out, so the law has exactly the possible configurations
/// (like a pitman-yor restaurant with alpha < 0 stopping at m tables, or alpha = 1 never joining one)
fn seat(
    n: usize,
    init: &[u16],
    new_table: impl Fn(usize, usize) -> f64,
    join: impl Fn(u16) -> f64,
) -> Law {
    let mut start = init.to_vec();
    start.sort_unstable();
    let n0: usize = init.iter().map(|&c| c as usize).sum();

    let mut states = Law::from([(start, 1.0)]);
    for j in n0..n {
        let mut next = Law::new();
        for (tables, &prob) in &states {
            let p_new = new_table(tables.len(), j);
            if p_new > 0.0 {
                let mut t = tables.clone();
                t.push(1);
                t.sort_unstable();
                *next.entry(t).or_default() += prob * p_new;
            }

            let total: f64 = tables.iter().map(|&c| join(c)).sum();
            for i in 0..tables.len() {
                // with p_new = 1 this can be 0 / 0, which is NaN and also left out
                let p_join = (1.0 - p_new) * join(tables[i]) / total;
                if p_join > 0.0 {
                    let mut t = tables.clone();
                    t[i] += 1;
                    t.sort_unstable();
                    *next.entry(t).or_default() += prob * p_join;
                }
            }
        }
        states = next;
    }
    states
}

/// pitman-yor chinese restaurant (alpha = 0 is the ewens one)
pub fn py_crp_law(n: usize, theta: f64, alpha: f64, init: &[u16]) -> Law {
    seat(
        n,
        init,
        |x, j| (theta + x as f64 * alpha) / (theta + j as f64),
        |c| c as f64 - alpha,
    )
}

/// the frequency-biased restaurant of BiasedCRPSampler
pub fn biased_crp_law(n: usize, theta: f64, bias: f64, init: &[u16]) -> Law {
    seat(
        n,
        init,
        |_, j| theta / (theta + j as f64),
        |c| (c as f64).powf(bias),
    )
}

/// condition a law on having k alleles
pub fn given_k(law: Law, k: usize) -> Law {
    normalize(law.into_iter().filter(|(c, _)| c.len() == k))
}

/// pearson chi-square test of N_SAMPLES sorted configurations from `sampler` against `law`.
/// cells with fewer than 5 expected counts are merged, and the test fails past the ~1e-6
/// upper quantile of chi^2, so with a fixed seed a false alarm is practically impossible
pub fn assert_fits(label: &str, sampler: &dyn Sampler<Pcg64>, law: &Law, seed: u64) {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut counts: BTreeMap<Vec<u16>, usize> = BTreeMap::new();
    for _ in 0..N_SAMPLES {
        let mut c = sampler.sample(&mut rng);
        c.sort_unstable();
        assert!(
            law.contains_key(&c),
            "{label}: sampled impossible configuration {c:?}"
        );
        *counts.entry(c).or_default() += 1;
    }

    // (observed, expected) per configuration, merging the rarest ones until each cell has e >= 5
    let mut cells: Vec<(f64, f64)> = law
        .iter()
        .map(|(c, &p)| {
            (
                counts.get(c).copied().unwrap_or(0) as f64,
                p * N_SAMPLES as f64,
            )
        })
        .collect();
    cells.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let mut acc = (0.0, 0.0);
    for (o, e) in cells {
        acc = (acc.0 + o, acc.1 + e);
        if acc.1 >= 5.0 {
            bins.push(acc);
            acc = (0.0, 0.0);
        }
    }
    if acc.1 > 0.0 {
        match bins.last_mut() {
            Some(b) => *b = (b.0 + acc.0, b.1 + acc.1),
            None => bins.push(acc),
        }
    }
    if bins.len() < 2 {
        // a single possible configuration, and we already checked the support
        return;
    }

    let stat: f64 = bins.iter().map(|&(o, e)| (o - e).powi(2) / e).sum();
    let df = (bins.len() - 1) as f64;
    // wilson-hilferty approximation of the chi^2_df quantile at z = 4.75 (~1e-6 upper tail)
    let crit = df * (1.0 - 2.0 / (9.0 * df) + 4.75 * (2.0 / (9.0 * df)).sqrt()).powi(3);
    assert!(
        stat < crit,
        "{label}: chi^2 = {stat:.1} on {df} df, past the critical value {crit:.1}"
    );
}
