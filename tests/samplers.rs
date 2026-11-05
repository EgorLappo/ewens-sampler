//! distribution and invariant tests for the samplers

mod common;

use common::{
    Law, assert_fits, biased_crp_law, esf_law, factorial, given_k, multiplicity_factorials,
    partitions, py_crp_law, py_free_law, py_law,
};
use ewens_sampler::Configuration;
use ewens_sampler::samplers::Sampler;
use ewens_sampler::samplers::ewens::theta::theta_mle;
use ewens_sampler::samplers::ewens::{
    BiasedCRPSampler, CRPSamplerK, ConditionalCRPSampler, FellerSampler, FellerSamplerK,
};
use ewens_sampler::samplers::pitman_yor::theta::theta_mle as theta_mle_py;
use ewens_sampler::samplers::pitman_yor::{PYSampler, PYSamplerK, default_theta};
use ewens_sampler::stats::ewens::watterson_statistic;
use rand::SeedableRng;
use rand_pcg::Pcg64;

// conditional on k the results should not depend on theta, so every sampler is run with both values
const THETAS: [f64; 2] = [0.3, 8.0];

// (theta, alpha) for pitman-yor without conditioning on k, where theta > -alpha, or theta = m |alpha|
// with m types for alpha < 0. these m |alpha| are exact in floating point, so the restaurant
// in common stops at m tables exactly. the last two have a single possible configuration
const PY_FREE: [(f64, f64); 9] = [
    (1.0, 0.0),
    (2.0, 0.5),
    (-0.3, 0.6),
    (0.5, 0.9),
    (1.5, -0.5),
    (10.0, -2.0),
    (30.0, -2.5),
    (2.0, -2.0),
    (3.0, 1.0),
];

fn conf(counts: &[u16]) -> Configuration {
    let s: Vec<String> = counts.iter().map(u16::to_string).collect();
    s.join(" ").parse().unwrap()
}

fn assert_close(a: &Law, b: &Law) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    for (c, p) in a {
        assert!((p - b[c]).abs() < 1e-12, "{c:?}: {p} vs {}", b[c]);
    }
}

// enumerating the restaurant should agree with the closed-form laws,
// and conditioning on k should make theta drop out, including for pitman-yor
#[test]
fn restaurant_matches_closed_form() {
    for (n, k) in [(7, 1), (8, 3), (10, 4), (7, 7)] {
        assert_eq!(py_law(n, k, 0.0).len(), partitions(n, k).len());
        for alpha in [0.0, 0.5, -1.5] {
            // negative alpha needs theta >= k |alpha|
            let t = default_theta(k, alpha);
            for theta in [t, t + 2.6] {
                assert_close(
                    &given_k(py_crp_law(n, theta, alpha, &[]), k),
                    &py_law(n, k, alpha),
                );
            }
        }
    }
    assert_close(&py_crp_law(7, 1.7, 0.0, &[]), &esf_law(7, 1.7));

    // and without conditioning, where theta stays
    for (theta, alpha) in PY_FREE {
        assert_close(
            &py_crp_law(8, theta, alpha, &[]),
            &py_free_law(8, theta, alpha),
        );
    }
    assert_close(&py_free_law(7, 1.7, 0.0), &esf_law(7, 1.7));
}

// at alpha = -1 every unordered configuration (every way of giving the counts to the k alleles)
// is equally likely, so P(c | n, k) is proportional to the k! / prod_j a_j! of them
#[test]
fn minus_one_is_uniform_over_unordered_configurations() {
    for (n, k) in [(8, 3), (10, 4), (7, 7)] {
        let per_unordered: Vec<f64> = py_law(n, k, -1.0)
            .into_iter()
            .map(|(c, p)| p / (factorial(k) / multiplicity_factorials(&c)))
            .collect();
        for w in &per_unordered {
            assert!(
                (w - per_unordered[0]).abs() < 1e-10 * per_unordered[0],
                "n={n} k={k}: {per_unordered:?}"
            );
        }
    }
}

// the check below has to be able to fail: ewens samples are far from a pitman-yor probability law
#[test]
#[should_panic(expected = "chi^2")]
fn goodness_of_fit_rejects_wrong_law() {
    assert_fits(
        "crp vs py",
        &CRPSamplerK::new(1.0, 10, 4).unwrap(),
        &py_law(10, 4, 0.5),
        1,
    );
}

// *exact goodness of fit at small n

fn fits_esf_for_any_theta(name: &str, make: impl Fn(f64, usize, usize) -> Box<dyn Sampler<Pcg64>>) {
    // the last two have a single possible configuration
    for (n, k) in [(10, 4), (9, 2), (7, 1), (6, 6)] {
        let law = py_law(n, k, 0.0);
        for theta in THETAS {
            let label = format!("{name} n={n} k={k} theta={theta}");
            assert_fits(&label, &*make(theta, n, k), &law, 1);
        }
    }
}

#[test]
fn feller_k_fits_esf() {
    fits_esf_for_any_theta("feller", |t, n, k| {
        Box::new(FellerSamplerK::new(t, n, k).unwrap())
    });
}

#[test]
fn crp_k_fits_esf() {
    fits_esf_for_any_theta("crp", |t, n, k| {
        Box::new(CRPSamplerK::new(t, n, k).unwrap())
    });
}

#[test]
fn conditional_crp_fits_esf() {
    fits_esf_for_any_theta("conditional crp", |t, n, k| {
        Box::new(ConditionalCRPSampler::new(t, n, k, None).unwrap())
    });
}

// bias = 1 should be the neutral model exactly
#[test]
fn biased_crp_without_bias_fits_esf() {
    fits_esf_for_any_theta("biased crp, bias 1", |t, n, k| {
        Box::new(BiasedCRPSampler::new(t, n, k, 1.0, None).unwrap())
    });
}

#[test]
fn feller_fits_unconditional_esf() {
    for theta in [0.5, 3.0] {
        let label = format!("feller theta={theta}");
        assert_fits(
            &label,
            &FellerSampler::new(theta, 7).unwrap(),
            &esf_law(7, theta),
            1,
        );
    }
}

#[test]
fn conditional_crp_with_initial_configuration_fits_exact_law() {
    // the last one has no new alleles to add
    for (n, k, init) in [(10, 4, &[1, 2][..]), (9, 3, &[3]), (8, 2, &[1, 2])] {
        let law = given_k(py_crp_law(n, 1.0, 0.0, init), k);
        for theta in THETAS {
            let sampler = ConditionalCRPSampler::new(theta, n, k, Some(&conf(init))).unwrap();
            let label = format!("conditional crp n={n} k={k} init={init:?} theta={theta}");
            assert_fits(&label, &sampler, &law, 1);
        }
    }
}

#[test]
fn biased_crp_fits_exact_law() {
    for bias in [0.5, 1.6] {
        for (n, k, init) in [(9, 3, &[][..]), (9, 4, &[1, 2])] {
            let law = given_k(biased_crp_law(n, 1.0, bias, init), k);
            let ic = (!init.is_empty()).then(|| conf(init));
            for theta in THETAS {
                let sampler = BiasedCRPSampler::new(theta, n, k, bias, ic.as_ref()).unwrap();
                let label =
                    format!("biased crp bias={bias} n={n} k={k} init={init:?} theta={theta}");
                assert_fits(&label, &sampler, &law, 1);
            }
        }
    }
}

// conditional on k the pitman-yor law depends only on alpha (alpha = 0 is ewens),
// for any theta > -alpha
#[test]
fn pitman_yor_k_fits_exact_law() {
    // alpha = 1 is the limit with a single possible configuration
    fits_py_law_for_any_theta(&[0.0, 0.5, 0.9, 1.0]);
}

#[test]
fn pitman_yor_k_fits_exact_law_for_negative_alpha() {
    fits_py_law_for_any_theta(&[-0.5, -1.0, -2.0, f64::NEG_INFINITY]);
}

#[test]
fn pitman_yor_fits_unconditional_law() {
    let n = 10;
    for (theta, alpha) in PY_FREE {
        let label = format!("pitman-yor alpha={alpha} theta={theta}");
        let law = py_free_law(n, theta, alpha);
        assert_fits(&label, &PYSampler::new(theta, alpha, n).unwrap(), &law, 1);
    }
}

fn fits_py_law_for_any_theta(alphas: &[f64]) {
    for &alpha in alphas {
        // the last two have a single possible configuration
        for (n, k) in [(10, 4), (9, 2), (7, 1), (6, 6)] {
            let law = py_law(n, k, alpha);
            // theta has to be > -alpha, and >= k |alpha| for alpha < 0
            let t = default_theta(k, alpha);
            for theta in [t, t + 7.7]
                .into_iter()
                .chain((alpha > 0.4).then_some(-0.4))
            {
                let sampler = PYSamplerK::new(theta, alpha, n, k).unwrap();
                let label = format!("pitman-yor alpha={alpha} n={n} k={k} theta={theta}");
                assert_fits(&label, &sampler, &law, 1);
            }
        }
    }
}

// the mle is what the sampler gets in practice, and unlike default_theta it can be negative,
// like for alpha = 0.9 with k = 2, or exactly k |alpha|, like for alpha = -10 with k = 4
#[test]
fn pitman_yor_k_fits_exact_law_at_the_mle() {
    for alpha in [0.5, 0.9, -1.0, -10.0] {
        for (n, k) in [(10, 4), (9, 2)] {
            let theta = theta_mle_py(n, k, alpha, None);
            let sampler = PYSamplerK::new(theta, alpha, n, k).unwrap();
            let label = format!("pitman-yor alpha={alpha} n={n} k={k} mle theta={theta}");
            assert_fits(&label, &sampler, &py_law(n, k, alpha), 1);
        }
    }
    assert!(theta_mle_py(9, 2, 0.9, None) < 0.0);
    assert_eq!(theta_mle_py(10, 4, -10.0, None), 40.0);
}

// *agreement at realistic sizes

// mean and variance of the homozygosity and of the largest allele count
fn summary(sampler: &dyn Sampler<Pcg64>, samples: usize, seed: u64) -> [(f64, f64); 2] {
    let mut rng = Pcg64::seed_from_u64(seed);
    let draws: Vec<[f64; 2]> = (0..samples)
        .map(|_| {
            let c = sampler.sample(&mut rng);
            [
                watterson_statistic(&c) as f64,
                *c.iter().max().unwrap() as f64,
            ]
        })
        .collect();
    [0, 1].map(|s| {
        let mean = draws.iter().map(|d| d[s]).sum::<f64>() / samples as f64;
        let var = draws.iter().map(|d| (d[s] - mean).powi(2)).sum::<f64>() / (samples - 1) as f64;
        (mean, var)
    })
}

// both mean summaries agree to within 5 standard errors
fn assert_same_summary(
    name: &str,
    reference: [(f64, f64); 2],
    other: [(f64, f64); 2],
    samples: usize,
) {
    for (stat, (m1, v1), (m2, v2)) in [
        ("homozygosity", reference[0], other[0]),
        ("largest count", reference[1], other[1]),
    ] {
        let z = (m2 - m1) / ((v1 + v2) / samples as f64).sqrt();
        assert!(
            z.abs() < 5.0,
            "{name}: mean {stat} {m2:.2} vs reference {m1:.2} (z = {z:.1})"
        );
    }
}

// we can't enumerate configurations at the sizes in the benchmarks, so check that all
// samplers agree with feller there, also for extreme values of theta
#[test]
fn samplers_agree_at_large_n() {
    let (n, k, samples) = (265, 81, 20_000);
    let theta = theta_mle(n, k, None);
    let reference = summary(&FellerSamplerK::new(theta, n, k).unwrap(), samples, 1);

    let others: [(&str, Box<dyn Sampler<Pcg64>>); 6] = [
        ("crp", Box::new(CRPSamplerK::new(theta, n, k).unwrap())),
        (
            "conditional crp",
            Box::new(ConditionalCRPSampler::new(theta, n, k, None).unwrap()),
        ),
        (
            "biased crp, bias 1",
            Box::new(BiasedCRPSampler::new(theta, n, k, 1.0, None).unwrap()),
        ),
        (
            "feller, theta 0.01",
            Box::new(FellerSamplerK::new(0.01, n, k).unwrap()),
        ),
        (
            "crp, theta 1000",
            Box::new(CRPSamplerK::new(1000.0, n, k).unwrap()),
        ),
        (
            "pitman-yor, alpha 0",
            Box::new(PYSamplerK::new(theta, 0.0, n, k).unwrap()),
        ),
    ];
    for (seed, (name, sampler)) in (2..).zip(others) {
        let other = summary(&*sampler, samples, seed);
        assert_same_summary(name, reference, other, samples);
    }
}

// for alpha > 0 there is no other conditional sampler to compare to at these sizes
// (but see the rejection check below), and the law given k still must not depend on theta
#[test]
fn pitman_yor_free_of_theta_at_large_n() {
    let (n, k, samples) = (265, 81, 20_000);
    // a reference theta and two others for each alpha, the smallest valid ones first
    for (alpha, thetas) in [(0.5, [1.0, -0.4, 50.0]), (-1.0, [82.0, 200.0, 1000.0])] {
        let reference = summary(
            &PYSamplerK::new(thetas[0], alpha, n, k).unwrap(),
            samples,
            1,
        );
        for (seed, &theta) in (2..).zip(&thetas[1..]) {
            let other = summary(&PYSamplerK::new(theta, alpha, n, k).unwrap(), samples, seed);
            let name = format!("pitman-yor alpha={alpha}, theta {theta} vs {}", thetas[0]);
            assert_same_summary(&name, reference, other, samples);
        }
    }
}

// mean and variance of the number of alleles after n customers. the next customer opens a table
// w.p. (theta + alpha k_j) / (theta + j), which is linear in k_j, so the moments follow
//   E k_(j+1) = E k_j + (theta + alpha E k_j) / (theta + j)
//   E k_(j+1)^2 = E k_j^2 + E (2 k_j + 1)(theta + alpha k_j) / (theta + j)
fn k_moments(n: usize, theta: f64, alpha: f64) -> (f64, f64) {
    // one customer at one table
    let (mut m1, mut m2) = (1.0, 1.0);
    for j in 1..n {
        let d = theta + j as f64;
        (m1, m2) = (
            m1 + (theta + alpha * m1) / d,
            m2 + (theta + (2.0 * theta + alpha) * m1 + 2.0 * alpha * m2) / d,
        );
    }
    (m1, m2 - m1 * m1)
}

// draws from a free-k sampler until one has k alleles
struct GivenK<S>(S, usize);

impl<S: Sampler<Pcg64>> Sampler<Pcg64> for GivenK<S> {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        loop {
            let c = self.0.sample(rng);
            if c.len() == self.1 {
                break c;
            }
        }
    }

    fn theta(&self) -> f64 {
        self.0.theta()
    }
}

// at these sizes the free sampler should still get the number of alleles right on average,
// and given k it should be PYSamplerK, which we can check by keeping its draws with k alleles.
// both have k around n / 5, as with mostly singletons the summaries can't tell laws apart
#[test]
fn pitman_yor_given_k_agrees_at_large_n() {
    // smaller than above, as with k free only 1 in 25 or 1 in 5 draws has the k we keep
    let (n, samples) = (120, 5_000);
    for (theta, alpha) in [(1.0, 0.5), (30.0, -1.0)] {
        let label = format!("pitman-yor alpha={alpha} theta={theta}");
        let sampler = PYSampler::new(theta, alpha, n).unwrap();

        let (mean, var) = k_moments(n, theta, alpha);
        let mut rng = Pcg64::seed_from_u64(1);
        let k_bar = (0..samples)
            .map(|_| sampler.sample(&mut rng).len() as f64)
            .sum::<f64>()
            / samples as f64;
        let z = (k_bar - mean) / (var / samples as f64).sqrt();
        assert!(
            z.abs() < 5.0,
            "{label}: mean k {k_bar:.2} vs {mean:.2} (z = {z:.1})"
        );

        // the expected k is about the most likely one, so rejection doesn't take too long
        let k = mean.round() as usize;
        let reference = summary(&PYSamplerK::new(theta, alpha, n, k).unwrap(), samples, 2);
        let other = summary(&GivenK(sampler, k), samples, 3);
        assert_same_summary(&format!("{label} given k={k}"), reference, other, samples);
    }
}

// *invariants over a wider range of parameters

fn assert_valid(label: &str, c: &[u16], n: usize, k: usize) {
    assert!(
        c.len() == k && !c.contains(&0) && c.iter().map(|&x| x as usize).sum::<usize>() == n,
        "{label}: {c:?} is not {k} positive counts summing to {n}"
    );
}

#[test]
fn samples_have_k_alleles_summing_to_n() {
    let mut rng = Pcg64::seed_from_u64(1);
    for n in [1, 2, 3, 7, 30, 120] {
        let mut ks = vec![1, 2, n / 2, n - 1, n];
        ks.retain(|&k| 1 <= k && k <= n);
        ks.sort_unstable();
        ks.dedup();
        for k in ks {
            for theta in [0.05, 1.0, 30.0] {
                let samplers: [(&str, Box<dyn Sampler<Pcg64>>); 9] = [
                    (
                        "feller",
                        Box::new(FellerSamplerK::new(theta, n, k).unwrap()),
                    ),
                    (
                        "pitman-yor 0.5",
                        Box::new(PYSamplerK::new(theta, 0.5, n, k).unwrap()),
                    ),
                    // these ignore the theta of the loop, which isn't valid for them
                    (
                        "pitman-yor -1",
                        Box::new(PYSamplerK::new(default_theta(k, -1.0), -1.0, n, k).unwrap()),
                    ),
                    (
                        "pitman-yor -inf",
                        Box::new(PYSamplerK::new(f64::INFINITY, f64::NEG_INFINITY, n, k).unwrap()),
                    ),
                    (
                        "pitman-yor 1",
                        Box::new(PYSamplerK::new(1.0, 1.0, n, k).unwrap()),
                    ),
                    ("crp", Box::new(CRPSamplerK::new(theta, n, k).unwrap())),
                    (
                        "conditional crp",
                        Box::new(ConditionalCRPSampler::new(theta, n, k, None).unwrap()),
                    ),
                    (
                        "biased crp 0.7",
                        Box::new(BiasedCRPSampler::new(theta, n, k, 0.7, None).unwrap()),
                    ),
                    (
                        "biased crp 1.4",
                        Box::new(BiasedCRPSampler::new(theta, n, k, 1.4, None).unwrap()),
                    ),
                ];
                for (name, sampler) in samplers {
                    for _ in 0..200 {
                        let label = format!("{name} n={n} k={k} theta={theta}");
                        assert_valid(&label, &sampler.sample(&mut rng), n, k);
                    }
                }
            }
        }
    }
}

#[test]
fn unconditional_samples_sum_to_n() {
    let mut rng = Pcg64::seed_from_u64(1);
    for n in [1, 2, 7, 120] {
        for theta in [0.05, 1.0, 30.0] {
            // with the most alleles each can give
            let samplers: [(&str, Box<dyn Sampler<Pcg64>>, usize); 5] = [
                ("feller", Box::new(FellerSampler::new(theta, n).unwrap()), n),
                (
                    "pitman-yor 0.5",
                    Box::new(PYSampler::new(theta, 0.5, n).unwrap()),
                    n,
                ),
                (
                    "pitman-yor 1",
                    Box::new(PYSampler::new(theta, 1.0, n).unwrap()),
                    n,
                ),
                // these ignore the theta of the loop, as for alpha < 0 it has to be m |alpha|
                // for m types, and then there are at most m alleles
                (
                    "pitman-yor -0.5, 3 types",
                    Box::new(PYSampler::new(1.5, -0.5, n).unwrap()),
                    3,
                ),
                (
                    "pitman-yor -2, 50 types",
                    Box::new(PYSampler::new(100.0, -2.0, n).unwrap()),
                    50,
                ),
            ];
            for (name, sampler, max_k) in samplers {
                for _ in 0..200 {
                    let c = sampler.sample(&mut rng);
                    let label = format!("{name} n={n} theta={theta}");
                    assert_valid(&label, &c, n, c.len());
                    assert!(c.len() <= max_k, "{label}: {c:?} has over {max_k} alleles");
                }
            }
        }
    }
}

#[test]
fn samples_extend_the_initial_configuration() {
    let mut rng = Pcg64::seed_from_u64(1);
    let n = 12;
    for init in [&[1][..], &[5], &[1, 1], &[2, 3], &[1, 1, 1, 1], &[11]] {
        let (n0, k0) = (init.iter().sum::<u16>() as usize, init.len());
        for k in k0..=(k0 + n - n0) {
            for theta in [0.05, 1.0, 30.0] {
                let ic = conf(init);
                let samplers: [(&str, Box<dyn Sampler<Pcg64>>); 2] = [
                    (
                        "conditional crp",
                        Box::new(ConditionalCRPSampler::new(theta, n, k, Some(&ic)).unwrap()),
                    ),
                    (
                        "biased crp 1.4",
                        Box::new(BiasedCRPSampler::new(theta, n, k, 1.4, Some(&ic)).unwrap()),
                    ),
                ];
                for (name, sampler) in samplers {
                    for _ in 0..200 {
                        let c = sampler.sample(&mut rng);
                        let label = format!("{name} n={n} k={k} init={init:?} theta={theta}");
                        assert_valid(&label, &c, n, k);
                        // the initial alleles come first and can only grow
                        assert!(c.iter().zip(init).all(|(a, b)| a >= b), "{label}: {c:?}");
                    }
                }
            }
        }
    }
}

// *parameter validation

#[test]
fn constructors_reject_invalid_parameters() {
    assert!(FellerSamplerK::new(1.0, 5, 0).is_err());
    assert!(FellerSamplerK::new(1.0, 5, 6).is_err());
    assert!(ConditionalCRPSampler::new(1.0, 5, 0, None).is_err());
    assert!(ConditionalCRPSampler::new(1.0, 5, 6, None).is_err());
    assert!(CRPSamplerK::new(1.0, 5, 0).is_err());
    assert!(CRPSamplerK::new(1.0, 5, 6).is_err());
    for bias in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            BiasedCRPSampler::new(1.0, 5, 2, bias, None).is_err(),
            "bias {bias}"
        );
    }
    // theta = 0 is fine without k, as everything is one allele
    for (theta, n) in [(-1.0, 5), (f64::INFINITY, 5), (f64::NAN, 5), (1.0, 0)] {
        assert!(FellerSampler::new(theta, n).is_err(), "theta={theta} n={n}");
    }
    assert!(FellerSampler::new(0.0, 5).is_ok());
}

#[test]
fn pitman_yor_k_rejects_invalid_parameters() {
    for (theta, alpha, n, k) in [
        (1.0, 0.5, 5, 0),
        (1.0, 0.5, 5, 6),
        (1.0, 1.5, 5, 2),
        (1.0, f64::NAN, 5, 2),
        // theta has to be > -alpha, and >= k |alpha| for alpha < 0
        (-0.5, 0.5, 5, 2),
        (2.5, -1.0, 5, 3),
        (f64::NAN, 0.5, 5, 2),
        (f64::INFINITY, 0.5, 5, 2),
    ] {
        assert!(
            PYSamplerK::new(theta, alpha, n, k).is_err(),
            "theta={theta} alpha={alpha} n={n} k={k}"
        );
    }
    // but with one allele, or at the limits of alpha, theta doesn't enter at all,
    // like the theta = -alpha that theta_mle gives for k = 1
    for (theta, alpha, k) in [(-0.5, 0.5, 1), (-1.0, 1.0, 3), (1.0, f64::NEG_INFINITY, 3)] {
        assert!(
            PYSamplerK::new(theta, alpha, 5, k).is_ok(),
            "theta={theta} alpha={alpha} k={k}"
        );
    }
}

#[test]
fn pitman_yor_rejects_invalid_parameters() {
    for (theta, alpha, n) in [
        // for alpha < 0, theta / |alpha| has to be a whole number of types, at least 1
        (2.5, -1.0, 10),
        (0.0, -1.0, 10),
        // which alpha = -inf can't have
        (f64::INFINITY, f64::NEG_INFINITY, 10),
        // otherwise theta > -alpha, and alpha <= 1
        (-0.5, 0.5, 10),
        (1.0, 1.5, 10),
        (f64::INFINITY, 0.5, 10),
        (f64::NAN, 0.5, 10),
        (1.0, f64::NAN, 10),
        (1.0, 0.5, 0),
    ] {
        assert!(
            PYSampler::new(theta, alpha, n).is_err(),
            "theta={theta} alpha={alpha} n={n} was accepted"
        );
    }
}

fn assert_rejects_initial_configuration(n: usize, k: usize, init: &[u16]) {
    let ic = conf(init);
    let label = format!("n={n} k={k} init={init:?}");
    assert!(
        ConditionalCRPSampler::new(1.0, n, k, Some(&ic)).is_err(),
        "{label}"
    );
    assert!(
        BiasedCRPSampler::new(1.0, n, k, 1.5, Some(&ic)).is_err(),
        "{label}"
    );
}

#[test]
fn constructors_reject_unreachable_initial_configurations() {
    // needs more new alleles than there are samples left
    assert_rejects_initial_configuration(10, 9, &[1, 3]);
    // leaves no samples to add
    assert_rejects_initial_configuration(7, 3, &[3, 4]);
    assert_rejects_initial_configuration(7, 2, &[3, 4]);
}

#[test]
fn constructors_reject_initial_configurations_with_more_than_k_alleles() {
    assert_rejects_initial_configuration(10, 2, &[1, 1, 1]);
}

#[test]
fn constructors_reject_initial_configurations_with_more_than_n_samples() {
    assert_rejects_initial_configuration(5, 3, &[3, 4]);
}
