//! distribution and invariant tests for the samplers

mod common;

use common::{
    Law, assert_fits, biased_crp_law, esf_law, factorial, given_k, multiplicity_factorials,
    partitions, py_crp_law, py_law,
};
use ewens_sampler::Configuration;
use ewens_sampler::samplers::Sampler;
use ewens_sampler::samplers::ewens::theta::theta_mle;
use ewens_sampler::samplers::ewens::{
    BiasedCRPSampler, CRPSamplerK, ConditionalCRPSampler, FellerSampler, FellerSamplerK,
};
use ewens_sampler::samplers::pitman_yor::{PYSamplerK, default_theta};
use ewens_sampler::stats::watterson_statistic;
use rand::SeedableRng;
use rand_pcg::Pcg64;

// conditional on k the results should not depend on theta, so every sampler is run with both values
const THETAS: [f64; 2] = [0.3, 8.0];

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
        &CRPSamplerK::new(1.0, 10, 4),
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
    fits_esf_for_any_theta("crp", |t, n, k| Box::new(CRPSamplerK::new(t, n, k)));
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
        assert_fits(&label, &FellerSampler::new(theta, 7), &esf_law(7, theta), 1);
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
                let sampler = PYSamplerK::new(theta, alpha, n, k);
                let label = format!("pitman-yor alpha={alpha} n={n} k={k} theta={theta}");
                assert_fits(&label, &sampler, &law, 1);
            }
        }
    }
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
        ("crp", Box::new(CRPSamplerK::new(theta, n, k))),
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
        ("crp, theta 1000", Box::new(CRPSamplerK::new(1000.0, n, k))),
        (
            "pitman-yor, alpha 0",
            Box::new(PYSamplerK::new(theta, 0.0, n, k)),
        ),
    ];
    for (seed, (name, sampler)) in (2..).zip(others) {
        let other = summary(&*sampler, samples, seed);
        assert_same_summary(name, reference, other, samples);
    }
}

// for alpha > 0 there is no other sampler to compare to at these sizes,
// but the law given k still must not depend on theta
#[test]
fn pitman_yor_free_of_theta_at_large_n() {
    let (n, k, samples) = (265, 81, 20_000);
    // a reference theta and two others for each alpha, the smallest valid ones first
    for (alpha, thetas) in [(0.5, [1.0, -0.4, 50.0]), (-1.0, [82.0, 200.0, 1000.0])] {
        let reference = summary(&PYSamplerK::new(thetas[0], alpha, n, k), samples, 1);
        for (seed, &theta) in (2..).zip(&thetas[1..]) {
            let other = summary(&PYSamplerK::new(theta, alpha, n, k), samples, seed);
            let name = format!("pitman-yor alpha={alpha}, theta {theta} vs {}", thetas[0]);
            assert_same_summary(&name, reference, other, samples);
        }
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
                        Box::new(PYSamplerK::new(theta, 0.5, n, k)),
                    ),
                    // these ignore the theta of the loop, which isn't valid for them
                    (
                        "pitman-yor -1",
                        Box::new(PYSamplerK::new(default_theta(k, -1.0), -1.0, n, k)),
                    ),
                    (
                        "pitman-yor -inf",
                        Box::new(PYSamplerK::new(f64::INFINITY, f64::NEG_INFINITY, n, k)),
                    ),
                    ("pitman-yor 1", Box::new(PYSamplerK::new(1.0, 1.0, n, k))),
                    ("crp", Box::new(CRPSamplerK::new(theta, n, k))),
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
            let sampler = FellerSampler::new(theta, n);
            for _ in 0..200 {
                let c = sampler.sample(&mut rng);
                assert_valid(&format!("feller n={n} theta={theta}"), &c, n, c.len());
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
    for bias in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            BiasedCRPSampler::new(1.0, 5, 2, bias, None).is_err(),
            "bias {bias}"
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
