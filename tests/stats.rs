//! tests for the test statistics, against the exact ewens and pitman-yor laws

mod common;

use std::cmp::Ordering;

use common::{factorial, multiplicity_factorials, partitions, py_law};
use ewens_sampler::Configuration;
use ewens_sampler::stats::Test;
use ewens_sampler::stats::ewens::{
    SlatkinTest, WattersonTest, slatkin_statistic, watterson_statistic,
};
use ewens_sampler::stats::pitman_yor::{ExactPTest, ScoreTest};

fn conf(counts: &[u16]) -> Configuration {
    let s: Vec<String> = counts.iter().map(u16::to_string).collect();
    s.join(" ").parse().unwrap()
}

#[test]
fn statistics_on_small_configurations() {
    assert_eq!(watterson_statistic(&[1, 2, 3]), 14);
    assert_eq!(watterson_statistic(&[5]), 25);
    assert!((slatkin_statistic(&[2, 3, 4]) - 24f64.ln()).abs() < 1e-12);
}

// ties go into both tails, so the observed configuration always counts towards its own p-values
#[test]
fn tails_contain_the_observed_configuration() {
    for c in [&[1, 1, 2, 8][..], &[4], &[1, 1, 1], &[2, 3, 3, 5]] {
        assert_eq!(
            SlatkinTest::new(&conf(c)).compare(c),
            Ordering::Equal,
            "slatkin {c:?}"
        );
        assert_eq!(
            WattersonTest::new(&conf(c)).compare(c),
            Ordering::Equal,
            "watterson {c:?}"
        );
    }
}

// in both tests the lower tail is the even side and the upper tail the uneven one
#[test]
fn lower_tails_are_the_even_side() {
    let observed = conf(&[2, 3, 3, 4]);
    let (s, w) = (SlatkinTest::new(&observed), WattersonTest::new(&observed));
    for (c, side) in [
        (&[3, 3, 3, 3][..], Ordering::Less),
        (&[1, 1, 1, 9], Ordering::Greater),
    ] {
        assert_eq!(s.compare(c), side, "slatkin {c:?}");
        assert_eq!(w.compare(c), side, "watterson {c:?}");
    }
}

// probability of a single unordered configuration (one assignment of the counts to the
// k alleles) consistent with each sorted configuration c: P(c|n,k) split over the
// k!/prod_j a_j! such assignments, so under ewens it goes like 1/prod_b c_b,
// and under pitman-yor like prod_b (1-alpha)_(c_b - 1) / c_b!
fn unordered_law(n: usize, k: usize, alpha: f64) -> Vec<(Vec<u16>, f64)> {
    py_law(n, k, alpha)
        .into_iter()
        .map(|(c, p)| {
            let assignments = factorial(k) / multiplicity_factorials(&c);
            (c, p / assignments)
        })
        .collect()
}

// equal up to rounding (distinct values differ by far more at these n)
fn is_tie(p: f64, q: f64) -> bool {
    (p - q).abs() <= 1e-9 * p.abs().max(q.abs())
}

// slatkin's exact test as corrected in slatkin (1996): c is in the lower tail of c0 when an
// unordered configuration consistent with c is no more probable than one consistent with c0,
// and in the upper tail when it is no less probable.
// NOTE: ranking by P(c|n,k) itself is the original 1994 test, which the correction retracts
#[test]
fn slatkin_compares_unordered_configuration_probabilities() {
    for n in 2..=14 {
        for k in 1..=n {
            let law = unordered_law(n, k, 0.0);
            for (c0, p0) in &law {
                let test = SlatkinTest::new(&conf(c0));
                for (c, p) in law.iter().filter(|(_, p)| !is_tie(*p, *p0)) {
                    assert_eq!(
                        test.compare(c),
                        p.total_cmp(p0),
                        "n={n} k={k}: P({c:?}) = {p:.3e} vs observed P({c0:?}) = {p0:.3e}"
                    );
                }
            }
        }
    }
}

// equally probable configurations belong to both of each other's tails, e.g. (2, 2, 10) and
// (1, 5, 8) have prod_b c_b = 40, but their sums of logs differ in the last bit
#[test]
fn slatkin_tails_include_exact_ties() {
    for n in 2..=14 {
        for k in 1..=n {
            let law = unordered_law(n, k, 0.0);
            for (c0, p0) in &law {
                let test = SlatkinTest::new(&conf(c0));
                for (c, _) in law.iter().filter(|(_, p)| is_tie(*p, *p0)) {
                    assert_eq!(
                        test.compare(c),
                        Ordering::Equal,
                        "n={n} k={k}: {c:?} is as probable as the observed {c0:?}"
                    );
                }
            }
        }
    }
}

// *pitman-yor

// the pitman-yor exact test ranks unordered configurations by their probability under
// pitman-yor, like slatkin's does under ewens, with ties in both tails.
// at alpha = -1 every configuration ties, and at alpha = -2 those with the same prod_b (c_b + 1)
#[test]
fn exact_test_compares_unordered_configuration_probabilities() {
    for alpha in [0.9, 0.5, 0.0, -0.5, -1.0, -2.0, -3.7] {
        for n in 2..=12 {
            for k in 1..=n {
                let law = unordered_law(n, k, alpha);
                for (c0, p0) in &law {
                    let test = ExactPTest::new(&conf(c0), alpha).unwrap();
                    for (c, p) in &law {
                        let want = if is_tie(*p, *p0) {
                            Ordering::Equal
                        } else {
                            p.total_cmp(p0)
                        };
                        assert_eq!(
                            test.compare(c),
                            want,
                            "alpha={alpha} n={n} k={k}: P({c:?}) = {p:.3e} vs observed P({c0:?}) = {p0:.3e}"
                        );
                    }
                }
            }
        }
    }
}

// at alpha = 0 it is slatkin's test
#[test]
fn exact_test_at_zero_is_slatkins() {
    for n in 2..=14 {
        for k in 1..=n {
            let configurations = partitions(n, k);
            for c0 in &configurations {
                let slatkin = SlatkinTest::new(&conf(c0));
                let exact = ExactPTest::new(&conf(c0), 0.0).unwrap();
                for c in &configurations {
                    assert_eq!(
                        exact.compare(c),
                        slatkin.compare(c),
                        "n={n} k={k}: {c:?} vs observed {c0:?}"
                    );
                }
            }
        }
    }
}

// exact ties at large n, where a plain running sum would drift past the tolerance:
// at alpha = 0, (2s, 2s, 10s) and (s, 5s, 8s) have the same prod_b c_b,
// and at alpha = -2 the same counts minus one have the same prod_b (c_b + 1)
#[test]
fn exact_test_keeps_ties_at_large_n() {
    for s in (2..=3000u16).step_by(37) {
        for (alpha, shift) in [(0.0, 0), (-2.0, 1)] {
            let a = [2 * s - shift, 2 * s - shift, 10 * s - shift];
            let b = [s - shift, 5 * s - shift, 8 * s - shift];
            let test = ExactPTest::new(&conf(&a), alpha).unwrap();
            assert_eq!(
                test.compare(&b),
                Ordering::Equal,
                "alpha={alpha}: {a:?} vs {b:?}"
            );
        }
    }
}

// the score's statistic is the derivative in alpha of the log probability, up to a constant
// that is the same for every configuration (the derivative of the normalization), so its
// differences match a numerical derivative of the log ratio of two probabilities
#[test]
fn score_is_the_derivative_of_the_log_probability() {
    let h = 1e-5;
    for alpha in [0.9, 0.5, 0.0, -0.5, -1.0, -2.5] {
        for (n, k) in [(8, 3), (10, 4), (12, 6)] {
            let (below, above) = (py_law(n, k, alpha - h), py_law(n, k, alpha + h));
            let configurations = partitions(n, k);
            let c0 = &configurations[0];
            let test = ScoreTest::new(&conf(c0), alpha).unwrap();
            for c in &configurations {
                let want = ((above[c] / above[c0]).ln() - (below[c] / below[c0]).ln()) / (2.0 * h);
                let got = test.statistic(c) - test.statistic(c0);
                assert!(
                    (got - want).abs() < 1e-6 * (1.0 + want.abs()),
                    "alpha={alpha} n={n} k={k} {c:?}: {got} vs {want}"
                );
            }
        }
    }
}

// and it ranks configurations by that derivative, with ties in both tails
#[test]
fn score_test_compares_derivatives() {
    // the derivative of the log probability, term by term
    let derivative = |c: &[u16], alpha: f64| -> f64 {
        c.iter()
            .map(|&r| (1..r).map(|m| 1.0 / (alpha - m as f64)).sum::<f64>())
            .sum()
    };
    for alpha in [0.9, 0.5, 0.0, -0.5, -1.0, -2.5] {
        for n in 2..=12 {
            for k in 1..=n {
                let configurations = partitions(n, k);
                for c0 in &configurations {
                    let test = ScoreTest::new(&conf(c0), alpha).unwrap();
                    let d0 = derivative(c0, alpha);
                    for c in &configurations {
                        let d = derivative(c, alpha);
                        let want = if is_tie(d, d0) {
                            Ordering::Equal
                        } else {
                            d.total_cmp(&d0)
                        };
                        assert_eq!(
                            test.compare(c),
                            want,
                            "alpha={alpha} n={n} k={k}: D({c:?}) = {d} vs observed D({c0:?}) = {d0}"
                        );
                    }
                }
            }
        }
    }
}

// at alpha = 0 the score is minus a sum of harmonic numbers H_(c_b - 1), which tie exactly for
// (4, 4, 3, 3, 3, 1) and (5, 5, 2, 2, 2, 2): 2 H_3 + 3 H_2 = 2 H_4 + 4 H_1 = 49 / 6
#[test]
fn score_test_keeps_exact_ties() {
    let test = ScoreTest::new(&conf(&[4, 4, 3, 3, 3, 1]), 0.0).unwrap();
    let tied = [5, 5, 2, 2, 2, 2];
    assert!((test.statistic(&tied) + 49.0 / 6.0).abs() < 1e-12);
    assert_eq!(test.compare(&tied), Ordering::Equal);
}

// the score's lower tail is the even side for every alpha, like in the ewens tests. the exact
// test's lower tail is the less probable side, which is the even one for alpha > -1, but the
// less even one for alpha < -1, where an allele's weight grows with its count
#[test]
fn pitman_yor_tails() {
    let observed = conf(&[2, 3, 3, 4]);
    let (even, uneven) = (&[3, 3, 3, 3][..], &[1, 1, 1, 9][..]);
    for alpha in [0.9, 0.5, 0.0, -0.5, -2.0, -10.0] {
        let score = ScoreTest::new(&observed, alpha).unwrap();
        assert_eq!(score.compare(even), Ordering::Less, "score alpha={alpha}");
        assert_eq!(
            score.compare(uneven),
            Ordering::Greater,
            "score alpha={alpha}"
        );
        let exact = ExactPTest::new(&observed, alpha).unwrap();
        let side = if alpha > -1.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        assert_eq!(exact.compare(even), side, "exact alpha={alpha}");
        assert_eq!(exact.compare(uneven), side.reverse(), "exact alpha={alpha}");
    }
    // and at alpha = -1 they all tie
    let exact = ExactPTest::new(&observed, -1.0).unwrap();
    assert_eq!(exact.compare(even), Ordering::Equal);
    assert_eq!(exact.compare(uneven), Ordering::Equal);
}

// there is nothing to test at the limits: alpha = 1 has a single configuration,
// and at alpha = -inf the statistics are infinite
#[test]
fn pitman_yor_tests_reject_the_limits() {
    let c = conf(&[1, 2, 3]);
    for alpha in [1.0, 1.5, f64::NEG_INFINITY, f64::NAN] {
        assert!(ExactPTest::new(&c, alpha).is_err(), "exact alpha={alpha}");
        assert!(ScoreTest::new(&c, alpha).is_err(), "score alpha={alpha}");
    }
}
