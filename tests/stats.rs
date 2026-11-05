//! tests for the test statistics, against the exact ewens sampling formula

mod common;

use std::cmp::Ordering;

use common::{factorial, multiplicity_factorials, py_law};
use ewens_sampler::Configuration;
use ewens_sampler::stats::Test;
use ewens_sampler::stats::ewens::{
    SlatkinTest, WattersonTest, slatkin_statistic, watterson_statistic,
};

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
// k alleles) consistent with each sorted configuration c: the ewens sampling formula
// P(c|n,k) split over the k!/prod_j a_j! such assignments, so it goes like 1/prod_b c_b
fn unordered_law(n: usize, k: usize) -> Vec<(Vec<u16>, f64)> {
    py_law(n, k, 0.0)
        .into_iter()
        .map(|(c, p)| {
            let assignments = factorial(k) / multiplicity_factorials(&c);
            (c, p / assignments)
        })
        .collect()
}

// equal up to rounding (distinct values differ by far more at these n)
fn is_tie(p: f64, q: f64) -> bool {
    (p - q).abs() <= 1e-9 * p.max(q)
}

// slatkin's exact test as corrected in slatkin (1996): c is in the lower tail of c0 when an
// unordered configuration consistent with c is no more probable than one consistent with c0,
// and in the upper tail when it is no less probable.
// NOTE: ranking by P(c|n,k) itself is the original 1994 test, which the correction retracts
#[test]
fn slatkin_compares_unordered_configuration_probabilities() {
    for n in 2..=14 {
        for k in 1..=n {
            let law = unordered_law(n, k);
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
            let law = unordered_law(n, k);
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
