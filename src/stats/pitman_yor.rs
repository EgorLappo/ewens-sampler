use std::cmp::Ordering;

use color_eyre::eyre::{Result, bail};

use super::Test;
use crate::Configuration;

// here are the two tests for the Pitman-Yor process: the exact probability test and a Rao score test analogue
//
//
// first one is based on the realization probability
// P ~ prod_{j=1}^n ( (1-alpha)_{j-1} / j! )^{c_j}
// where n is the number of samples (customers),
// c_j is the number of allelic types (tables) with count j
//
// reparameterize it with (r_1, ..., r_k) where r_i is the count for
// allele (table) i
//
// then
// log P ~ sum_{i=1}^k [ ( sum_{m=1}^{r_i - 1} log((m - alpha)/m) ) - log(r_i)]
// as a check, alpha=0 returns the slatkin test statistic
// (we can cache the terms for (sums of) logs in a lookup table)
//
//
// for the second we will use the derivative of log P:
// d/d alpha log P = sum_{i=1}^k sum_{m=1}^{r_i - 1} 1/{alpha - m}
// where we can also cache the sums for all m
//
//
// numerically, the two sums in log P cancel at alpha = -1, where every allele has weight 1,
// and come out as rounding noise instead of 0, which breaks the ties there.
// so instead we add up the ratios of the weights of consecutive counts,
// (r - 1 - alpha) / r = 1 - (1 + alpha) / r, and keep -log P like slatkin_statistic:
// -log P ~ sum_{i=1}^k sum_{m=2}^{r_i} -log(1 - (1 + alpha) / m)
// which is exactly 0 at alpha = -1, and the sum of log r_i at alpha = 0.

// at alpha = 1 the only configuration is (n - k + 1, 1, ..., 1), as every count above 1 has
// weight (1 - alpha)_(r - 1) = 0, and at alpha = -inf the terms are infinite, so we stop short of both
fn check_alpha(alpha: f64) -> Result<()> {
    if !(alpha > f64::NEG_INFINITY && alpha < 1.0) {
        bail!("invalid parameters alpha={alpha}. requires -inf < alpha < 1");
    }
    Ok(())
}

// table[r] = sum_{m=2}^r term(m) for every count r <= n, so table[1] = 0 (table[0] is never used).
// the running sum is compensated (neumaier), which keeps every entry within a couple of EPSILON
fn running_sums(n: usize, term: impl Fn(usize) -> f64) -> Box<[f64]> {
    let mut table = vec![0.0; n + 1];
    // the sum together with the low bits lost in additions
    let (mut sum, mut lost) = (0.0f64, 0.0f64);
    for m in 2..=n {
        let x = term(m);
        let t = sum + x;
        lost += if sum.abs() >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
        table[m] = sum + lost;
    }
    table.into_boxed_slice()
}

#[inline]
fn table_sum(table: &[f64], uconf: &[u16]) -> f64 {
    uconf.iter().map(|&r| table[r as usize]).sum()
}

// rounded comparison with ties within tolerance, like in ewens code
fn compare_rounded(x: f64, target: f64, k: usize) -> Ordering {
    let tol = 4.0 * k as f64 * f64::EPSILON * target.abs();
    if x < target - tol {
        Ordering::Less
    } else if x > target + tol {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub struct ExactPTest {
    // -log w(r) for every count possible r, where w(r) = (1 - alpha)_(r - 1) / r! is the weight of an allele
    table: Box<[f64]>,
    target: f64,
}

impl ExactPTest {
    pub fn new(uconf: &Configuration, alpha: f64) -> Result<Self> {
        check_alpha(alpha)?;
        let table = running_sums(uconf.n(), |m| -(-(1.0 + alpha) / m as f64).ln_1p());
        let target = table_sum(&table, &uconf.0);
        Ok(Self { table, target })
    }
}

impl Test for ExactPTest {
    type Value = f64;

    fn name(&self) -> &'static str {
        "exact"
    }

    #[inline]
    fn statistic(&self, uconf: &[u16]) -> Self::Value {
        table_sum(&self.table, uconf)
    }

    fn compare(&self, tconf: &[u16]) -> Ordering {
        compare_rounded(self.statistic(tconf), self.target, tconf.len()).reverse()
    }
}

pub struct ScoreTest {
    // sum_{m=1}^{r-1} 1 / (alpha - m) for every count r
    table: Box<[f64]>,
    target: f64,
}

impl ScoreTest {
    pub fn new(uconf: &Configuration, alpha: f64) -> Result<Self> {
        check_alpha(alpha)?;
        // going from count m - 1 to m adds the term for m - 1
        let table = running_sums(uconf.n(), |m| 1.0 / (alpha - (m - 1) as f64));
        let target = table_sum(&table, &uconf.0);
        Ok(Self { table, target })
    }
}

impl Test for ScoreTest {
    type Value = f64;

    fn name(&self) -> &'static str {
        "score"
    }

    #[inline]
    fn statistic(&self, uconf: &[u16]) -> Self::Value {
        table_sum(&self.table, uconf)
    }

    fn compare(&self, tconf: &[u16]) -> Ordering {
        compare_rounded(self.statistic(tconf), self.target, tconf.len())
    }
}
