use std::cmp::Ordering;

use super::Test;
use crate::Configuration;

pub struct SlatkinTest(f64);

impl SlatkinTest {
    pub fn new(uconf: &Configuration) -> Self {
        let target = slatkin_statistic(&uconf.0);
        Self(target)
    }
}

impl Test for SlatkinTest {
    type Value = f64;

    fn name(&self) -> &'static str {
        "slatkin"
    }

    #[inline]
    fn statistic(uconf: &[u16]) -> Self::Value {
        slatkin_statistic(uconf)
    }

    fn compare(&self, tconf: &[u16]) -> Ordering {
        let lp_c = slatkin_statistic(tconf);
        // now we want to compare P(c|k) with P(c_0|k), with P(c|k) <= P(c_0|k) as the lower tail
        //   in -log-p that is -log P(c|k) >= -log P(c_0|k), so the comparison flips.
        // the less probable configurations are the more even ones,
        // so the lower tail is the even side.
        // equally probable configurations like (2, 2, 10) and (1, 5, 8) can still differ in
        // the last bits of their sums of logs, and those ties belong in both tails.
        // two sums of k nonnegative logs differ by at most ~(k + 1) * EPSILON relative
        // from rounding, so anything within a few times that of the target is a tie
        let tol = 4.0 * tconf.len() as f64 * f64::EPSILON * self.0;
        if lp_c > self.0 + tol {
            Ordering::Less
        } else if lp_c < self.0 - tol {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

#[inline]
pub fn slatkin_statistic(uconf: &[u16]) -> f64 {
    uconf.iter().map(|x| (*x as f64).ln()).sum()
}

// NOTE: similarly but differently,
// note that there is no "conditional Watterson test"
// we are just testing for homozygosity using the conditionally sampled configurations

pub struct WattersonTest(usize);

impl WattersonTest {
    pub fn new(uconf: &Configuration) -> Self {
        let target = watterson_statistic(&uconf.0);
        Self(target)
    }
}

impl Test for WattersonTest {
    type Value = usize;

    fn name(&self) -> &'static str {
        "watterson"
    }

    #[inline]
    fn statistic(uconf: &[u16]) -> Self::Value {
        watterson_statistic(uconf)
    }

    fn compare(&self, tconf: &[u16]) -> Ordering {
        let f_c = watterson_statistic(tconf);
        // lower homozygosity is more even, so the lower tail is the even side here too
        f_c.cmp(&self.0)
    }
}

#[inline]
pub fn watterson_statistic(uconf: &[u16]) -> usize {
    uconf
        .iter()
        .map(|x| {
            let x = *x as usize;
            x * x
        })
        .sum()
}
