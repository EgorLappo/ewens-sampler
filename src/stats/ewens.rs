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

    fn test(&self, tconf: &[u16]) -> bool {
        let lp_c = slatkin_statistic(tconf);
        // now we want to test if P(c|k) <= P(c_0|k)
        //   in -log-p, we test -log P(c|k) >= -log P(c_0|k)
        // equally probable configurations like (2, 2, 10) and (1, 5, 8) can still differ in
        // the last bits of their sums of logs, and those ties belong in the tail.
        // two sums of k nonnegative logs differ by at most ~(k + 1) * EPSILON relative
        // from rounding, so lower the target by a few times that
        let tol = 4.0 * tconf.len() as f64 * f64::EPSILON * self.0;
        lp_c >= self.0 - tol
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

    fn test(&self, tconf: &[u16]) -> bool {
        let f_c = watterson_statistic(tconf);
        f_c <= self.0
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
