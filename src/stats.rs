use crate::Configuration;

// NOTE: for future me and everyone else :3
//   see how we don't have a special separate function for conditional tests
// this is because conditional test adds a constant term to all probabilities
// in our logarithmic calculation, so it ends up cancelling out.
//   this only works because we are comparing log-probabilities with '<' here.
//   this would *also* work if we compared probabilities with 1 by division
//   this would NOT work if we compared actual probabilities with '<'

pub trait Test {
    type Value;
    fn name(&self) -> &'static str;
    fn statistic(uconf: &[u16]) -> Self::Value;
    fn test(&self, tconf: &[u16]) -> bool;
}

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
        lp_c >= self.0
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
