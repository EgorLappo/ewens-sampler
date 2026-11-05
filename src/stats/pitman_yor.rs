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
// for the second we wiil use the derivative of log P:
// d/d alpha log P = sum_{i=1}^k sum_{m=1}^{r_i - 1} 1/{alpha - m}
// where we can also cache the sums for all m

pub struct ExactPTest(f64);

impl ExactPTest {
    pub fn new(uconf: &Configuration) -> Self {
        let target = exact_p_statistic(&uconf.0);
        Self(target)
    }
}

pub struct ScoreTest(f64);

impl ScoreTest {
    pub fn new(uconf: &Configuration) -> Self {
        let target = score_statistic(&uconf.0);
        Self(target)
    }
}

#[inline]
pub fn exact_p_statistic(uconf: &[u16]) -> f64 {
    todo!()
}

#[inline]
pub fn score_statistic(uconf: &[u16]) -> f64 {
    todo!()
}
