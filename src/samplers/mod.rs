use rand::{Rng, RngExt};

pub mod ewens;
pub mod pitman_yor;

// trait for samplers
pub trait Sampler<R: Rng> {
    fn sample(&self, rng: &mut R) -> Vec<u16>;
    fn theta(&self) -> f64;
}

// *conditional bernoulli sampling, shared by all samplers
// see reference module in ewens for the code that follows the
// algorithm directly without logs
#[inline]
fn ln_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn conditional_bernoulli_sample<R: Rng>(n: usize, s: usize, p: &[f64], rng: &mut R) -> Vec<usize> {
    let mut out = Vec::new();
    conditional_bernoulli_sample_into(n, s, p, rng, &mut out);
    out
}

fn conditional_bernoulli_sample_into<R: Rng>(
    n: usize,
    s: usize,
    p: &[f64],
    rng: &mut R,
    out: &mut Vec<usize>,
) {
    // set all to zeros
    out.clear();
    out.resize(n, 0);

    if s == 0 {
        return;
    }
    let mut rem = s;
    for j in 0..n {
        if rem == 0 {
            // short-circuit 1
            break;
        }
        if rem == n - j {
            // short-circuit 2
            out[j..n].fill(1);
            rem = 0;
            break;
        }
        if rng.random::<f64>() < p[(rem - 1) * n + j] {
            out[j] = 1;
            rem -= 1;
        }
    }
    debug_assert_eq!(rem, 0);
}
