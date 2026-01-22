use rand::Rng;

pub mod crp;
pub mod feller;

pub use crp::{CRPSamplerK, ConditionalCRPSampler};
pub use feller::{FellerSampler, FellerSamplerK};

pub trait Sampler {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16>;
}

// generates a sample of bernoulli variables, each with parameter p_i,
// conditional on their sum being equal to s
fn conditional_bernoulli_q(s: usize, p: &[f64]) -> Vec<Vec<f64>> {
    // we are following arxiv:2012.03103 for the sampling algorithm

    let n = p.len();

    if s > n {
        panic!("conditional_bernoulli: sum > number of variables");
    }

    // q[i,j] = probability that the sum of the variables number j to n is i
    let mut q: Vec<Vec<f64>> = vec![vec![0.0; n]; s + 1];

    // fill in q[0,:]
    //   (these are cases where all bernoulli trials failed, so products of 1-p_i)
    for j in 0..n {
        q[0][j] = p[j..n].iter().map(|x| 1. - x).product();
    }

    // fill in q[1,:]
    q[1][n - 1] = p[n - 1];
    for j in (0..(n - 1)).rev() {
        q[1][j] = p[j] * q[0][j + 1] + (1. - p[j]) * q[1][j + 1];
    }

    for i in 2..(s + 1) {
        // q[i][j] is zero when i > n - j + 1,
        //   equivalently, we only need to consider entries where j <= n - i + 1
        for j in (0..(n - i + 1)).rev() {
            if i <= n - j + 1 {
                q[i][j] = p[j] * q[i - 1][j + 1] + (1. - p[j]) * q[i][j + 1];
            }
        }
    }

    q
}

fn conditional_bernoulli_sample<R: Rng>(s: usize, p: &[f64], q: &[f64], rng: &mut R) -> Vec<usize> {
    // q here is flattened so q[i, j] is q[(s+1)*i + j]

    let n = p.len();
    let mut ans = Vec::with_capacity(n);
    let mut csum = 0;

    let p0 = p[0] * q[n * (s - 1) + 1] / q[n * s];
    let x0 = (rng.random::<f64>() < p0) as usize;
    csum += x0;
    ans.push(x0);

    for (i, pi) in p.iter().enumerate().take(p.len() - 1).skip(1) {
        // if we already got the sum, the rest must be zero
        if csum == s {
            ans.push(0);
            continue;
        }
        //let pi = pi * q[s - csum - 1][i + 1] / q[s - csum][i];
        let pi = pi * q[n * (s - csum - 1) + i + 1] / q[n * (s - csum) + i];
        let xi = (rng.random::<f64>() < pi) as usize;
        csum += xi;
        ans.push(xi);
    }

    ans.push(s - csum);

    ans
}
