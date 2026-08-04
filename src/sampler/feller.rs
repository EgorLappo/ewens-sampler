use crate::sampler::{conditional_bernoulli_probs, conditional_bernoulli_sample, Sampler};
use color_eyre::{eyre::bail, Result};
use rand::RngExt;
use rand_pcg::Pcg64;

#[derive(Debug, Clone)]
pub struct FellerSampler {
    pub theta: f64,
    pub n: usize,
    p: Box<[f64]>,
}

impl FellerSampler {
    pub fn new(theta: f64, n: usize) -> Self {
        let p = feller_bernoulli_p(n, theta);
        Self { theta, n, p }
    }
}

impl Sampler for FellerSampler {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        // in feller coupling, the configuration is defined
        // by looking at spacings between 1s in `samp`

        let mut configuration = Vec::with_capacity(self.n);
        configuration.push(1);

        let mut i = 0;

        for s in self.p.iter().map(|&p| (rng.random::<f64>() < p) as usize) {
            if s == 0 {
                configuration[i] += 1;
            } else {
                configuration.push(1);
                i += 1;
            }
        }

        configuration
    }

    fn theta(&self) -> f64 {
        self.theta
    }
}

#[derive(Debug, Clone)]
pub struct FellerSamplerK {
    pub theta: f64,
    pub _n: usize,
    pub k: usize,
    n_vars: usize,
    probs: Box<[f64]>,
}

impl FellerSamplerK {
    pub fn new(theta: f64, n: usize, k: usize) -> Result<Self> {
        if k < 1 || k > n {
            bail!("invalid parameters n={n}, k={k}. requires 1 <= k <= n");
        }

        // original unconstrained probabitities
        let p = feller_bernoulli_p(n, theta);
        // conditioned ones
        let probs = conditional_bernoulli_probs(k - 1, &p);
        Ok(Self {
            theta,
            _n: n,
            k,
            n_vars: p.len(),
            probs,
        })
    }
}

impl Sampler for FellerSamplerK {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        let zetas = conditional_bernoulli_sample(self.n_vars, self.k - 1, &self.probs, rng);

        // in feller coupling, the configuration is defined
        // by looking at spacings between 1s in `samp`

        let mut configuration = Vec::with_capacity(self.k);
        configuration.push(1);

        let mut i = 0;

        for s in zetas.iter() {
            if *s == 0 {
                configuration[i] += 1;
            } else {
                configuration.push(1);
                i += 1;
            }
        }

        configuration
    }

    fn theta(&self) -> f64 {
        self.theta
    }
}

fn feller_bernoulli_p(n: usize, theta: f64) -> Box<[f64]> {
    // compute probabilities for feller coupling
    // p_j = theta / (theta + j - 1),
    //   where j = n, ..., 1 in this reverse order
    // NOTE: we do not need last variable as p_1 = 1
    (2..=n)
        .rev()
        .map(|j| theta / (theta + (j as f64) - 1.))
        .collect()
}
