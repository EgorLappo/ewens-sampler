use crate::sampler::{Sampler, conditional_bernoulli_q, conditional_bernoulli_sample};
use rand::Rng;

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
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16> {
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

        configuration.sort_unstable();

        configuration
    }
}

#[derive(Debug, Clone)]
pub struct FellerSamplerK {
    pub _theta: f64,
    pub _n: usize,
    pub k: usize,
    p: Box<[f64]>,
    q: Box<[f64]>,
}

impl FellerSamplerK {
    pub fn new(theta: f64, n: usize, k: usize) -> Self {
        let p = feller_bernoulli_p(n, theta);
        let q = conditional_bernoulli_q(k - 1, &p)
            .into_iter()
            .flatten()
            .collect();
        Self {
            _theta: theta,
            _n: n,
            k,
            p,
            q,
        }
    }
}

impl Sampler for FellerSamplerK {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16> {
        let zetas = conditional_bernoulli_sample(self.k - 1, &self.p, &self.q, rng);

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

        configuration.sort_unstable();

        configuration
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
