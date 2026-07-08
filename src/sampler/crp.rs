use crate::sampler::{conditional_bernoulli_q, conditional_bernoulli_sample, Sampler};
use color_eyre::eyre::{bail, Result, WrapErr};
use rand::{Rng, RngExt};

#[derive(Debug, Clone)]
pub struct CRPSamplerK {
    pub _theta: f64,
    pub n: usize,
    pub k: usize,
    p: Box<[f64]>,
    q: Box<[f64]>,
}

impl CRPSamplerK {
    pub fn new(theta: f64, n: usize, k: usize) -> Self {
        let p = crp_bernoulli_p(n, theta);
        let q = conditional_bernoulli_q(k - 1, &p)
            .into_iter()
            .flatten()
            .collect();
        Self {
            _theta: theta,
            n,
            k,
            p,
            q,
        }
    }
}

impl Sampler for CRPSamplerK {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16> {
        let zetas = conditional_bernoulli_sample(self.k - 1, &self.p, &self.q, rng);

        // in the chinese restaurant process, the configuration is built
        // sample-by sample

        // assignment of samples to cycles
        let mut assignment = vec![0; self.n];
        // first element must go into the first cycle
        assignment[0] = 0;

        // variables to keep track of progress
        let mut cycle = 0;

        let mut configuration = vec![0; self.k];
        configuration[cycle] += 1;

        for (samples_assigned, s) in (1..).zip(zetas.iter()) {
            if *s == 0 {
                // if s == 0, we must place
                // the sample in the same cycle as
                // a randomly chosen previous sample
                let choice = rng.random_range(..samples_assigned);
                let cycle_choice = assignment[choice];
                assignment[samples_assigned] = cycle_choice;

                // increment configuration
                configuration[cycle_choice] += 1;
            } else {
                // if s == 1, we must start a new cycle
                cycle += 1;
                assignment[samples_assigned] = cycle;

                // increment configuration
                configuration[cycle] += 1;
            }
        }

        //let mut configuration = dbg!(configuration);

        configuration.sort_unstable();

        configuration
    }
}

fn crp_bernoulli_p(n: usize, theta: f64) -> Box<[f64]> {
    // compute probabilities for the chinese restaurant process
    // p_j = theta / (theta + j - 1),
    //   where j = 1, ..., n
    // NOTE: we do not need first variable as p_1 = 1
    (2..=n).map(|j| theta / (theta + (j as f64) - 1.)).collect()
}

#[derive(Debug, Clone)]
pub struct ConditionalCRPSampler {
    pub _theta: f64,
    pub n: usize,
    pub k: usize,
    p: Box<[f64]>,
    q: Box<[f64]>,
    k_init: usize,
    n_init: usize,
    conf_init: Vec<u16>,
    assignment_init: Vec<usize>,
}

impl ConditionalCRPSampler {
    pub fn new(theta: f64, n: usize, k: usize, initial_configuration: &str) -> Result<Self> {
        let conf_init = if initial_configuration.is_empty() {
            // empty initial configuration means we sample as usual
            Vec::new()
        } else {
            initial_configuration
                .split(' ')
                .map(|s| s.parse::<u16>())
                .collect::<Result<Vec<_>, _>>()
                .wrap_err_with(|| {
                    format!(
                        "failed while parsing configuration {:?} into unsigned integers",
                        initial_configuration
                    )
                })?
        };

        let k_init = conf_init.len();
        let n_init = conf_init.iter().sum::<u16>() as usize;

        if k_init > k {
            bail!(
                "provided initial configuration '{}' has  {}>k={} entries",
                initial_configuration,
                k_init,
                k
            );
        }

        if n_init >= n {
            bail!(
                "sum of entries of initial configuration '{}' is {}>=n={}",
                initial_configuration,
                n_init,
                n
            )
        }

        // reconstruct initial assignment as well

        let mut assignment_init = vec![0; n];
        let mut samples_assigned = 0;
        for (cycle, &x) in conf_init.iter().enumerate() {
            for _ in 0..x {
                assignment_init[samples_assigned] = cycle;
                samples_assigned += 1;
            }
        }

        // only get p_{n_init + 1} up to p_n
        let p = conditional_crp_bernoulli_p(n_init, n, theta);

        // thus only worry about sampling zeta_{n_init + 1} up to zeta_n
        // if k == k_init, this is an empty vector (all zetas will be zero)
        let q = if k == k_init {
            Vec::new().into_boxed_slice()
        } else {
            conditional_bernoulli_q(k - k_init, &p)
                .into_iter()
                .flatten()
                .collect()
        };

        Ok(ConditionalCRPSampler {
            _theta: theta,
            n,
            k,
            p,
            q,
            n_init,
            k_init,
            conf_init,
            assignment_init,
        })
    }
}

impl Sampler for ConditionalCRPSampler {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16> {
        // in the chinese restaurant process, the configuration is built
        // sample-by sample

        // assignment of samples to cycles
        let mut configuration = self.conf_init.clone();
        let mut assignment = self.assignment_init.clone();

        // samples_assigned will index assignment (it has elements 0..samples_assigned filled in)
        let mut samples_assigned = self.n_init;
        // cycle refers to most recently used cycle_id, zero-indexed, so do -1
        let mut cycle = self.k_init - 1;

        // first deal with the case in which k == k_init,
        // as then we don't even need to sample the zetas, they are all zero
        if self.k == self.k_init {
            // p here keeps track of the number of unassigned samples instead of zetas
            for _ in 0..self.p.len() {
                let choice = rng.random_range(..samples_assigned);
                let cycle_choice = assignment[choice];
                assignment[samples_assigned] = cycle_choice;

                // increment configuration
                configuration[cycle_choice] += 1;
                samples_assigned += 1;
            }
        } else {
            // otherwise, finish sampling the configuration as usual
            let zetas = conditional_bernoulli_sample(self.k - self.k_init, &self.p, &self.q, rng);

            for s in zetas.iter() {
                if *s == 0 {
                    // if s == 0, we must place
                    // the sample in the same cycle as
                    // a randomly chosen previous sample
                    let choice = rng.random_range(..samples_assigned);
                    let cycle_choice = assignment[choice];
                    assignment[samples_assigned] = cycle_choice;

                    // increment configuration
                    configuration[cycle_choice] += 1;
                } else {
                    // if s == 1, we must start a new cycle
                    cycle += 1;
                    assignment[samples_assigned] = cycle;

                    // increment configuration
                    // here we fill a new element
                    configuration.push(1);
                    // configuration[cycle] += 1;
                }

                samples_assigned += 1;
            }
        }

        configuration.sort_unstable();

        configuration
    }
}

// the biased sampler here is trying to model
// frequency dependence among restaurant-goers
// when bias is = 1 we get neutral model exactly, only much slower
// when bias > 1 (e.g. 1.1, 1.2, ...), customers prefer tables with more people
// when bias < 1, customers prefer empti-er table (probability of occupying an empty table doesn't change)
// mechanically, this is achieved by an adjustment to a table assignment rule,
//   where it's not the frequency of people at the table that matters, but frequency^bias
#[derive(Debug, Clone)]
pub struct BiasedCRPSampler {
    crp: ConditionalCRPSampler,
    bias: f64,
}

impl BiasedCRPSampler {
    pub fn new(
        theta: f64,
        n: usize,
        k: usize,
        bias: f64,
        initial_configuration: &str,
    ) -> Result<Self> {
        let crp = ConditionalCRPSampler::new(theta, n, k, initial_configuration)?;
        Ok(Self { crp, bias })
    }
}

impl Sampler for BiasedCRPSampler {
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<u16> {
        // in the chinese restaurant process, the configuration is built
        // sample-by sample

        // assignment of samples to cycles
        let mut configuration = self.crp.conf_init.clone();
        let mut assignment = self.crp.assignment_init.clone();

        // samples_assigned will index assignment (it has elements 0..samples_assigned filled in)
        let mut samples_assigned = self.crp.n_init;
        // cycle refers to most recently used cycle_id, zero-indexed, so do -1
        let mut cycle = self.crp.k_init - 1;

        // first deal with the case in which k == k_init,
        // as then we don't even need to sample the zetas, they are all zero
        if self.crp.k == self.crp.k_init {
            // p here keeps track of the number of unassigned samples instead of zetas
            for _ in 0..self.crp.p.len() {
                let cycle_choice = choose_biased(&configuration, samples_assigned, self.bias, rng);
                assignment[samples_assigned] = cycle_choice;

                // increment configuration
                configuration[cycle_choice] += 1;
                samples_assigned += 1;
            }
        } else {
            // otherwise, finish sampling the configuration as usual
            let zetas = conditional_bernoulli_sample(
                self.crp.k - self.crp.k_init,
                &self.crp.p,
                &self.crp.q,
                rng,
            );

            for s in zetas.iter() {
                if *s == 0 {
                    // if s == 0, we must place
                    // the sample in the same cycle as
                    // a randomly chosen previous sample
                    let cycle_choice =
                        choose_biased(&configuration, samples_assigned, self.bias, rng);
                    assignment[samples_assigned] = cycle_choice;

                    // increment configuration
                    configuration[cycle_choice] += 1;
                } else {
                    // if s == 1, we must start a new cycle
                    cycle += 1;
                    assignment[samples_assigned] = cycle;

                    // increment configuration
                    // here we fill a new element
                    configuration.push(1);
                    // configuration[cycle] += 1;
                }

                samples_assigned += 1;
            }
        }

        configuration.sort_unstable();

        configuration
    }
}

fn conditional_crp_bernoulli_p(n_init: usize, n: usize, theta: f64) -> Box<[f64]> {
    // compute probabilities for the chinese restaurant process
    // p_j = theta / (theta + j - 1),
    //   where j = 1, ..., n
    // NOTE: we do not need first variable as p_1 = 1
    ((n_init + 1)..=n)
        .map(|j| theta / (theta + (j as f64) - 1.))
        .collect()
}

fn choose_biased<R: Rng>(xs: &[u16], l: usize, bias: f64, rng: &mut R) -> usize {
    // compute frequencies and bias them
    let biased_xs: Vec<f64> = xs
        .iter()
        .map(|x| (*x as f64 / l as f64).powf(bias))
        .collect();
    // get the norm
    let norm: f64 = biased_xs.iter().sum();
    // get a random value in [0, norm)
    let r = rng.random::<f64>() * norm;
    // find the right i
    let mut i = 0;
    let mut csum = 0.0;
    for x in biased_xs {
        csum += x;
        if r < csum {
            return i;
        }
        i += 1;
    }
    i
}
