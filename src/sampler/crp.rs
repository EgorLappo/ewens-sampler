use crate::sampler::{conditional_bernoulli_probs, conditional_bernoulli_sample, Sampler};
use color_eyre::eyre::{bail, Result, WrapErr};
use rand::{Rng, RngExt};
use rand_pcg::Pcg64;

#[derive(Debug, Clone)]
pub struct CRPSamplerK {
    pub _theta: f64,
    pub n: usize,
    pub k: usize,
    n_vars: usize,
    probs: Box<[f64]>,
}

impl CRPSamplerK {
    pub fn new(theta: f64, n: usize, k: usize) -> Self {
        // unconstrained probabilities for zeta
        let p = crp_bernoulli_p(n, theta);
        // conditioned ones
        let probs = conditional_bernoulli_probs(k - 1, &p);
        Self {
            _theta: theta,
            n,
            k,
            n_vars: p.len(),
            probs,
        }
    }
}

impl Sampler for CRPSamplerK {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        let zetas = conditional_bernoulli_sample(self.n_vars, self.k - 1, &self.probs, rng);

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
    n_vars: usize,
    probs: Box<[f64]>,
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
        let probs = if k == k_init {
            Vec::new().into_boxed_slice()
        } else {
            conditional_bernoulli_probs(k - k_init, &p)
        };

        Ok(ConditionalCRPSampler {
            _theta: theta,
            n,
            k,
            n_vars: p.len(),
            probs,
            n_init,
            k_init,
            conf_init,
            assignment_init,
        })
    }
}

impl Sampler for ConditionalCRPSampler {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        // in the chinese restaurant process, the configuration is built
        // sample-by sample

        // assignment of samples to cycles
        let mut configuration = self.conf_init.clone();
        let mut assignment = self.assignment_init.clone();

        // samples_assigned will index assignment (it has elements 0..samples_assigned filled in)
        let mut samples_assigned = self.n_init;

        // first deal with the case in which k == k_init,
        // as then we don't even need to sample the zetas, they are all zero
        if self.k == self.k_init {
            // p here keeps track of the number of unassigned samples instead of zetas
            for _ in 0..self.n_vars {
                let choice = rng.random_range(..samples_assigned);
                let cycle_choice = assignment[choice];
                assignment[samples_assigned] = cycle_choice;

                // increment configuration
                configuration[cycle_choice] += 1;
                samples_assigned += 1;
            }
        } else {
            // otherwise, finish sampling the configuration as usual
            let zetas =
                conditional_bernoulli_sample(self.n_vars, self.k - self.k_init, &self.probs, rng);

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

                    // increment configuration
                    // here we fill a new element
                    configuration.push(1);
                    assignment[samples_assigned] = configuration.len() - 1;
                }

                samples_assigned += 1;
            }
        }

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
    // caches x^bias for integer x=0..=n
    pow_cache: Box<[f64]>,
}

impl BiasedCRPSampler {
    pub fn new(
        theta: f64,
        n: usize,
        k: usize,
        bias: f64,
        initial_configuration: &str,
    ) -> Result<Self> {
        if bias <= 0.0 || bias.is_infinite() {
            bail!(format!("invalid bias value {bias:?}. must be > 0"));
        }
        let crp = ConditionalCRPSampler::new(theta, n, k, initial_configuration)?;
        let pow_cache = (0..=n).map(|x| (x as f64).powf(bias)).collect();
        Ok(Self { crp, pow_cache })
    }
}

impl Sampler for BiasedCRPSampler {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16> {
        // in the chinese restaurant process, the configuration is built
        // sample-by sample

        // assignment of samples to cycles
        let mut configuration = self.crp.conf_init.clone();
        let mut assignment = self.crp.assignment_init.clone();

        // samples_assigned will index assignment (it has elements 0..samples_assigned filled in)
        let mut samples_assigned = self.crp.n_init;

        // first deal with the case in which k == k_init,
        // as then we don't even need to sample the zetas, they are all zero
        if self.crp.k == self.crp.k_init {
            // p here keeps track of the number of unassigned samples instead of zetas
            for _ in 0..self.crp.n_vars {
                let cycle_choice = choose_biased(&configuration, &self.pow_cache, rng);
                assignment[samples_assigned] = cycle_choice;

                // increment configuration
                configuration[cycle_choice] += 1;
                samples_assigned += 1;
            }
        } else {
            // otherwise, finish sampling the configuration as usual
            let zetas = conditional_bernoulli_sample(
                self.crp.n_vars,
                self.crp.k - self.crp.k_init,
                &self.crp.probs,
                rng,
            );

            for s in zetas.iter() {
                if *s == 0 {
                    // if s == 0, we must place
                    // the sample in the same cycle as
                    // a randomly chosen previous sample
                    let cycle_choice = choose_biased(&configuration, &self.pow_cache, rng);
                    assignment[samples_assigned] = cycle_choice;

                    // increment configuration
                    configuration[cycle_choice] += 1;
                } else {
                    // if s == 1, we must start a new cycle

                    // increment configuration
                    // here we fill a new element
                    configuration.push(1);
                    assignment[samples_assigned] = configuration.len() - 1;
                }

                samples_assigned += 1;
            }
        }

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

fn choose_biased<R: Rng>(xs: &[u16], pow: &[f64], rng: &mut R) -> usize {
    // get the norm
    let mut norm = 0.0;
    for &x in xs {
        norm += pow[x as usize];
    }

    // get a random value in [0, norm)
    let r = rng.random::<f64>() * norm;
    let mut csum = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        // access cached bias frequencies directly
        csum += pow[x as usize];
        if r < csum {
            return i;
        }
    }
    xs.len() - 1
}
