use clap::{ArgGroup, Parser};
use color_eyre::eyre::{Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::io::Write;

use ewens_sampler::Configuration;
use ewens_sampler::samplers::Sampler;
use ewens_sampler::samplers::ewens::{
    self, BiasedCRPSampler, ConditionalCRPSampler, FellerSampler, FellerSamplerK,
    check_initial_configuration,
};
use ewens_sampler::samplers::pitman_yor::{self, PYSampler, PYSamplerK};
use ewens_sampler::stats::Test;
use ewens_sampler::stats::ewens::{SlatkinTest, WattersonTest};
use ewens_sampler::stats::pitman_yor::{ExactPTest, ScoreTest};

const PROGRESS_STYLE: &str =
    "{spinner:.purple} [{elapsed}/{duration}] [{bar:.cyan/blue}] {human_pos}/{human_len}";

fn main() -> Result<()> {
    color_eyre::install()?;

    let opts = Opts::parse();

    // n and k come from the configuration to test, or from the options when just sampling
    // (where clap makes sure we have n)
    let (n, k) = match &opts.test {
        Some(tc) => (tc.n(), Some(tc.k())),
        None => (opts.n.unwrap(), opts.k),
    };

    // the process and what it is conditioned on decide the sampler
    let sampler = make_sampler(opts.model(), n, opts.conditioning(n, k)?)?;

    // then, will we test or sample?
    match &opts.test {
        Some(tc) => run_test(&*sampler, &make_tests(opts.model(), tc)?, tc, &opts),
        None => run_sampler(&*sampler, opts.samples, opts.seed, opts.quiet),
    }
}

/// the process the configurations come from
#[derive(Debug, Clone, Copy)]
enum Model {
    Ewens,
    /// the frequency-biased restaurant (experimental)
    Biased {
        bias: f64,
    },
    PitmanYor {
        alpha: f64,
    },
}

/// what the configurations are conditioned on, besides n
#[derive(Debug, Clone, Copy)]
enum Conditioning<'a> {
    /// nothing, so theta matters and has to be given
    Free { theta: f64 },
    /// the number of alleles, so theta drops out and we use its mle
    K { k: usize },
    /// k and an initial configuration that the samples extend
    Initial { k: usize, ic: &'a Configuration },
}

/// one sampler for every model and conditioning
fn make_sampler(model: Model, n: usize, given: Conditioning) -> Result<Box<dyn Sampler<Pcg64>>> {
    use Conditioning::*;
    use Model::*;

    // theta only matters without k. otherwise it is the mle, which for an
    // initial configuration is fitted to the samples after it
    let theta = match (model, given) {
        (_, Free { theta }) => theta,
        (PitmanYor { alpha }, K { k }) => pitman_yor::theta::theta_mle(n, k, alpha, None),
        (PitmanYor { alpha }, Initial { k, ic }) => {
            pitman_yor::theta::theta_mle(n, k, alpha, Some((ic.n(), ic.k())))
        }
        (_, K { k }) => ewens::theta::theta_mle(n, k, None),
        (_, Initial { k, ic }) => ewens::theta::theta_mle(n, k, Some((ic.n(), ic.k()))),
    };

    let sampler: Box<dyn Sampler<Pcg64>> = match (model, given) {
        (Ewens, Free { .. }) => Box::new(FellerSampler::new(theta, n)?),
        (Ewens, K { k }) => Box::new(FellerSamplerK::new(theta, n, k)?),
        (Ewens, Initial { k, ic }) => Box::new(ConditionalCRPSampler::new(theta, n, k, Some(ic))?),
        // clap only takes a bias with k
        (Biased { .. }, Free { .. }) => unreachable!("bias without k"),
        (Biased { bias }, K { k }) => Box::new(BiasedCRPSampler::new(theta, n, k, bias, None)?),
        (Biased { bias }, Initial { k, ic }) => {
            Box::new(BiasedCRPSampler::new(theta, n, k, bias, Some(ic))?)
        }
        (PitmanYor { alpha }, Free { .. }) => Box::new(PYSampler::new(theta, alpha, n)?),
        (PitmanYor { alpha }, K { k }) => Box::new(PYSamplerK::new(theta, alpha, n, k)?),
        (PitmanYor { .. }, Initial { .. }) => {
            unimplemented!("pitman-yor sampling from an initial configuration")
        }
    };
    Ok(sampler)
}

/// the tests for every model: pitman-yor gets its exact and score tests at the same alpha,
/// and the others slatkin's and watterson's
fn make_tests(model: Model, tc: &Configuration) -> Result<Vec<Box<dyn Test>>> {
    let tests: Vec<Box<dyn Test>> = match model {
        Model::PitmanYor { alpha } => vec![
            Box::new(ExactPTest::new(tc, alpha)?),
            Box::new(ScoreTest::new(tc, alpha)?),
        ],
        Model::Ewens | Model::Biased { .. } => vec![
            Box::new(SlatkinTest::new(tc)),
            Box::new(WattersonTest::new(tc)),
        ],
    };
    Ok(tests)
}

#[derive(Debug, Clone, Parser)]
#[command(version,
    about = "a ewens and pitman-yor distribution utility",
    long_about = None,
    group(ArgGroup::new("conditional").args(["k", "test"])),
)]
struct Opts {
    #[arg(
        short,
        value_name = "N",
        help = "number of sampled alleles",
        required_unless_present("test"), // if test configuration present we can compute n
        conflicts_with("test") // so forbid listing both
    )]
    n: Option<usize>,
    #[arg(
        short,
        value_name = "K",
        help = "number of alleles (for sampling with fixed k)",
        conflicts_with("test") // also don't give me k if i have configuration
    )]
    k: Option<usize>,
    #[arg(
        long,
        short,
        value_name = "REPLICATES",
        default_value_t = 1000,
        help = "number of sampled configurations to generate"
    )]
    samples: usize,
    #[arg(long, default_value_t = 231, help = "random seed")]
    seed: u64,
    #[arg(
        short,
        long,
        value_name = "THETA",
        help = "value of theta (doesn't affect the results when k is fixed)",
        allow_negative_numbers = true, // pitman-yor takes theta > -alpha
        required_unless_present("conditional") // don't need theta if have conditional sampling
    )]
    theta: Option<f64>,
    #[arg(
        short,
        long,
        value_name = "ALPHA",
        help = "if provided, sample from the Pitman-Yor distribution. alpha has to lie in [-inf, 1]",
        value_parser = parse_alpha,
        allow_hyphen_values = true, // for negative alpha, and -inf
        conflicts_with("bias") // pitman-yor supersedes the biased restaurant
    )]
    alpha: Option<f64>,
    #[arg(
        short,
        long = "initial-configuration",
        help = "initial configuration to start sampling from, uses conditional CRP sampler",
        requires("conditional") // cant sample starting from a fixed configuration unless with fixed k
    )]
    initial_configuration: Option<Configuration>,
    #[arg(
        short,
        long,
        help = "frequency-dependent bias factor of restaurant tables (experimental)",
        requires("conditional")
    )]
    bias: Option<f64>,
    #[arg(
        long,
        help = "configuration to run the exact test on; a space-separated list of unsigned integers; must have `k` elements with values summing to `n`"
    )]
    test: Option<Configuration>,
    #[arg(
        short,
        long,
        help = "output test results in JSON rather than human-readable form"
    )]
    json: bool,
    #[arg(short, long, help = "quiet run (no progress bar)")]
    quiet: bool,
}

impl Opts {
    fn model(&self) -> Model {
        match (self.alpha, self.bias) {
            // clap doesn't take both
            (Some(alpha), _) => Model::PitmanYor { alpha },
            // bias 1 is the neutral model
            (None, Some(bias)) if bias != 1.0 => Model::Biased { bias },
            _ => Model::Ewens,
        }
    }

    fn conditioning(&self, n: usize, k: Option<usize>) -> Result<Conditioning<'_>> {
        Ok(match (k, &self.initial_configuration) {
            // clap asks for theta without k, and only takes an initial configuration with k
            (None, _) => Conditioning::Free {
                theta: self.theta.unwrap(),
            },
            (Some(k), None) => {
                // validate before theta_mle sees it, as the pitman-yor one panics
                if !(1..=n).contains(&k) {
                    bail!("invalid parameters n={n}, k={k}. requires 1 <= k <= n");
                }
                Conditioning::K { k }
            }
            (Some(k), Some(ic)) => {
                // validate before theta_mle sees it, which also rules out a bad k
                check_initial_configuration(n, k, ic)?;
                Conditioning::Initial { k, ic }
            }
        })
    }
}

fn parse_alpha(s: &str) -> Result<f64, String> {
    let alpha: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if alpha <= 1. {
        Ok(alpha)
    } else {
        Err("alpha has to be at most 1".to_string())
    }
}

/// run sampler and print configurations to stdout
fn run_sampler(sampler: &dyn Sampler<Pcg64>, samples: usize, seed: u64, quiet: bool) -> Result<()> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();

    let pb = if quiet {
        None
    } else {
        let style = ProgressStyle::with_template(PROGRESS_STYLE).unwrap();
        let pb = ProgressBar::new(samples as u64).with_style(style);
        Some(pb)
    };

    for _ in 0..samples {
        let mut samp = sampler.sample(&mut rng);
        // only sort when outputting in ASCII
        samp.sort_unstable();

        writeln!(stdout, "{}", samp.iter().format(" "))?;

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("done sampling");
    }

    stdout.flush()?;

    Ok(())
}

/// run sampler and test specific configuration
fn run_test(
    sampler: &dyn Sampler<Pcg64>,
    tests: &[Box<dyn Test>],
    test_configuration: &Configuration,
    opts: &Opts,
) -> Result<()> {
    let ic = opts.initial_configuration.as_ref();
    let samples = opts.samples;
    let seed = opts.seed;
    let mut rng = Pcg64::seed_from_u64(seed);

    let pb = if opts.quiet {
        None
    } else {
        let style = ProgressStyle::with_template(PROGRESS_STYLE).unwrap();
        let pb = ProgressBar::new(samples as u64).with_style(style);
        Some(pb)
    };

    // replicates in the lower and upper tails of each test,
    // where ties with the observed configuration count in both
    let mut tails = vec![(0usize, 0usize); tests.len()];

    for _ in 0..samples {
        let mut samp = sampler.sample(&mut rng);
        // only sort when outputting in ASCII
        samp.sort_unstable();

        for (t, (lower, upper)) in tests.iter().zip(&mut tails) {
            let o = t.compare(&samp);
            *lower += o.is_le() as usize;
            *upper += o.is_ge() as usize;
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("done testing");
    }

    // the observed configuration is in both of its own tails.
    let p = |count: usize| (count + 1) as f64 / (samples + 1) as f64;

    if opts.json {
        let results: Vec<_> = tests
            .iter()
            .zip(&tails)
            .map(|(t, &(lower, upper))| {
                serde_json::json!({
                    "test": t.name(),
                    "lower_count": lower,
                    "upper_count": upper,
                    "total_count": samples,
                    "p_lower": p(lower),
                    "p_upper": p(upper),
                })
            })
            .collect();
        let mut val = serde_json::json!({
            "configuration": test_configuration.to_string(),
            "theta": sampler.theta(),
            "seed": opts.seed,
            "replicates": samples,
            "tests": results,
        });

        if let Some(val) = val.as_object_mut() {
            if let Some(ic) = ic {
                val.insert(
                    "initial_configuration".to_string(),
                    serde_json::json!(ic.to_string()),
                );
            }

            if let Some(b) = opts.bias {
                val.insert("bias".to_string(), serde_json::json!(b));
            }

            if let Some(a) = opts.alpha {
                val.insert("alpha".to_string(), serde_json::json!(a));
            }
        }

        let string = serde_json::to_string(&val)?;
        println!("{}", string);
    } else {
        // print the results here in human-readable form
        println!("seed\t{}", seed);
        println!("configuration\t'{}'", test_configuration);
        if let Some(ic) = ic {
            println!("initial configuration\t'{}'", ic);
        }
        if let Some(b) = opts.bias {
            println!("bias\t{}", b)
        }
        if let Some(a) = opts.alpha {
            println!("alpha\t{}", a)
        }
        println!("theta\t{}", sampler.theta());

        for (t, &(lower, upper)) in tests.iter().zip(&tails) {
            println!("{}\tlower_count\t{}", t.name(), lower);
            println!("{}\tupper_count\t{}", t.name(), upper);
            println!("{}\ttotal_count\t{}", t.name(), samples);
            println!("{}\tp_lower\t{}", t.name(), p(lower));
            println!("{}\tp_upper\t{}", t.name(), p(upper));
        }
    }
    Ok(())
}
