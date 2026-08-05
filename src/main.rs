use clap::{ArgGroup, Parser};
use color_eyre::eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::io::Write;

use ewinfer::Configuration;
use ewinfer::sampler::{
    BiasedCRPSampler, ConditionalCRPSampler, FellerSampler, FellerSamplerK, Sampler,
};
use ewinfer::stats::{SlatkinTest, Test, WattersonTest};
use ewinfer::theta::theta_mle;

const PROGRESS_STYLE: &str =
    "{spinner:.purple} [{elapsed}/{duration}] [{bar:.cyan/blue}] {human_pos}/{human_len}";

fn main() -> Result<()> {
    color_eyre::install()?;

    let opts = Opts::parse();

    // first, will we test or sample?
    if let Some(ref tc) = opts.test {
        let n = tc.n();
        let k = tc.k();

        // do we have an initial configuration or not
        if let Some(ref ic) = opts.initial_configuration {
            let n0 = ic.n();
            let k0 = ic.k();
            let theta = theta_mle(n, k, Some((n0, k0)));
            // do we have nontrivial bias requested?
            if let Some(bias) = opts.bias
                && bias != 1.0
            {
                // use biased CRP
                let sampler = BiasedCRPSampler::new(theta, n, k, bias, Some(&ic))?;
                run_test(sampler, tc, Some(&ic), &opts)?;
            } else {
                // just use CRP
                let sampler = ConditionalCRPSampler::new(theta, n, k, Some(&ic))?;
                run_test(sampler, tc, Some(&ic), &opts)?;
            }
        } else {
            let theta = theta_mle(n, k, None);
            // do we have nontrivial bias requested?
            if let Some(bias) = opts.bias
                && bias != 1.0
            {
                // here ic is empty
                let sampler = BiasedCRPSampler::new(theta, n, k, bias, None)?;
                run_test(sampler, tc, None, &opts)?;
            } else {
                // if not, just sample with Feller again
                let sampler = FellerSamplerK::new(theta, n, k)?;
                run_test(sampler, tc, None, &opts)?;
            }
        }
    } else {
        // if we are just sampling, is it fixed k or not?
        if let Some(k) = opts.k {
            let n = opts.n.unwrap();

            // if fixed k, do we have an initial configuration or not
            if let Some(ic) = opts.initial_configuration {
                let n0 = ic.n();
                let k0 = ic.k();
                let theta = theta_mle(n, k, Some((n0, k0)));

                // do we have nontrivial bias requested?
                if let Some(bias) = opts.bias
                    && bias != 1.0
                {
                    // use biased CRP
                    let sampler = BiasedCRPSampler::new(theta, n, k, bias, Some(&ic))?;
                    run_sampler(sampler, opts.samples, opts.seed, opts.quiet)?;
                } else {
                    // just use CRP
                    let sampler = ConditionalCRPSampler::new(theta, n, k, Some(&ic))?;
                    run_sampler(sampler, opts.samples, opts.seed, opts.quiet)?;
                }
            } else {
                let theta = theta_mle(n, k, None);
                // do we have nontrivial bias requested?
                if let Some(bias) = opts.bias
                    && bias != 1.0
                {
                    // here ic is empty
                    let sampler = BiasedCRPSampler::new(theta, n, k, bias, None)?;
                    run_sampler(sampler, opts.samples, opts.seed, opts.quiet)?;
                } else {
                    // if not, just sample with Feller again
                    let sampler = FellerSamplerK::new(theta, n, k)?;
                    run_sampler(sampler, opts.samples, opts.seed, opts.quiet)?;
                }
            }
        } else {
            // without fixed k we just use Feller sampler,
            // and n, theta should be enforced by clap
            let n = opts.n.unwrap();
            let theta = opts.theta.unwrap();

            let sampler = FellerSampler::new(theta, n);
            run_sampler(sampler, opts.samples, opts.seed, opts.quiet)?;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Parser)]
#[command(version,
    about = "a ewens distribution utility",
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
        required_unless_present("conditional") // don't need theta if have conditional sampling
    )]
    theta: Option<f64>,
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

/// run sampler and print configurations to stdout
fn run_sampler(sampler: impl Sampler, samples: usize, seed: u64, quiet: bool) -> Result<()> {
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
    sampler: impl Sampler,
    test_configuration: &Configuration,
    ic: Option<&Configuration>,
    opts: &Opts,
) -> Result<()> {
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

    // writing this non-generically for now
    // if MAYBE we make more tests, we can have a
    // list of "dyn Test"s and go over them
    let s = SlatkinTest::new(&test_configuration);
    let w = WattersonTest::new(&test_configuration);

    let mut s_total: usize = 0;
    let mut w_total: usize = 0;

    for _ in 0..samples {
        let mut samp = sampler.sample(&mut rng);
        // only sort when outputting in ASCII
        samp.sort_unstable();

        if s.test(&samp) {
            s_total += 1;
        }
        if w.test(&samp) {
            w_total += 1;
        }

        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("done testing");
    }

    let s_ptail = (s_total + 1) as f64 / (samples + 1) as f64;
    let w_ptail = (w_total + 1) as f64 / (samples + 1) as f64;

    if opts.json {
        let mut val = serde_json::json!({
            "configuration": test_configuration.to_string(),
            "theta": sampler.theta(),
            "seed": opts.seed,
            "replicates": samples,
            "tests": [
               {
                   "test": s.name(),
                   "tail_count": s_total,
                   "total_count": samples,
                   "p_tail": s_ptail,
               },
               {
                   "test": w.name(),
                   "tail_count": w_total,
                   "total_count": samples,
                   "p_tail": w_ptail,
               }
            ]
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
        println!("theta\t{}", sampler.theta());

        println!("{}\ttail_count \t{}", s.name(), s_total);
        println!("{}\ttotal_count\t{}", s.name(), samples);
        println!("{}\tp_tail\t{}", s.name(), s_ptail);

        println!("{}\ttail_count \t{}", w.name(), w_total);
        println!("{}\ttotal_count\t{}", w.name(), samples);
        println!("{}\tp_tail\t{}", w.name(), w_ptail);
    }
    Ok(())
}
