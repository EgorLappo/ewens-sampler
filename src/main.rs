use byteorder::{LittleEndian, WriteBytesExt};
use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{bail, Result};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::io::Write;

use ewinfer::sampler::{
    BiasedCRPSampler, ConditionalCRPSampler, FellerSampler, FellerSamplerK, Sampler,
};
use ewinfer::tests::{slatkin_test, watterson_test};

fn main() -> Result<()> {
    color_eyre::install()?;

    let opts = Opts::parse();

    match opts.command {
        Command::Sample {
            n,
            k,
            samples,
            seed,
            theta,
            initial_configuration,
            bias,
            fmt,
        } => {
            if let Some(k) = k {
                // sample conditional on k
                if let Some(ic) = initial_configuration {
                    // use crp if initial configuration provided
                    // and run the biased version if, well, bias is provided...
                    if let Some(bias) = bias {
                        let sampler = BiasedCRPSampler::new(theta, n, k, bias, &ic)?;
                        sample(sampler, samples, seed, fmt, k)?;
                    } else {
                        let sampler = ConditionalCRPSampler::new(theta, n, k, &ic)?;
                        sample(sampler, samples, seed, fmt, k)?;
                    }
                } else {
                    // if bias given, fallback on the (slow) biased CRP with empy IC
                    if let Some(bias) = bias {
                        let sampler = BiasedCRPSampler::new(theta, n, k, bias, "")?;
                        sample(sampler, samples, seed, fmt, k)?;
                    } else {
                        // use conditional feller otherwise
                        let sampler = FellerSamplerK::new(theta, n, k);
                        sample(sampler, samples, seed, fmt, k)?;
                    }
                };
            } else {
                if initial_configuration.is_some() {
                    bail!(
                        "please provide value of 'k' for sampling with fixed initial configuration!"
                    )
                }
                if bias.is_some() {
                    bail!("please provide value of 'k' for biased sampling")
                }

                // otherwise, sample unconditionally
                let sampler = FellerSampler::new(theta, n);
                sample(sampler, samples, seed, fmt, n)?;
            }
        }
        Command::Test {
            kind,
            ref configuration,
        } => match kind {
            TestKind::Slatkin => slatkin_test(configuration)?,
            TestKind::Watterson => watterson_test(configuration)?,
        },
    }

    Ok(())
}

#[derive(Debug, Clone, Parser)]
#[command(version, about = "a ewens distribution utility", long_about = None)]
struct Opts {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// output as sequence of (native-endian) u16 values;
    /// when reading, consume it in chunks to get configurations;
    /// chunk size is k if sampling with fixed k, or n if sampling from unconstrained Ewens distribution
    Binary,
    /// output as ASCII characters, space-separated, one configuration per line;
    Tabular,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum TestKind {
    /// Slatkin's exact test
    Slatkin,
    /// Watterson's homozygosity test
    Watterson,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    Sample {
        #[arg(short, value_name = "N", help = "number of samples")]
        n: usize,
        #[arg(
            short,
            value_name = "K",
            help = "number of alleles (for sampling with fixed k)"
        )]
        k: Option<usize>,
        #[arg(
            default_value_t = 100,
            help = "number of sampled configurations to generate"
        )]
        samples: usize,
        #[arg(long, default_value_t = 231, help = "random seed")]
        seed: u64,
        #[arg(
            short,
            long,
            value_name = "THETA",
            default_value_t = 1.0,
            help = "value of theta (doesn't affect the results when k is fixed)"
        )]
        theta: f64,
        #[arg(
            short,
            long = "initial-configuration",
            help = "initial configuration to start sampling from, uses conditional CRP sampler"
        )]
        initial_configuration: Option<String>,
        #[arg(
            short,
            long,
            help = "frequency-dependent bias factor of restaurant tables (should not be used in most analyses)"
        )]
        bias: Option<f64>,
        #[arg(value_enum, short='c', long="format", default_value_t = OutputFormat::Binary, help = "output format")]
        fmt: OutputFormat,
    },
    Test {
        #[arg(value_enum, help = "test kind", value_name="KIND", default_value_t=TestKind::Slatkin)]
        kind: TestKind,
        #[arg(
            help = "configuration to run the exact test on; a space-separated list of unsigned integers; must have `k` elements with values summing to `n`"
        )]
        configuration: String,
    },
}

fn sample(
    sampler: impl Sampler,
    samples: usize,
    seed: u64,
    format: OutputFormat,
    slen: usize,
) -> Result<()> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut stdout = std::io::stdout();

    let style = ProgressStyle::with_template(
        "{spinner:.purple} [{elapsed}/{duration}] [{bar:.cyan/blue}] {human_pos}/{human_len}",
    )
    .unwrap();

    match format {
        OutputFormat::Binary => {
            for _ in (0..samples).progress_with_style(style) {
                // if sampled with fixed k, slen == k, if not, slen == n
                let mut s = sampler.sample(&mut rng);
                s.resize(slen, 0);
                for x in s {
                    stdout.write_u16::<LittleEndian>(x)?
                }
            }
        }
        OutputFormat::Tabular => {
            for _ in (0..samples).progress_with_style(style) {
                let mut samp = sampler.sample(&mut rng);
                // only sort when outputting in ASCII
                samp.sort_unstable();

                writeln!(stdout, "{}", samp.iter().format(" "))?;
            }
        }
    }

    stdout.flush()?;

    Ok(())
}
