use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::eyre::{bail, Result, WrapErr};
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use rand::{rngs::SmallRng, SeedableRng};
use std::io::{ErrorKind, Read, Write};

use ewens_sampler::sampler::{ConditionalCRPSampler, FellerSampler, FellerSamplerK, Sampler};

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
            fmt,
        } => {
            if let Some(k) = k {
                // sample conditional on k
                if let Some(ic) = initial_configuration {
                    // use crp if initial configuration provided
                    let sampler = ConditionalCRPSampler::new(theta, n, k, &ic)?;
                    sample(sampler, samples, seed, fmt, k)?;
                } else {
                    // use feller otherwise
                    let sampler = FellerSamplerK::new(theta, n, k);
                    sample(sampler, samples, seed, fmt, k)?;
                }
            } else {
                // sample unconditionally
                if initial_configuration.is_some() {
                    bail!(
                        "please provide value of 'k' for sampling with fixed initial configuration!"
                    )
                }
                let sampler = FellerSampler::new(theta, n);
                sample(sampler, samples, seed, fmt, n)?;
            }
        }
        Command::Test { ref configuration } => test(configuration)?,
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
        #[arg(value_enum, short='c', long="format", default_value_t = OutputFormat::Binary, help = "output format")]
        fmt: OutputFormat,
    },
    Test {
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
    let mut rng = SmallRng::seed_from_u64(seed);
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
                let samp = sampler.sample(&mut rng);

                writeln!(stdout, "{}", samp.iter().format(" "))?;
            }
        }
    }

    stdout.flush()?;

    Ok(())
}

fn test(configuration: &str) -> Result<()> {
    // parse the configuration
    //   the configuration is a list like "34 12 7 9 2 1 1" of length k and summing to n
    //   probability is proportional to 1/ the product of the values,
    //   so the log-probability is proportional to - sum of logs of these
    let uconf = configuration
        .split(' ')
        .map(|s| s.parse::<u16>())
        .collect::<Result<Vec<_>, _>>()
        .wrap_err_with(|| {
            format!(
                "failed while parsing configuration {:?} into unsigned integers",
                configuration
            )
        })?;

    let k = uconf.len();
    // let n = uconf.iter().sum::<u16>();

    // this is the -log P(c_0|k)
    let lp: f64 = uconf.into_iter().map(|x| (x as f64).ln()).sum();

    // now read the file from stdin as bytes
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    let buf_size = 2 * k * 1000;

    let mut buf = vec![0; buf_size];

    let mut total: usize = 0;
    let mut count: usize = 0;

    loop {
        match stdin.read_exact(&mut buf) {
            Ok(_) => {
                // iter over configurations
                let chunks = buf.chunks_exact(2 * k);

                chunks.for_each(|c| {
                    // iter over configuration entries
                    // to get - log P(c|k)
                    // println!("{:?}", c.len());
                    let lp_c: f64 = c
                        .chunks_exact(2)
                        .map(|x| (LittleEndian::read_u16(x) as f64).ln())
                        .sum();

                    // now we want to test if P(c|k) <= P(c_0|k)
                    //   in -log-p, we test -log P(c|k) >= -log P(c_0|k)
                    if lp_c >= lp {
                        count += 1
                    }

                    total += 1;
                });
            }

            Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => {
                break;
            }

            Err(e) => {
                bail!(e)
            }
        }
    }

    println!("{:?} {:?}", count, total);
    Ok(())
}
