//! criterion benchmarks for the samplers
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use itertools::Itertools;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use ewens_sampler::Configuration;
use ewens_sampler::samplers::Sampler;
use ewens_sampler::samplers::ewens::theta::theta_mle;
use ewens_sampler::samplers::ewens::{
    CRPSamplerK, ConditionalCRPSampler, FellerSampler, FellerSamplerK,
};
use ewens_sampler::samplers::pitman_yor::PYSamplerK;
use ewens_sampler::stats::{SlatkinTest, Test, WattersonTest};

const SEED: u64 = 231;

// sample sizes from small data sets to past the largest one in slatkin (1996), n = 16975
const NS: [usize; 5] = [50, 200, 1_000, 5_000, 25_000];
// numbers of alleles
const KS: [usize; 3] = [5, 20, 80];

// every (n, k) on the grid with k <= n
fn grid() -> impl Iterator<Item = (usize, usize)> {
    NS.into_iter()
        .cartesian_product(KS)
        .filter(|&(n, k)| k <= n)
}

type Make = fn(f64, usize, usize) -> Box<dyn Sampler<Pcg64>>;

// the conditional samplers, built for (theta, n, k)
fn samplers() -> [(&'static str, Make); 4] {
    [
        ("feller", |t, n, k| {
            Box::new(FellerSamplerK::new(t, n, k).unwrap())
        }),
        ("crp", |t, n, k| Box::new(CRPSamplerK::new(t, n, k))),
        ("conditional_crp", |t, n, k| {
            Box::new(ConditionalCRPSampler::new(t, n, k, None).unwrap())
        }),
        // alpha matters for speed, as joining a table is retried w.p. alpha at most
        ("pitman_yor", |t, n, k| {
            Box::new(PYSamplerK::new(t, 0.5, n, k))
        }),
    ]
}

// building the table of conditional probabilities

fn bench_setup(c: &mut Criterion) {
    for (name, make) in samplers() {
        let mut group = c.benchmark_group(format!("setup/{name}"));
        for (n, k) in grid() {
            let theta = theta_mle(n, k, None);
            // throughput is in entries of the table
            group.throughput(Throughput::Elements((n * k) as u64));
            group.bench_function(BenchmarkId::new(format!("k={k}"), n), |b| {
                b.iter(|| make(black_box(theta), black_box(n), black_box(k)))
            });
        }
        group.finish();
    }
}

// drawing one configuration

fn bench_sample(c: &mut Criterion) {
    for (name, make) in samplers() {
        let mut group = c.benchmark_group(format!("sample/{name}"));
        for (n, k) in grid() {
            let sampler = make(theta_mle(n, k, None), n, k);
            let mut rng = Pcg64::seed_from_u64(SEED);
            // throughput is in customers, so flat lines mean O(n) per draw
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(BenchmarkId::new(format!("k={k}"), n), |b| {
                b.iter(|| sampler.sample(&mut rng))
            });
        }
        group.finish();
    }
}

// baseline: rejection sampling
// theta_mle gives best possible acceptance probability
fn bench_rejection(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample/rejection");
    for (n, k) in grid() {
        let sampler = FellerSampler::new(theta_mle(n, k, None), n);
        let mut rng = Pcg64::seed_from_u64(SEED);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new(format!("k={k}"), n), |b| {
            b.iter(|| {
                loop {
                    let conf = sampler.sample(&mut rng);
                    if conf.len() == k {
                        break conf;
                    }
                }
            })
        });
    }
    group.finish();
}

// one replicate of the test, as in the cli: draw, sort, and evaluate both statistics

fn bench_replicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("replicate/feller");
    for (n, k) in grid() {
        let sampler = FellerSamplerK::new(theta_mle(n, k, None), n, k).unwrap();
        let mut rng = Pcg64::seed_from_u64(SEED);

        // test against some configuration with these n and k
        let mut observed = sampler.sample(&mut rng);
        observed.sort_unstable();
        let observed: Configuration = observed.iter().join(" ").parse().unwrap();
        let (s, w) = (SlatkinTest::new(&observed), WattersonTest::new(&observed));

        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new(format!("k={k}"), n), |b| {
            b.iter(|| {
                let mut samp = sampler.sample(&mut rng);
                samp.sort_unstable();
                (s.test(&samp), w.test(&samp))
            })
        });
    }
    group.finish();
}

// bench passing different thetas into fix-K samplers
// this shouldn't change speed
fn bench_theta(c: &mut Criterion) {
    let (n, k) = (265, 81);
    let mut group = c.benchmark_group("theta/feller");
    for theta in [0.01, 1.0, theta_mle(n, k, None), 100.0] {
        let sampler = FellerSamplerK::new(theta, n, k).unwrap();
        let mut rng = Pcg64::seed_from_u64(SEED);
        group.bench_function(BenchmarkId::from_parameter(format!("{theta:.3}")), |b| {
            b.iter(|| sampler.sample(&mut rng))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    // with the default 3s warm up and 5s measurement the whole grid takes far too long
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_setup, bench_sample, bench_rejection, bench_replicate, bench_theta
}
criterion_main!(benches);
