use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use ewens_sampler::sampler::*;
use rand::{SeedableRng, rngs::SmallRng};

pub fn feller_theta_comparison_small(c: &mut Criterion) {
    // first test on small value (n,k) = (16,7)
    let (n, k) = (16, 7);
    let mut group = c.benchmark_group("theta_comparison_small");

    for theta in [0.01, 0.5, 1.0, 5.0, 10.0, 25.0, 100.0].into_iter() {
        let feller = FellerSamplerK::new(black_box(theta), black_box(n), black_box(k));
        let mut rng = SmallRng::seed_from_u64(123);

        group.bench_with_input(BenchmarkId::from_parameter(theta), &theta, |b, _theta| {
            b.iter(|| feller.sample(&mut rng));
        });
    }

    group.finish();
}

pub fn feller_theta_comparison_large(c: &mut Criterion) {
    // then test on large value (n,k) = (265,81)
    let (n, k) = (265, 81);
    let mut group = c.benchmark_group("theta_comparison_large");

    for theta in [0.01, 0.5, 1.0, 5.0, 10.0, 25.0, 100.0].into_iter() {
        let feller = FellerSamplerK::new(black_box(theta), black_box(n), black_box(k));
        let mut rng = SmallRng::seed_from_u64(123);

        group.bench_with_input(BenchmarkId::from_parameter(theta), &theta, |b, _theta| {
            b.iter(|| feller.sample(&mut rng));
        });
    }

    group.finish();
}

pub fn feller_crp_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("method_comparison");

    let (n, k) = (265, 81);
    let mut rng = SmallRng::seed_from_u64(123);
    let feller = FellerSamplerK::new(black_box(1.0), black_box(n), black_box(k));
    let crp = CRPSamplerK::new(black_box(1.0), black_box(n), black_box(k));

    group.bench_function("feller", |b| b.iter(|| feller.sample(&mut rng)));

    group.bench_function("crp", |b| b.iter(|| crp.sample(&mut rng)));

    group.finish();
}

pub fn bench_feller(c: &mut Criterion) {
    let (n, k) = (265, 81);
    let mut rng = SmallRng::seed_from_u64(231);
    let feller = FellerSamplerK::new(black_box(1.0), black_box(n), black_box(k));

    c.bench_function("feller", |b| b.iter(|| feller.sample(&mut rng)));
}

pub fn bench_crp(c: &mut Criterion) {
    let (n, k) = (265, 81);
    let mut rng = SmallRng::seed_from_u64(231);
    let crp = CRPSamplerK::new(black_box(1.0), black_box(n), black_box(k));

    c.bench_function("crp", |b| b.iter(|| crp.sample(&mut rng)));
}

criterion_group!(
    comparison_benches,
    feller_theta_comparison_small,
    feller_theta_comparison_large,
    feller_crp_comparison
);
criterion_group!(implementation_benches, bench_feller, bench_crp);
criterion_main!(comparison_benches, implementation_benches);
