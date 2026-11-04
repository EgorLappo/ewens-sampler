use super::ln_add_exp;
pub mod crp;
pub mod feller;
pub mod theta;

pub use crp::{BiasedCRPSampler, CRPSamplerK, ConditionalCRPSampler, check_initial_configuration};
pub use feller::{FellerSampler, FellerSamplerK};

// *conditional bernoulli probabilities for the ewens zetas
// see reference module below for the code that follows the
// algorithm directly without logs
fn conditional_bernoulli_log_q(s: usize, p: &[f64]) -> Box<[f64]> {
    debug_assert!(
        p.iter().all(|&x| (0.0..=1.0).contains(&x)),
        "p outside [0,1]"
    );
    let n = p.len();
    assert!(s <= n, "conditional_bernoulli: sum > number of variables");

    // the one thing here is that in log space we will see lots of -inf's rather than 0's
    // also, optimize the pointer chasing with a flat vec rather than nesting

    // q[i,j] = probability that the sum of the variables number j to n is i
    let mut lq = vec![f64::NEG_INFINITY; (s + 1) * n];
    if n == 0 {
        return lq.into_boxed_slice();
    }

    let ln_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let ln_1mp: Vec<f64> = p.iter().map(|x| (1.0 - x).ln()).collect();

    // fill in q[0,:]
    //   (these are cases where all bernoulli trials failed, so products of 1-p_i)
    let mut acc = 0.0;
    for j in (0..n).rev() {
        acc += ln_1mp[j];
        lq[j] = acc;
    }

    if s == 0 {
        return lq.into_boxed_slice();
    }

    // if we have more "rows", fill them next
    // a neat trick is to define an access helper doing addition for the flat array
    // and guarding edge cases (se we loop from 1 now)
    #[inline]
    fn at(lq: &[f64], n: usize, i: usize, j: usize) -> f64 {
        if j >= n {
            if i == 0 { 0.0 } else { f64::NEG_INFINITY }
        } else {
            lq[i * n + j]
        }
    }

    for i in 1..=s {
        for j in (0..=(n - i)).rev() {
            // q[i][j] = p[j] * q[i - 1][j + 1] + (1. - p[j]) * q[i][j + 1];
            let a = ln_p[j] + at(&lq, n, i - 1, j + 1);
            let b = ln_1mp[j] + at(&lq, n, i, j + 1);
            lq[i * n + j] = ln_add_exp(a, b);
        }
    }

    lq.into_boxed_slice()
}

// now we can split off a function recovering the probabilities that are
// let pi = pi * q[n * (s - csum - 1) + i + 1] / q[n * (s - csum) + i];
fn conditional_bernoulli_probs(s: usize, p: &[f64]) -> Box<[f64]> {
    let n = p.len();
    if s == 0 {
        return Vec::new().into_boxed_slice();
    }

    let lq = conditional_bernoulli_log_q(s, p);
    let ln_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();

    let mut probs = vec![0.0f64; s * n];

    for i in 1..=s {
        for j in 0..=(n - i) {
            let ln_denum = lq[i * n + j];
            let ln_num = ln_p[j]
                + if j + 1 == n {
                    0.0
                } else {
                    lq[(i - 1) * n + j + 1]
                };
            let val = (ln_num - ln_denum).exp();
            probs[(i - 1) * n + j] = val.clamp(0.0, 1.0);
        }
    }
    probs.into_boxed_slice()
}

#[allow(dead_code)]
mod reference {
    //! reference implementation without logarithms

    use rand::{Rng, RngExt};

    // generates a sample of bernoulli variables, each with parameter p_i,
    // conditional on their sum being equal to s
    fn conditional_bernoulli_q(s: usize, p: &[f64]) -> Vec<Vec<f64>> {
        // we are following arxiv:2012.03103 for the sampling algorithm

        let n = p.len();

        if s > n {
            panic!("conditional_bernoulli: sum > number of variables");
        }

        // q[i,j] = probability that the sum of the variables number j to n is i
        let mut q: Vec<Vec<f64>> = vec![vec![0.0; n]; s + 1];

        // fill in q[0,:]
        //   (these are cases where all bernoulli trials failed, so products of 1-p_i)
        for j in 0..n {
            q[0][j] = p[j..n].iter().map(|x| 1. - x).product();
        }

        // fill in q[1,:]
        q[1][n - 1] = p[n - 1];
        for j in (0..(n - 1)).rev() {
            q[1][j] = p[j] * q[0][j + 1] + (1. - p[j]) * q[1][j + 1];
        }

        for i in 2..(s + 1) {
            // q[i][j] is zero when i > n - j + 1,
            //   equivalently, we only need to consider entries where j <= n - i + 1
            for j in (0..(n - i + 1)).rev() {
                if i <= n - j + 1 {
                    q[i][j] = p[j] * q[i - 1][j + 1] + (1. - p[j]) * q[i][j + 1];
                }
            }
        }

        q
    }

    fn conditional_bernoulli_sample<R: Rng>(
        s: usize,
        p: &[f64],
        q: &[f64],
        rng: &mut R,
    ) -> Vec<usize> {
        // q here is flattened so q[i, j] is q[(s+1)*i + j]

        let n = p.len();
        let mut ans = Vec::with_capacity(n);
        let mut csum = 0;

        let p0 = p[0] * q[n * (s - 1) + 1] / q[n * s];
        let x0 = (rng.random::<f64>() < p0) as usize;
        csum += x0;
        ans.push(x0);

        for (i, pi) in p.iter().enumerate().take(p.len() - 1).skip(1) {
            // if we already got the sum, the rest must be zero
            if csum == s {
                ans.push(0);
                continue;
            }
            //let pi = pi * q[s - csum - 1][i + 1] / q[s - csum][i];
            let pi = pi * q[n * (s - csum - 1) + i + 1] / q[n * (s - csum) + i];
            let xi = (rng.random::<f64>() < pi) as usize;
            csum += xi;
            ans.push(xi);
        }

        ans.push(s - csum);

        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::conditional_bernoulli_sample;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    // all 0/1 sequences of length len
    fn sequences(len: usize) -> impl Iterator<Item = Vec<usize>> {
        (0..1usize << len).map(move |b| (0..len).map(|m| (b >> m) & 1).collect())
    }

    fn path_prob(p: &[f64], zs: &[usize]) -> f64 {
        p.iter()
            .zip(zs)
            .map(|(&pm, &z)| if z == 1 { pm } else { 1.0 - pm })
            .product()
    }

    // probability that the sum of variables j to n-1 is i, by enumeration
    fn brute_q(p: &[f64], i: usize, j: usize) -> f64 {
        sequences(p.len() - j)
            .filter(|zs| zs.iter().sum::<usize>() == i)
            .map(|zs| path_prob(&p[j..], &zs))
            .sum()
    }

    // a few probability vectors, including the leading p = 1 of the conditional crp
    fn test_ps() -> [Vec<f64>; 4] {
        [
            vec![0.3, 0.9, 0.05, 0.5, 0.7, 0.2],
            (0..7).map(|j| 1.5 / (1.5 + j as f64)).collect(),
            vec![0.999, 0.001, 0.6],
            vec![0.4],
        ]
    }

    #[test]
    fn log_q_matches_brute_force() {
        for p in test_ps() {
            let n = p.len();
            for s in 0..=n {
                let lq = conditional_bernoulli_log_q(s, &p);
                for i in 0..=s {
                    for j in 0..n {
                        let (got, want) = (lq[i * n + j].exp(), brute_q(&p, i, j));
                        assert!(
                            (got - want).abs() < 1e-12,
                            "{p:?} s={s} i={i} j={j}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn probs_match_brute_force() {
        for p in test_ps() {
            let n = p.len();
            for s in 1..=n {
                let probs = conditional_bernoulli_probs(s, &p);
                for i in 1..=s {
                    for j in 0..=(n - i) {
                        let denom = brute_q(&p, i, j);
                        if denom == 0.0 {
                            // the sampler never gets here
                            continue;
                        }
                        let want = p[j] * brute_q(&p, i - 1, j + 1) / denom;
                        let got = probs[(i - 1) * n + j];
                        assert!(
                            (got - want).abs() < 1e-12,
                            "{p:?} s={s} i={i} j={j}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    // every sample should sum to s and be a possible outcome,
    // e.g. variables with p = 1 always come out as 1
    #[test]
    fn samples_are_possible_and_sum_to_s() {
        let mut rng = Pcg64::seed_from_u64(1);
        for p in test_ps() {
            let n = p.len();
            for s in (0..=n).filter(|&s| brute_q(&p, s, 0) > 0.0) {
                let probs = conditional_bernoulli_probs(s, &p);
                for _ in 0..500 {
                    let zs = conditional_bernoulli_sample(n, s, &probs, &mut rng);
                    assert_eq!(zs.len(), n);
                    assert_eq!(zs.iter().sum::<usize>(), s, "{p:?} s={s}: {zs:?}");
                    assert!(path_prob(&p, &zs) > 0.0, "{p:?} s={s}: impossible {zs:?}");
                }
            }
        }
    }
}
