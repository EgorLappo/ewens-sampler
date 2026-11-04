use rand::{Rng, RngExt};
use rand_pcg::Pcg64;

use super::ln_add_exp;

// helpers for sampling from the pitman yor distribution given
// values of n, k, and alpha
//
// we do this by choosing a theta, then defining n bernoulli variables
// zeta_j = (theta + x * alpha)/(theta + j), x=0..=s, j=0..(n-1)
// (j is 0-based, so it is the number of customers already seated,
// and x is the number of tables already open)
// then conditioning on zetas summing to s with a recursive calculation of
// new conditional probabilities

// first, the vector of probabilities for zeta
//
// flat, of shape (s+1)*n, so p[x, j] is p[n * x + j]
// only x <= j is reachable (can't have more tables than customers),
// the rest stay at 0 so they are valid probabilities that never get used
// NOTE: needs theta > 0 and 0 <= alpha < 1 for p[x, j] to be in (0, 1] when x <= j
fn py_zetas(n: usize, s: usize, theta: f64, alpha: f64) -> Box<[f64]> {
    assert!(s <= n, "py_zetas: sum > number of variables");
    assert!(
        theta > 0.0 && (0.0..1.0).contains(&alpha),
        "py_zetas: need theta > 0 and 0 <= alpha < 1, got theta={theta}, alpha={alpha}"
    );

    let mut p = vec![0.0; n * (s + 1)];

    for x in 0..=s {
        for j in x..n {
            p[n * x + j] = (theta + (x as f64) * alpha) / (theta + (j as f64));
        }
    }

    p.into_boxed_slice()
}

// bernoulli zetas for the Pitman-Yor sampling
//
// with an extra parameter alpha, the natural q becomes a 3D array
// rather than a 2D one in ewens:
//   q[x,i,j] is the probability that the sum of variables numbered
//   j to n-1 is i given that the sum of variables numbered
//   0 to j-1 is x
// but x+i is the same in every term of the recursion below, so when
// conditioning on the total being s we only ever need the slice x = s-i.
// so we only store that:
//   lq[i, j] = log q[s-i, i, j]
// which has the same (s+1)*n shape as in ewens
fn conditional_bernoulli_log_q(n: usize, s: usize, p: &[f64]) -> Box<[f64]> {
    debug_assert!(
        p.iter().all(|&x| (0.0..=1.0).contains(&x)),
        "p outside [0,1]"
    );
    assert!(s <= n, "conditional_bernoulli: sum > number of variables");
    assert_eq!(
        p.len(),
        n * (s + 1),
        "conditional_bernoulli: p is not (s+1)*n"
    );

    let mut lq = vec![f64::NEG_INFINITY; (s + 1) * n];
    if n == 0 {
        return lq.into_boxed_slice();
    }

    let ln_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let ln_1mp: Vec<f64> = p.iter().map(|x| (1.0 - x).ln()).collect();

    // first, q[s,0,j] can be filled directly without recursion,
    // as this is where all remaining bernoulli trials fail
    //
    // lq[0, j] = sum_(m=j)^(n-1) log (1-p[s,m])
    let mut acc = 0.0;
    for j in (0..n).rev() {
        acc += ln_1mp[n * s + j];
        lq[j] = acc;
    }

    if s == 0 {
        return lq.into_boxed_slice();
    }

    // access helper guarding j = n (past the last variable),
    // where q[s, 0, n] = 1 and q[x, i, n] = 0 for i > 0.
    // this also covers the single bernoulli variable case q[x, 1, n-1] = p[x, n-1]
    #[inline]
    fn at(lq: &[f64], n: usize, i: usize, j: usize) -> f64 {
        if j >= n {
            if i == 0 { 0.0 } else { f64::NEG_INFINITY }
        } else {
            lq[i * n + j]
        }
    }

    // now, the recursion is
    //   q[x, i, j] = p[x, j] q[x+1, i-1, j+1] + (1 - p[x, j]) q[x, i, j+1]
    // which with x = s - i is
    //   lq[i, j] = log( p[s-i, j] e^q[i-1, j+1] + (1 - p[s-i, j]) e^lq[i, j+1] )
    // so we go up the indices of i and down the indices of j.
    // q is zero when i > n-j, so j only goes up to n-i; x > j is unreachable,
    // but p is 0 there so it is harmless and reachable entries never read it
    for i in 1..=s {
        let x = s - i;
        for j in (0..=(n - i)).rev() {
            let a = ln_p[n * x + j] + at(&lq, n, i - 1, j + 1);
            let b = ln_1mp[n * x + j] + at(&lq, n, i, j + 1);
            lq[i * n + j] = ln_add_exp(a, b);
        }
    }

    lq.into_boxed_slice()
}

// actual conditional bernoulli variables are sampled with probs P that
// now depend on the sum of previous samples y through p, but since the total is fixed
// at s, that is the same as depending on the remaining sum s - y, like in ewens:
//
// let P[j, s] be the probability of jth Bernoulli variable zeta_j conditional on the total sum of n being equal to s.
// then,
// P[j, s] = p[y, j] q[y + 1, s - y - 1, j+1] / q[y, s - y, j]
//   where y = sum_(m=1)^(j-1) zeta_m (y is 1-based math notation), and P[1, s] is the case y = 0
//
// with i = s - y remaining and lq[i, j] = log q[s-i, i, j] this is
//   P = p[s-i, j] e^lq[i-1, j+1] / e^lq[i, j]
fn conditional_bernoulli_probs(n: usize, s: usize, theta: f64, alpha: f64) -> Box<[f64]> {
    if s == 0 {
        return Vec::new().into_boxed_slice();
    }

    let p = py_zetas(n, s, theta, alpha);
    let ln_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let lq = conditional_bernoulli_log_q(n, s, &p);

    // here we fill the probabilities used in conditional sampling, also
    // of shape s*n like in ewens, so with i remaining it is probs[(i-1)*n + j]
    // and the shared conditional_bernoulli_sample_into works unchanged
    let mut probs = vec![0.0f64; s * n];

    // fill probs
    for i in 1..=s {
        let x = s - i;
        // i <= n-j, as there have to be enough variables left
        for j in 0..=(n - i) {
            let ln_denum = lq[i * n + j];
            // past the last variable q is 1 for nothing remaining, and j = n-1 only has i = 1
            let ln_num = ln_p[n * x + j]
                + if j + 1 == n {
                    0.0
                } else {
                    lq[(i - 1) * n + j + 1]
                };
            probs[(i - 1) * n + j] = (ln_num - ln_denum).exp().clamp(0.0, 1.0);
        }
    }

    probs.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    // all 0/1 sequences of length len
    fn sequences(len: usize) -> impl Iterator<Item = Vec<usize>> {
        (0..1usize << len).map(move |b| (0..len).map(|m| (b >> m) & 1).collect())
    }

    // probability of a zeta sequence for variables j.. starting from x open tables
    fn path_prob(n: usize, p: &[f64], x: usize, j: usize, zs: &[usize]) -> f64 {
        let mut x = x;
        let mut prob = 1.0;
        for (m, &z) in (j..).zip(zs) {
            let pm = p[n * x + m];
            prob *= if z == 1 { pm } else { 1.0 - pm };
            x += z;
        }
        prob
    }

    // lq[i, j] against q[s-i, i, j] by enumerating all suffixes
    #[test]
    fn log_q_matches_brute_force() {
        for (n, s, theta, alpha) in [
            (6, 3, 1.3, 0.4),
            (7, 4, 0.2, 0.9),
            (5, 5, 2.0, 0.5),
            (6, 1, 1.0, 0.0),
            (4, 0, 1.0, 0.3),
        ] {
            let p = py_zetas(n, s, theta, alpha);
            let lq = conditional_bernoulli_log_q(n, s, &p);
            for i in 0..=s {
                for j in 0..n {
                    let q: f64 = sequences(n - j)
                        .filter(|zs| zs.iter().sum::<usize>() == i)
                        .map(|zs| path_prob(n, &p, s - i, j, &zs))
                        .sum();
                    let got = lq[i * n + j].exp();
                    assert!(
                        (got - q).abs() < 1e-12,
                        "n={n} s={s} i={i} j={j}: {got} vs {q}"
                    );
                }
            }
        }
    }

    // q[x, i, j] by enumerating all suffixes
    fn brute_q(n: usize, p: &[f64], x: usize, i: usize, j: usize) -> f64 {
        sequences(n - j)
            .filter(|zs| zs.iter().sum::<usize>() == i)
            .map(|zs| path_prob(n, p, x, j, &zs))
            .sum()
    }

    // probs[i-1, j] against p[x, j] q[x+1, i-1, j+1] / q[x, i, j] with x = s-i,
    // everywhere the sampler can get to
    #[test]
    fn probs_match_brute_force() {
        for (n, s, theta, alpha) in [
            (6, 3, 1.3, 0.4),
            (7, 4, 0.2, 0.9),
            (5, 5, 2.0, 0.5),
            (6, 1, 1.0, 0.0),
        ] {
            let p = py_zetas(n, s, theta, alpha);
            let probs = conditional_bernoulli_probs(n, s, theta, alpha);
            for i in 1..=s {
                let x = s - i;
                // x <= j, as there can't be more tables than customers
                for j in x..=(n - i) {
                    let want = p[n * x + j] * brute_q(n, &p, x + 1, i - 1, j + 1)
                        / brute_q(n, &p, x, i, j);
                    let got = probs[(i - 1) * n + j];
                    assert!(
                        (got - want).abs() < 1e-12,
                        "n={n} s={s} i={i} j={j}: {got} vs {want}"
                    );
                }
            }
        }
    }

    // probability that the shared conditional_bernoulli_sample_into draws zs from probs,
    // following its shortcuts: nothing remaining means the rest are 0, and as many remaining
    // as variables left means the rest are 1
    fn sampler_prob(n: usize, s: usize, probs: &[f64], zs: &[usize]) -> f64 {
        let mut rem = s;
        let mut prob = 1.0;
        for (j, &z) in zs.iter().enumerate() {
            let pj = if rem == 0 {
                0.0
            } else if rem == n - j {
                1.0
            } else {
                probs[(rem - 1) * n + j]
            };
            prob *= if z == 1 { pj } else { 1.0 - pj };
            if prob == 0.0 {
                return 0.0;
            }
            rem -= z;
        }
        prob
    }

    // drawing with probs should give every sequence its probability under the zetas
    // conditioned on summing to s, and nothing else
    #[test]
    fn probs_give_conditional_law() {
        for (n, s, alpha) in [(7, 3, 0.6), (6, 1, 0.3), (6, 6, 0.5), (8, 5, 0.0)] {
            for theta in [0.3, 25.0] {
                let p = py_zetas(n, s, theta, alpha);
                let probs = conditional_bernoulli_probs(n, s, theta, alpha);
                let total = brute_q(n, &p, 0, s, 0);
                for zs in sequences(n) {
                    let want = if zs.iter().sum::<usize>() == s {
                        path_prob(n, &p, 0, 0, &zs) / total
                    } else {
                        0.0
                    };
                    let got = sampler_prob(n, s, &probs, &zs);
                    assert!(
                        (got - want).abs() < 1e-12,
                        "n={n} s={s} alpha={alpha} theta={theta} {zs:?}: {got} vs {want}"
                    );
                }
            }
        }
    }

    // the law of the zetas given their sum should not depend on theta
    #[test]
    fn conditional_law_free_of_theta() {
        let (n, s, alpha) = (7, 3, 0.6);
        let law = |theta: f64| -> Vec<f64> {
            let p = py_zetas(n, s, theta, alpha);
            let lq = conditional_bernoulli_log_q(n, s, &p);
            let total = lq[s * n].exp();
            sequences(n)
                .filter(|zs| zs.iter().sum::<usize>() == s)
                .map(|zs| path_prob(n, &p, 0, 0, &zs) / total)
                .collect()
        };
        let (a, b) = (law(0.3), law(25.0));
        assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }
}
