use super::{Sampler, conditional_bernoulli_sample, ln_add_exp};
use color_eyre::eyre::{Result, bail};
use rand::{Rng, RngExt};

pub mod theta;

// helpers for sampling from the pitman yor distribution given
// values of n, k, and alpha
//
// we do this by choosing a theta, then like crp_bernoulli_p in ewens, defining n bernoulli
// variables only for the customers after the first, who always opens a table.
// with j 0-based, variable j is for the customer who finds j + 1 customers already seated
// at x + 1 tables, where x is the sum of the variables before it:
// zeta_j = (theta + (x + 1) * alpha)/(theta + j + 1), x=0..=s, j=0..(n-1)
// then conditioning on zetas summing to s = k - 1 with a recursive calculation of
// new conditional probabilities

// first, the vector of probabilities for zeta
//
// n is the number of variables, so one less than the number of customers
// (crp_bernoulli_p in ewens takes the number of customers and iterates over 2..=n instead).
// this is because here we have two-dimensional array of probabilities
//
// we use it flattened, with shape (s+1)*n, so p[x, j] is p[n * x + j]
// only x <= j is reachable (the j variables before j can't sum to more),
// the rest stay at 0 so they are valid probabilities that never get used
//
// NOTE: for p[x, j] to be in [0, 1] when x <= j we need alpha < 1 and theta + alpha > 0,
// and for alpha < 0 also theta + k alpha >= 0 with k = s + 1. that is the finite dirichlet case
// theta = m |alpha| with at least m >= k types, and conditioning on k means we never get further.
// alpha = -inf is handled as a limit, and alpha = 1 can't be reached this way (see PYSamplerK)
fn py_zetas(n: usize, s: usize, theta: f64, alpha: f64) -> Box<[f64]> {
    assert!(s <= n, "py_zetas: sum > number of variables");

    let mut p = vec![0.0; n * (s + 1)];

    if alpha == f64::NEG_INFINITY {
        // with theta = k |alpha| and alpha -> -inf, every customer picks one of k cells uniformly
        // and opens a new table when the cell is empty, so it no longer depends on theta or j
        for x in 0..=s {
            for j in x..n {
                p[n * x + j] = ((s - x) as f64) / ((s + 1) as f64);
            }
        }
        return p.into_boxed_slice();
    }

    assert!(
        alpha < 1.0 && theta + alpha > 0.0 && theta + ((s + 1) as f64) * alpha >= 0.0,
        "py_zetas: need alpha < 1, theta > -alpha and theta >= -k alpha (k = {}), got theta={theta}, alpha={alpha}",
        s + 1
    );

    for x in 0..=s {
        for j in x..n {
            p[n * x + j] = (theta + ((x + 1) as f64) * alpha) / (theta + ((j + 1) as f64));
        }
    }

    p.into_boxed_slice()
}

/// a theta that works for any alpha given k: theta cancels once we condition on k,
/// but the probabilities in py_zetas need theta > -alpha and theta >= -k alpha
pub fn default_theta(k: usize, alpha: f64) -> f64 {
    1.0 + k as f64 * (-alpha).max(0.0)
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

// the chinese restaurant shared by both samplers: n customers end up at k tables, with zetas
// for the customers after the first saying who opens a new one, and everyone else joining
// a table of size n_c with probability proportional to n_c - alpha
fn seat<R: Rng>(n: usize, k: usize, alpha: f64, zetas: &[usize], rng: &mut R) -> Vec<u16> {
    // in the chinese restaurant process, the configuration is built
    // sample-by sample

    // assignment of samples to cycles
    let mut assignment = vec![0; n];
    // first element must go into the first cycle
    assignment[0] = 0;

    // variables to keep track of progress
    let mut cycle = 0;

    let mut configuration = vec![0; k];
    configuration[cycle] += 1;

    for (samples_assigned, s) in (1..).zip(zetas.iter()) {
        if *s == 0 {
            // if s == 0, we must place the sample in an existing cycle c,
            // but now with probability proportional to n_c - alpha.
            // a randomly chosen previous sample is in cycle c w.p. n_c / samples_assigned
            let cycle_choice = if alpha >= 0.0 {
                // so for alpha >= 0 we keep its cycle w.p. (n_c - alpha) / n_c and
                // otherwise choose again, which works out at least a (1 - alpha) fraction of the time
                loop {
                    let choice = rng.random_range(..samples_assigned);
                    let c = assignment[choice];
                    let size = configuration[c] as f64;
                    if rng.random::<f64>() * size < size - alpha {
                        break c;
                    }
                }
            } else {
                // for alpha < 0, n_c - alpha = n_c + |alpha| is more than n_c, so instead mix
                // copying a randomly chosen previous sample (total weight samples_assigned)
                // with choosing one of the cycle + 1 cycles uniformly (total weight (cycle + 1) |alpha|).
                // at alpha = -inf this is always uniform
                let p_copy = samples_assigned as f64
                    / (samples_assigned as f64 - (cycle + 1) as f64 * alpha);
                if rng.random::<f64>() < p_copy {
                    assignment[rng.random_range(..samples_assigned)]
                } else {
                    rng.random_range(..=cycle)
                }
            };
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

// free k: without conditioning, the zetas are the restaurant's own and are drawn one at a time,
// then the customers are seated as in PYSamplerK.
//
// feller coupling doesn't carry over. it builds the configuration one allele at a time,
// and in ewens the current allele stops growing w.p. theta / (theta + r) with r samples left,
// so the stops are independent bernoullis and the counts are the spacings between them.
// under pitman-yor the allele of the first sample still has
// 1 + betabinomial(n - 1; 1 - alpha, theta + alpha) copies, with pitman-yor (alpha, theta + alpha)
// for the rest, but its stopping probabilities only come out in closed form at alpha = 0
#[derive(Debug, Clone)]
pub struct PYSampler {
    pub theta: f64,
    pub alpha: f64,
    pub n: usize,
    // the most tables there can be: m for alpha < 0, where theta = m |alpha|, and n otherwise
    max_k: usize,
}

impl PYSampler {
    pub fn new(theta: f64, alpha: f64, n: usize) -> Result<Self> {
        if !(n >= 1 && alpha > f64::NEG_INFINITY && alpha <= 1.0) {
            bail!("invalid parameters n={n}, alpha={alpha}. requires n >= 1 and -inf < alpha <= 1");
        }
        let max_k = if alpha < 0.0 {
            // the finite dirichlet case with m types, where zeta is 0 once all m are seen.
            // any other theta would give negative probabilities
            let m = theta / -alpha;
            if !(m.is_finite() && m.round() >= 1.0 && (m - m.round()).abs() <= 1e-9 * m) {
                bail!(
                    "invalid parameters theta={theta}, alpha={alpha}. requires theta = m |alpha| for a whole number m >= 1 of types"
                );
            }
            m.round() as usize
        } else {
            if !(theta.is_finite() && theta + alpha > 0.0) {
                bail!("invalid parameters theta={theta}, alpha={alpha}. requires theta > -alpha");
            }
            n
        };
        Ok(Self {
            theta,
            alpha,
            n,
            max_k,
        })
    }
}

impl<R: Rng> Sampler<R> for PYSampler {
    fn sample(&self, rng: &mut R) -> Vec<u16> {
        // the zetas of py_zetas: variable j is for the customer who finds j + 1 customers
        // already seated at x + 1 tables, where x is the sum of the variables before it
        let mut x = 0;
        let zetas: Vec<usize> = (0..self.n - 1)
            .map(|j| {
                let p =
                    (self.theta + ((x + 1) as f64) * self.alpha) / (self.theta + ((j + 1) as f64));
                let z = (x + 1 < self.max_k && rng.random::<f64>() < p) as usize;
                x += z;
                z
            })
            .collect();

        seat(self.n, x + 1, self.alpha, &zetas, rng)
    }

    fn theta(&self) -> f64 {
        self.theta
    }
}

#[derive(Debug, Clone)]
pub struct PYSamplerK {
    pub theta: f64,
    pub alpha: f64,
    pub n: usize,
    pub k: usize,
    n_vars: usize,
    probs: Box<[f64]>,
}

impl PYSamplerK {
    pub fn new(theta: f64, alpha: f64, n: usize, k: usize) -> Result<Self> {
        if k < 1 || k > n {
            bail!("invalid parameters n={n}, k={k}. requires 1 <= k <= n");
        }
        if alpha.is_nan() || alpha > 1.0 {
            bail!("invalid parameters alpha={alpha}. requires alpha <= 1");
        }
        // theta only enters through the zetas, so it can be anything with one allele,
        // or at alpha = 1 or -inf (like the theta = -alpha that theta_mle gives for k = 1).
        // otherwise this is what py_zetas needs
        if k > 1
            && alpha > f64::NEG_INFINITY
            && alpha < 1.0
            && !(theta.is_finite() && theta + alpha > 0.0 && theta + k as f64 * alpha >= 0.0)
        {
            bail!(
                "invalid parameters theta={theta}, alpha={alpha}, k={k}. requires theta > -alpha and theta >= k |alpha|"
            );
        }

        // zetas for customers 2..=n, like crp_bernoulli_p in ewens
        // (first one is forced to be 1)
        let n_vars = n - 1;
        // conditioned ones, except at alpha = 1, where the most unbalanced configuration
        // is the only possible one (nothing to sample)
        let probs = if alpha == 1.0 {
            Vec::new().into_boxed_slice()
        } else {
            conditional_bernoulli_probs(n_vars, k - 1, theta, alpha)
        };
        Ok(Self {
            theta,
            alpha,
            n,
            k,
            n_vars,
            probs,
        })
    }
}

impl<R: Rng> Sampler<R> for PYSamplerK {
    fn sample(&self, rng: &mut R) -> Vec<u16> {
        // as alpha -> 1 every allele with more than one copy gets a factor (1 - alpha) -> 0,
        // so all the mass goes to the configuration with only one of them, (n - k + 1, 1, ..., 1)
        if self.alpha == 1.0 {
            let mut configuration = vec![1; self.k];
            configuration[0] = (self.n - self.k + 1) as u16;
            return configuration;
        }

        let zetas = conditional_bernoulli_sample(self.n_vars, self.k - 1, &self.probs, rng);
        seat(self.n, self.k, self.alpha, &zetas, rng)
    }

    fn theta(&self) -> f64 {
        self.theta
    }
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
            // negative alpha needs theta >= k |alpha|
            (6, 3, 5.0, -1.0),
            (7, 4, 11.0, -2.0),
            (6, 3, f64::INFINITY, f64::NEG_INFINITY),
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
            (6, 3, 5.0, -1.0),
            (6, 3, f64::INFINITY, f64::NEG_INFINITY),
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
        // s = 0 is k = 1, everyone at the first table
        for (n, s, alpha) in [
            (7, 3, 0.6),
            (6, 1, 0.3),
            (6, 6, 0.5),
            (8, 5, 0.0),
            (5, 0, 0.4),
            (7, 3, -1.0),
            (6, 2, -2.5),
            (6, 3, f64::NEG_INFINITY),
        ] {
            let theta0 = default_theta(s + 1, alpha);
            for theta in [theta0, theta0 + 25.0] {
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
        let (n, s) = (7, 3);
        let law = |theta: f64, alpha: f64| -> Vec<f64> {
            let p = py_zetas(n, s, theta, alpha);
            let lq = conditional_bernoulli_log_q(n, s, &p);
            let total = lq[s * n].exp();
            sequences(n)
                .filter(|zs| zs.iter().sum::<usize>() == s)
                .map(|zs| path_prob(n, &p, 0, 0, &zs) / total)
                .collect()
        };
        // theta only has to be above -alpha, and for alpha < 0 at least k |alpha|
        for (alpha, theta_a, theta_b) in [(0.6, -0.3, 25.0), (-0.7, 2.8, 60.0)] {
            let (a, b) = (law(theta_a, alpha), law(theta_b, alpha));
            assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-12, "alpha={alpha}: {x} vs {y}");
            }
        }
    }

    // alpha = -inf is the limit of very negative alpha
    #[test]
    fn negative_infinity_is_the_limit() {
        let (n, s) = (7, 3);
        let law = |alpha: f64| -> Vec<f64> {
            let probs = conditional_bernoulli_probs(n, s, default_theta(s + 1, alpha), alpha);
            sequences(n)
                .map(|zs| sampler_prob(n, s, &probs, &zs))
                .collect()
        };
        let (a, b) = (law(f64::NEG_INFINITY), law(-1e6));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }
}
