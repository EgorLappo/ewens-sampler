const BISECT_ITERS: usize = 100;

/// find MLE for theta given alpha by bisecting for critical point
/// `inits` are for conditional inference given initial configuration
pub fn theta_mle(n: usize, k: usize, alpha: f64, inits: Option<(usize, usize)>) -> f64 {
    let (n0, k0) = inits.unwrap_or((0, 0));
    assert!(
        1 <= k && k <= n && alpha <= 1.,
        "theta_mle: need 1 <= k <= n and alpha <= 1, got n={n}, k={k}, alpha={alpha}"
    );
    // otherwise k - k0 underflows, or with no samples left to fit the bracket search never ends
    assert!(
        k0 <= k && (k0 == k || n0 < n),
        "theta_mle: initial configuration with n0={n0}, k0={k0} does not fit n={n}, k={k}"
    );

    // the limits have nothing to fit: at alpha = -inf theta = k |alpha| is infinite too,
    // and at alpha = 1 the configuration is (n - k + 1, 1, ..., 1) whatever theta is
    if alpha == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    if alpha == 1. {
        return f64::NAN;
    }

    // the lowest theta allowed (written as 0 - alpha so that ewens gets 0 and not -0)
    let lowest = if alpha < 0. {
        k as f64 * -alpha
    } else {
        0. - alpha
    };

    // the first sample opens a table w.p. theta / theta = 1, so it says nothing about theta
    let (n0, k0) = (n0.max(1), k0.max(1));

    if k - k0 == 0 {
        // give lowest one on no new variants, like theta = 4 N mu = 0 in ewens
        return lowest;
    }

    // score is just d log L / d theta = 0 rewritten as a num-denum difference:
    // log L = sum_(i=1)^(k-1) log(theta + i alpha) - sum_(j=1)^(n-1) log(theta + j) + const
    // score = sum_(i=1)^(k-1) 1 / (theta + i alpha) - sum_(j=1)^(n-1) 1 / (theta + j)
    let score = |theta: f64| -> f64 {
        (k0..k)
            .map(|i| 1. / (theta + i as f64 * alpha))
            .sum::<f64>()
            - (n0..n).map(|i| 1. / (theta + i as f64)).sum::<f64>()
    };

    // it can peak there with new alleles too, like for alpha < 0 far from 0,
    // or with an initial configuration that already has a few alleles
    if score(lowest) <= 0. {
        return lowest;
    }

    // proceed by bisection, on theta - lowest

    // find the bracket that must contain the root by IVT
    let (mut lo, mut hi) = (1., 1.);

    while score(lowest + lo) < 0. {
        lo *= 0.5;
    }
    // with a new allele for every new sample there is no root, as the likelihood keeps growing,
    // and like with expected k in ewens this stops at a huge theta,
    // once the sums are equal in floating point
    while score(lowest + hi) > 0. {
        hi *= 2.;
    }

    // bisect in log space
    let (mut lo, mut hi) = (lo.ln(), hi.ln());
    for _ in 0..BISECT_ITERS {
        let mid = (hi + lo) * 0.5;
        if score(lowest + mid.exp()) > 0. {
            // the likelihood still grows
            lo = mid;
        } else {
            // we are past the peak
            hi = mid;
        }
    }

    // report the final mid
    let theta = lowest + ((hi + lo) * 0.5).exp();

    // for alpha < 0 pitman-yor needs theta = m |alpha| with a whole number m of types.
    // the likelihood has a single peak there (with all theta + i alpha below all theta + i,
    // the first sum in the score falls faster, relative to its size, than the second),
    // so the best m is one of the two whole numbers around theta / |alpha|, both at least k.
    // they can tie, like 8 and 9 types for alpha = -1, n = 10, k = 5, and then we take the smaller
    if alpha < 0. {
        let log_lik = |theta: f64| -> f64 {
            (k0..k)
                .map(|i| (theta + i as f64 * alpha).ln())
                .sum::<f64>()
                - (n0..n).map(|i| (theta + i as f64).ln()).sum::<f64>()
        };
        let m = (theta / -alpha).max(k as f64);
        let (below, above) = (m.floor() * -alpha, m.ceil() * -alpha);
        return if log_lik(below) >= log_lik(above) {
            below
        } else {
            above
        };
    }

    theta
}

#[cfg(test)]
mod tests {
    use super::*;

    // probability of k alleles after n samples given k0 after n0, by following the number
    // of tables through the restaurant instead of the closed form.
    // tables past k never come back, so they are dropped
    fn prob_k(theta: f64, alpha: f64, n: usize, k: usize, n0: usize, k0: usize) -> f64 {
        let mut p = vec![0.; k + 2];
        p[k0] = 1.;
        for j in n0..n {
            let mut next = vec![0.; k + 2];
            for x in k0..=k {
                let new = (theta + x as f64 * alpha) / (theta + j as f64);
                next[x + 1] += p[x] * new;
                next[x] += p[x] * (1. - new);
            }
            p = next;
        }
        p[k]
    }

    // expected number of alleles, from the same restaurant
    fn expected_k(theta: f64, alpha: f64, n: usize) -> f64 {
        (1..n).fold(1., |m, j| m + (theta + alpha * m) / (theta + j as f64))
    }

    // the mle should beat a slightly smaller and a slightly larger theta,
    // or only the larger one when it is the lowest theta allowed.
    // for alpha < 0 it is a whole number of types, and those are one type more and one less,
    // which can tie with it
    fn assert_peaks(n: usize, k: usize, alpha: f64, inits: Option<(usize, usize)>) -> f64 {
        let (n0, k0) = inits.unwrap_or((0, 0));
        let theta = theta_mle(n, k, alpha, inits);
        let lowest = if alpha < 0. {
            k as f64 * -alpha
        } else {
            -alpha
        };
        let p = |t: f64| prob_k(t, alpha, n, k, n0, k0);
        let label = format!("n={n} k={k} alpha={alpha} inits={inits:?}: theta={theta}");
        let (step, tie) = if alpha < 0. {
            let m = theta / -alpha;
            assert!(
                (m - m.round()).abs() < 1e-9 * m && m >= k as f64,
                "{label}: {m} types"
            );
            (-alpha, 1e-12)
        } else {
            (1e-3 * (theta - lowest).max(1e-3), 0.)
        };
        let beats = |t: f64| p(theta) > p(t) * (1. - tie);
        assert!(beats(theta + step), "{label}");
        if theta > lowest {
            assert!(beats(theta - step), "{label}");
        }
        theta
    }

    // alpha = -1 has L(m + 1) / L(m) = m (m + 1) / ((m - k + 1)(m + n)), which is 1 at m = 8
    // for n = 10, k = 5
    #[test]
    fn mle_takes_fewer_types_on_a_tie() {
        assert_eq!(assert_peaks(10, 5, -1., None), 8.);
        assert!((prob_k(8., -1., 10, 5, 0, 0) / prob_k(9., -1., 10, 5, 0, 0) - 1.).abs() < 1e-12);
    }

    #[test]
    fn mle_maximizes_probability_of_k() {
        for alpha in [0., 0.5, 0.9, -1., -10.] {
            for n in [10, 265] {
                for k in [2, n / 2, n - 1] {
                    assert_peaks(n, k, alpha, None);
                }
            }
        }
    }

    #[test]
    fn mle_maximizes_probability_of_k_after_initial_configuration() {
        for alpha in [0., 0.5, -1.] {
            for (n, k, n0, k0) in [(20, 6, 5, 2), (265, 81, 100, 30), (50, 4, 3, 3)] {
                assert_peaks(n, k, alpha, Some((n0, k0)));
            }
        }
    }

    // only in ewens the mle is where expected k is k
    #[test]
    fn mle_fits_expected_k_only_for_ewens() {
        for (n, k) in [(10, 4), (265, 81)] {
            let ek = expected_k(theta_mle(n, k, 0., None), 0., n);
            assert!(
                (ek - k as f64).abs() < 1e-9 * k as f64,
                "n={n} k={k}: E[K] = {ek}"
            );
        }
        // k = 81 is the most likely for a theta that expects about 130 alleles
        let ek = expected_k(theta_mle(265, 81, 0.9, None), 0.9, 265);
        assert!(ek > 120., "E[K] = {ek}");
    }

    // bad initial configurations should stop with a message, not hang or underflow
    #[test]
    #[should_panic(expected = "does not fit")]
    fn mle_rejects_initial_configuration_covering_all_samples() {
        theta_mle(7, 3, 0.5, Some((7, 2)));
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn mle_rejects_initial_configuration_with_more_than_k_alleles() {
        theta_mle(10, 2, 0.5, Some((3, 3)));
    }

    #[test]
    fn mle_is_lowest_theta_without_new_alleles() {
        assert_eq!(theta_mle(20, 4, 0.5, Some((10, 4))), -0.5);
        assert_eq!(theta_mle(20, 1, 0.5, None), -0.5);
        assert_eq!(theta_mle(20, 4, -1., Some((10, 4))), 4.);
        // and ewens gets 0, as in its own theta module
        let theta = theta_mle(20, 1, 0., None);
        assert!(theta == 0. && theta.is_sign_positive(), "{theta}");
    }

    // and it can be the lowest theta with new alleles too
    #[test]
    fn mle_can_be_lowest_theta() {
        assert_eq!(assert_peaks(100, 5, -10., None), 50.);
        assert_eq!(assert_peaks(1000, 6, 0.9, Some((10, 5))), -0.9);
    }

    #[test]
    fn mle_is_huge_when_every_sample_is_a_new_allele() {
        for alpha in [0., 0.5, -1.] {
            let theta = theta_mle(30, 30, alpha, None);
            assert!(theta > 1e12, "alpha={alpha}: {theta}");
        }
    }

    // so it is a model the free sampler takes, e.g. to simulate from the fitted one
    #[test]
    fn mle_is_a_whole_number_of_types_for_negative_alpha() {
        for (n, k, alpha) in [
            (265, 81, -1.),
            (265, 81, -0.3),
            (50, 25, -2.5),
            (100, 5, -10.),
        ] {
            let theta = assert_peaks(n, k, alpha, None);
            crate::samplers::pitman_yor::PYSampler::new(theta, alpha, n).unwrap();
        }
        // the peak is at 115.7 types
        let theta = theta_mle(265, 81, -1., None);
        assert!(theta == 115. || theta == 116., "{theta}");
    }

    #[test]
    fn mle_increases_with_k() {
        for alpha in [0., 0.5, -1.] {
            let thetas: Vec<f64> = (1..=30).map(|k| theta_mle(30, k, alpha, None)).collect();
            // whole numbers of types can repeat
            let increases = |w: &[f64]| w[0] < w[1] || (alpha < 0. && w[0] == w[1]);
            assert!(
                thetas.windows(2).all(increases),
                "alpha={alpha}: {thetas:?}"
            );
        }
    }

    #[test]
    fn limits_have_nothing_to_fit() {
        assert_eq!(theta_mle(20, 4, f64::NEG_INFINITY, None), f64::INFINITY);
        assert!(theta_mle(20, 4, 1., None).is_nan());
    }
}
