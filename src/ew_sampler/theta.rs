const BISECT_ITERS: usize = 100;

/// find MLE for theta by fitting expected k given theta
/// (which should be the same as the MLE estimate)
/// `inits` are for conditional inference given initial configuration
pub fn theta_mle(n: usize, k: usize, inits: Option<(usize, usize)>) -> f64 {
    let (n0, k0) = inits.unwrap_or((0, 0));
    // otherwise k - k0 underflows, or with no samples left to fit the bracket search never ends
    assert!(
        k0 <= k && (k0 == k || n0 < n),
        "theta_mle: initial configuration with n0={n0}, k0={k0} does not fit n={n}, k={k}"
    );

    if k - k0 == 0 {
        // return 0 as no mutations => theta = 4 N mu = 0
        return 0.0;
    }

    let kf = (k - k0) as f64;
    let exp_k = |theta: f64| -> f64 { (n0..n).map(|i| theta / (theta + i as f64)).sum() };

    // proceed by bisection

    // find the bracket that must contain the root by IVT
    let (mut lo, mut hi) = (1., 1.);

    while exp_k(lo) > kf {
        lo *= 0.5;
    }
    while exp_k(hi) < kf {
        hi *= 2.;
    }

    // bisect in log space
    let (mut lo, mut hi) = (lo.ln(), hi.ln());
    // TODO? early stopping, convergence?
    //   (we have a very good function with a nice range of typical values in data, so maybe no)
    for i in 0..BISECT_ITERS {
        let mid = (hi + lo) * 0.5;
        if exp_k(mid.exp()) < kf {
            // we undershoot
            lo = mid;
        } else {
            // we overshoot
            hi = mid;
        }
    }

    // report the final mid
    ((hi + lo) * 0.5).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // expected number of alleles among samples n0+1..n
    fn expected_k(theta: f64, n: usize, n0: usize) -> f64 {
        (n0..n).map(|i| theta / (theta + i as f64)).sum()
    }

    #[test]
    fn mle_fits_expected_k() {
        for n in [2, 10, 265] {
            for k in [1, 2, n / 2, n - 1, n] {
                let theta = theta_mle(n, k, None);
                let ek = expected_k(theta, n, 0);
                assert!(
                    (ek - k as f64).abs() < 1e-9 * k as f64,
                    "n={n} k={k}: E[K]({theta}) = {ek}"
                );
            }
        }
    }

    #[test]
    fn mle_fits_expected_k_after_initial_configuration() {
        for (n, k, n0, k0) in [(20, 6, 5, 2), (265, 81, 100, 30), (50, 4, 3, 3)] {
            let theta = theta_mle(n, k, Some((n0, k0)));
            let ek = expected_k(theta, n, n0);
            assert!(
                (ek - (k - k0) as f64).abs() < 1e-9 * k as f64,
                "n={n} k={k} n0={n0} k0={k0}: E[K]({theta}) = {ek}"
            );
        }
    }

    // bad initial configurations should stop with a message, not hang or underflow
    #[test]
    #[should_panic(expected = "does not fit")]
    fn mle_rejects_initial_configuration_covering_all_samples() {
        theta_mle(7, 3, Some((7, 2)));
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn mle_rejects_initial_configuration_with_more_than_k_alleles() {
        theta_mle(10, 2, Some((3, 3)));
    }

    #[test]
    fn mle_is_zero_without_new_alleles() {
        assert_eq!(theta_mle(20, 4, Some((10, 4))), 0.0);
    }

    #[test]
    fn mle_increases_with_k() {
        let thetas: Vec<f64> = (1..=30).map(|k| theta_mle(30, k, None)).collect();
        assert!(thetas.windows(2).all(|w| w[0] < w[1]), "{thetas:?}");
    }
}
