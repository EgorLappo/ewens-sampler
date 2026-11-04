const BISECT_ITERS: usize = 100;

/// find MLE for theta by fitting expected k given theta
/// (which should be the same as the MLE estimate)
/// `inits` are for conditional inference given initial configuration
pub fn theta_mle(n: usize, k: usize, inits: Option<(usize, usize)>) -> f64 {
    let (n0, k0) = inits.unwrap_or((0, 0));

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
