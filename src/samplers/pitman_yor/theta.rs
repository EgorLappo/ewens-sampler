const BISECT_ITERS: usize = 100;

// MLE for theta given k, alpha in the Pitman-Yor case
// is found by fitting expected k:
// Ek ~ ( prod_0^{k-1} (theta + i alpha) ) / theta_{(n)}
// where theta_{(n)} = theta * (theta + 1) ...
//
// ofc, for conditional we start with n0, k0 as in ewens module
//
// NOTE: this should work for alpha>=0, and for alpha < 0
// we just use theta = 1 + k|alpha| as a default safe choice
// (but have to test whether other values give better/stabler numerics)
pub fn theta_mle(n: usize, k: usize, alpha: f64, inits: Option<(usize, usize)>) -> f64 {
    let (n0, k0) = inits.unwrap_or((0, 0));
    if alpha < 0. {
        return 1. + (k as f64) * (-alpha);
    }

    1.
}
