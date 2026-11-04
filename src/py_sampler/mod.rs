use rand::{Rng, RngExt};
use rand_pcg::Pcg64;

pub trait PYSampler {
    fn sample(&self, rng: &mut Pcg64) -> Vec<u16>;
}
