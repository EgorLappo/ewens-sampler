use color_eyre::{Result, eyre::WrapErr, eyre::bail};
use itertools::Itertools;
use rand::Rng;

pub mod ew_sampler;
pub mod py_sampler;
pub mod stats;

// trait for samplers
pub trait Sampler<R: Rng> {
    fn sample(&self, rng: &mut R) -> Vec<u16>;
    fn theta(&self) -> f64;
}

// newtype for configurations
#[derive(Clone, Debug)]
pub struct Configuration(Vec<u16>);

impl Configuration {
    pub fn n(&self) -> usize {
        self.0.iter().map(|&c| c as usize).sum()
    }
    pub fn k(&self) -> usize {
        self.0.len()
    }
    pub fn parse(s: &str) -> Result<Self> {
        let mut counts = s
            .split_whitespace()
            .map(|t| t.parse::<u16>().wrap_err_with(|| format!("entry {t:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        if counts.is_empty() {
            bail!("Provided configuration is empty");
        }
        if counts.contains(&0) {
            bail!("allele counts must be >= 1");
        }

        // sort the input configurations
        counts.sort_unstable();

        Ok(Self(counts))
    }
}

impl std::str::FromStr for Configuration {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).map_err(|e| format!("{e:#}")) // {:#} keeps the chain
    }
}

impl std::fmt::Display for Configuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().format(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sorts_counts() {
        let c: Configuration = " 3 1  2 ".parse().unwrap();
        assert_eq!(c.0, [1, 2, 3]);
        assert_eq!((c.n(), c.k()), (6, 3));
        assert_eq!(c.to_string(), "1 2 3");
    }

    #[test]
    fn parse_rejects_bad_input() {
        for s in ["", "   ", "1 0 2", "1 x", "-1", "70000"] {
            assert!(Configuration::parse(s).is_err(), "{s:?} parsed");
        }
    }
}
