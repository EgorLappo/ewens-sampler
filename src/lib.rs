use color_eyre::{Result, eyre::WrapErr, eyre::bail};
use itertools::Itertools;

pub mod sampler;
pub mod stats;

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
