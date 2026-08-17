## A Ewens distribution utility

### Installation


### Examples


### Help messages:

```
> ./ewens-sampler --help
a ewens distribution utility

Usage: ewinfer [OPTIONS]

Options:
  -n <N>
          number of sampled alleles
  -k <K>
          number of alleles (for sampling with fixed k)
  -s, --samples <REPLICATES>
          number of sampled configurations to generate [default: 1000]
      --seed <SEED>
          random seed [default: 231]
  -t, --theta <THETA>
          value of theta (doesn't affect the results when k is fixed) [default: 1]
  -i, --initial-configuration <INITIAL_CONFIGURATION>
          initial configuration to start sampling from, uses conditional CRP sampler
  -b, --bias <BIAS>
          frequency-dependent bias factor of restaurant tables (experimental)
      --test <TEST>
          configuration to run the exact test on; a space-separated list of unsigned integers; must have `k` elements with values summing to `n`
  -j, --json
          output test results in JSON rather than human-readable form
  -q, --quiet
          quiet run (no progress bar)
  -h, --help
          Print help
  -V, --version
          Print version
```
