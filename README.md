## A Ewens distribution utility

Help messages:

```
> ./ewinfer --help
a ewens distribution utility

Usage: ewinfer <COMMAND>

Commands:
  sample  
  test    
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

```
> ./ewinfer sample --help
Usage: ewinfer sample [OPTIONS] -n <N> [SAMPLES]

Arguments:
  [SAMPLES]
          number of sampled configurations to generate
          
          [default: 100]

Options:
  -n <N>
          number of samples

  -k <K>
          number of alleles (for sampling with fixed k)

      --seed <SEED>
          random seed
          
          [default: 231]

  -t, --theta <THETA>
          value of theta (doesn't affect the results when k is fixed)
          
          [default: 1]

  -i, --initial-configuration <INITIAL_CONFIGURATION>
          initial configuration to start sampling from, uses conditional CRP sampler

  -b, --bias <BIAS>
          frequency-dependent bias factor of restaurant tables (should not be used in most analyses)

  -c, --format <FMT>
          output format

          Possible values:
          - binary:  output as sequence of (native-endian) u16 values; when reading, consume it in chunks to get configurations; chunk size is k if sampling with fixed k, or n if sampling from unconstrained Ewens distribution
          - tabular: output as ASCII characters, space-separated, one configuration per line;
          
          [default: binary]

  -h, --help
          Print help (see a summary with '-h')
```

```
> ./ewinfer test --help
Usage: ewinfer test [KIND] <CONFIGURATION>

Arguments:
  [KIND]
          test kind

          Possible values:
          - slatkin:   Slatkin's exact test
          - watterson: Watterson's homozygosity test
          
          [default: slatkin]

  <CONFIGURATION>
          configuration to run the exact test on; a space-separated list of unsigned integers; must have `k` elements with values summing to `n`

Options:
  -h, --help
          Print help (see a summary with '-h')
```
