use byteorder::{ByteOrder, LittleEndian};
use color_eyre::eyre::{bail, Result, WrapErr};
use std::io::{ErrorKind, Read};

// NOTE: for future me and everyone else :3
//   see how we don't have a special separate function for conditional tests
// this is because conditional test adds a constant term to all probabilities
// in our logarithmic calculation, so it ends up cancelling out.
//   this only works because we are comparing log-probabilities with '<' here.
//   this would *also* work if we compared probabilities with 1 by division
//   this would NOT work if we compared actual probabilities with '<'

pub fn slatkin_test(configuration: &str) -> Result<()> {
    // parse the configuration
    //   the configuration is a list like "34 12 7 9 2 1 1" of length k and summing to n
    //   probability is proportional to 1/ the product of the values,
    //   so the log-probability is proportional to - sum of logs of these
    let uconf = configuration
        .split(' ')
        .map(|s| s.parse::<u16>())
        .collect::<Result<Vec<_>, _>>()
        .wrap_err_with(|| {
            format!(
                "failed while parsing configuration {:?} into unsigned integers",
                configuration
            )
        })?;

    let k = uconf.len();
    // let n = uconf.iter().sum::<u16>();

    // this is the -log P(c_0|k)
    let lp: f64 = uconf.into_iter().map(|x| (x as f64).ln()).sum();

    // now read the file from stdin as bytes
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    let buf_size = 2 * k * 1000;

    let mut buf = vec![0; buf_size];

    let mut total: usize = 0;
    let mut count: usize = 0;

    loop {
        match stdin.read_exact(&mut buf) {
            Ok(_) => {
                // iter over configurations
                let chunks = buf.chunks_exact(2 * k);

                chunks.for_each(|c| {
                    // iter over configuration entries
                    // to get - log P(c|k)
                    // println!("{:?}", c.len());
                    let lp_c: f64 = c
                        .chunks_exact(2)
                        .map(|x| (LittleEndian::read_u16(x) as f64).ln())
                        .sum();

                    // now we want to test if P(c|k) <= P(c_0|k)
                    //   in -log-p, we test -log P(c|k) >= -log P(c_0|k)
                    if lp_c >= lp {
                        count += 1
                    }

                    total += 1;
                });
            }

            Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => {
                break;
            }

            Err(e) => {
                bail!(e)
            }
        }
    }

    println!("{:?} {:?}", count, total);
    Ok(())
}

// NOTE: similarly but differently,
// note that there is no "conditional Watterson test"
// we are just testing for homozygosity using the conditionally sampled configurations

pub fn watterson_test(configuration: &str) -> Result<()> {
    // this works very similar to slatkin's test
    let uconf = configuration
        .split(' ')
        .map(|s| s.parse::<u16>())
        .collect::<Result<Vec<_>, _>>()
        .wrap_err_with(|| {
            format!(
                "failed while parsing configuration {:?} into unsigned integers",
                configuration
            )
        })?;

    let k = uconf.len();

    // unlike with slatkin, we just sum the squares of entries
    let f_0: usize = uconf
        .into_iter()
        .map(|x| {
            let x = x as usize;
            x * x
        })
        .sum();

    // now read the file from stdin as bytes
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    let buf_size = 2 * k * 1000;

    let mut buf = vec![0; buf_size];

    let mut total: usize = 0;
    let mut count: usize = 0;

    loop {
        match stdin.read_exact(&mut buf) {
            Ok(_) => {
                // iter over configurations
                let chunks = buf.chunks_exact(2 * k);

                chunks.for_each(|c| {
                    let f_c: usize = c
                        .chunks_exact(2)
                        .map(LittleEndian::read_u16)
                        .map(|x| {
                            let x = x as usize;
                            x * x
                        })
                        .sum();

                    if f_c <= f_0 {
                        count += 1
                    }

                    total += 1;
                });
            }

            Err(ref e) if e.kind() == ErrorKind::UnexpectedEof => {
                break;
            }

            Err(e) => {
                bail!(e)
            }
        }
    }

    println!("{:?} {:?}", count, total);
    Ok(())
}
