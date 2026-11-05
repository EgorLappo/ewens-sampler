//! end-to-end tests of the command line tool

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use ewens_sampler::samplers::pitman_yor::theta::theta_mle;

const BIN: &str = env!("CARGO_BIN_EXE_ewens-sampler");
const TIMEOUT: Duration = Duration::from_secs(10);

// run the binary and kill it if it takes too long, so a hang fails the test
fn run(args: &[&str]) -> Output {
    let mut child = Command::new(BIN)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > TIMEOUT {
            child.kill().unwrap();
            panic!(
                "`ewens-sampler {}` still running after {TIMEOUT:?}",
                args.join(" ")
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    child.wait_with_output().unwrap()
}

fn configurations(args: &[&str]) -> Vec<Vec<u16>> {
    let out = run(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| l.split(' ').map(|t| t.parse().unwrap()).collect())
        .collect()
}

fn json(args: &[&str]) -> serde_json::Value {
    let out = run(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

// errors should come back as messages, not panics or hangs
fn error(args: &[&str]) -> String {
    let out = run(args);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "`ewens-sampler {}` succeeded",
        args.join(" ")
    );
    assert!(
        !stderr.contains("panicked"),
        "`ewens-sampler {}` panicked:\n{stderr}",
        args.join(" ")
    );
    stderr
}

// *sampling

#[test]
fn samples_with_fixed_k() {
    for extra in [
        &[][..],
        &["-i", "1 3"],
        &["-b", "1.5"],
        &["-i", "1 3", "-b", "0.5"],
        // pitman-yor, over the whole range of alpha
        &["-a", "0.5"],
        &["-a", "-1"],
        &["-a", "-inf"],
        &["-a", "1"],
    ] {
        let args = [&["-n", "12", "-k", "4", "-s", "100", "-q"][..], extra].concat();
        let confs = configurations(&args);
        assert_eq!(confs.len(), 100);
        for c in confs {
            assert!(
                c.len() == 4 && c.iter().sum::<u16>() == 12,
                "{args:?}: {c:?}"
            );
            assert!(c.is_sorted() && c[0] >= 1, "{args:?}: {c:?}");
        }
    }
}

#[test]
fn samples_without_fixed_k() {
    // with the most alleles each can give: pitman-yor takes a negative theta for alpha > 0,
    // and for alpha < 0 theta = 3 |alpha| is 3 types
    for (extra, max_k) in [
        (&["-t", "2.5"][..], 12),
        (&["-t", "2.5", "-a", "0.5"], 12),
        (&["-t", "-0.3", "-a", "0.5"], 12),
        (&["-t", "3", "-a", "-1"], 3),
    ] {
        let args = [&["-n", "12", "-s", "100", "-q"][..], extra].concat();
        let confs = configurations(&args);
        assert_eq!(confs.len(), 100);
        for c in confs {
            assert!(
                c.iter().sum::<u16>() == 12 && c.is_sorted() && c[0] >= 1 && c.len() <= max_k,
                "{args:?}: {c:?}"
            );
        }
    }
}

#[test]
fn seed_determines_samples() {
    let args = |seed| ["-n", "30", "-k", "7", "-s", "50", "-q", "--seed", seed];
    assert_eq!(configurations(&args("5")), configurations(&args("5")));
    assert_ne!(configurations(&args("5")), configurations(&args("6")));
}

// *testing

#[test]
fn test_reports_json() {
    let v = json(&["--test", "8 1 2 1", "-s", "500", "-q", "--json"]);
    assert_eq!(v["configuration"], "1 1 2 8");
    assert_eq!(v["replicates"], 500);
    assert!(v["theta"].as_f64().unwrap() > 0.0);
    assert!(v.get("initial_configuration").is_none() && v.get("bias").is_none());

    let tests = v["tests"].as_array().unwrap();
    let names: Vec<_> = tests.iter().map(|t| t["test"].as_str().unwrap()).collect();
    assert_eq!(names, ["slatkin", "watterson"]);
    for t in tests {
        let lower = t["lower_count"].as_u64().unwrap();
        let upper = t["upper_count"].as_u64().unwrap();
        // every replicate is in one of the tails, and ties are in both
        assert!(lower <= 500 && upper <= 500 && lower + upper >= 500, "{t}");
        assert_eq!(t["total_count"], 500);
        assert_eq!(t["p_lower"].as_f64().unwrap(), (lower + 1) as f64 / 501.0);
        assert_eq!(t["p_upper"].as_f64().unwrap(), (upper + 1) as f64 / 501.0);
    }
}

// with k = n the only configuration is all singletons, so every replicate ties with it
// and is in both tails
#[test]
fn test_counts_ties_in_both_tails() {
    let v = json(&["--test", "1 1 1 1", "-s", "100", "-q", "--json"]);
    for t in v["tests"].as_array().unwrap() {
        assert_eq!(t["lower_count"], 100, "{t}");
        assert_eq!(t["upper_count"], 100, "{t}");
        assert_eq!(t["p_lower"], 1.0, "{t}");
        assert_eq!(t["p_upper"], 1.0, "{t}");
    }
}

// (1, 9) is the most probable configuration for n = 10 and k = 2 under ewens, so its lower tail
// has every replicate, and its upper tail is just its own probability, 0.39.
// reading the upper side off the lower tail as 1 - p_lower would reject it at any level
#[test]
fn test_upper_tail_of_the_most_probable_configuration() {
    let v = json(&["--test", "1 9", "-s", "2000", "-q", "--json"]);
    for t in v["tests"].as_array().unwrap() {
        assert_eq!(t["p_lower"], 1.0, "{t}");
        let p = t["p_upper"].as_f64().unwrap();
        assert!((0.35..0.44).contains(&p), "{t}");
    }
}

#[test]
fn test_reports_initial_configuration_and_bias() {
    let v = json(&[
        "--test", "1 1 2 8", "-i", "2 1", "-b", "1.3", "-s", "200", "-q", "--json",
    ]);
    assert_eq!(v["initial_configuration"], "1 2");
    assert_eq!(v["bias"], 1.3);
}

// with alpha the test runs under pitman-yor, with its own tests, and reports its theta mle
#[test]
fn test_reports_alpha() {
    for alpha in [0.5, -1.0] {
        let a = alpha.to_string();
        let v = json(&["--test", "8 1 2 1", "-a", &a, "-s", "200", "-q", "--json"]);
        assert_eq!(v["alpha"], alpha);
        assert_eq!(v["theta"], theta_mle(12, 4, alpha, None));
        assert!(v.get("bias").is_none());
        let tests = v["tests"].as_array().unwrap();
        let names: Vec<_> = tests.iter().map(|t| t["test"].as_str().unwrap()).collect();
        assert_eq!(names, ["exact", "score"]);
    }
}

// at alpha = -1 every configuration is as probable as any other, so the exact test ties
// everything, while the score still tells them apart
#[test]
fn test_exact_ties_everything_at_minus_one() {
    let v = json(&["--test", "8 1 2 1", "-a", "-1", "-s", "200", "-q", "--json"]);
    let exact = &v["tests"][0];
    assert_eq!(exact["test"], "exact");
    assert_eq!(exact["lower_count"], 200, "{exact}");
    assert_eq!(exact["upper_count"], 200, "{exact}");
    let score = &v["tests"][1];
    assert!(score["lower_count"].as_u64().unwrap() < 200, "{score}");
}

#[test]
fn test_reports_human_readable() {
    let out = run(&["--test", "1 1 2 8", "-s", "100", "-q"]);
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    for line in [
        "configuration\t'1 1 2 8'",
        "slatkin\tp_lower\t",
        "slatkin\tp_upper\t",
        "watterson\tp_lower\t",
        "watterson\tp_upper\t",
    ] {
        assert!(s.contains(line), "missing {line:?} in\n{s}");
    }
}

// *bad input

#[test]
fn rejects_k_larger_than_n() {
    assert!(error(&["-n", "5", "-k", "6", "-q"]).contains("invalid parameters"));
    // also for pitman-yor, whose sampler would panic
    assert!(error(&["-n", "5", "-k", "6", "-a", "0.5", "-q"]).contains("invalid parameters"));
}

#[test]
fn rejects_alpha_above_one() {
    assert!(error(&["-n", "12", "-k", "4", "-a", "1.5", "-q"]).contains("at most 1"));
}

// the limits can be sampled from, but have nothing to test
#[test]
fn rejects_testing_at_the_limits_of_alpha() {
    for a in ["1", "-inf"] {
        let e = error(&["--test", "8 1 2 1", "-a", a, "-s", "10", "-q"]);
        assert!(e.contains("-inf < alpha < 1"), "{e}");
    }
}

#[test]
fn rejects_alpha_with_bias() {
    error(&["-n", "12", "-k", "4", "-a", "0.5", "-b", "1.3", "-q"]);
}

#[test]
fn rejects_negative_theta_without_alpha() {
    assert!(error(&["-n", "12", "-t", "-1", "-q"]).contains("0 <= theta"));
}

// for alpha < 0 the free sampler needs a whole number of types, and says so instead of panicking
#[test]
fn rejects_theta_that_is_not_a_whole_number_of_types() {
    assert!(error(&["-n", "12", "-t", "2.5", "-a", "-1", "-q"]).contains("whole number"));
}

#[test]
fn rejects_zero_counts() {
    assert!(error(&["--test", "1 0 2", "-q"]).contains("allele counts must be >= 1"));
}

#[test]
fn rejects_initial_configuration_with_more_than_k_alleles() {
    error(&["-n", "10", "-k", "2", "-i", "1 1 1", "-q"]);
}

#[test]
fn rejects_initial_configuration_with_all_n_samples() {
    error(&["-n", "7", "-k", "3", "-i", "3 4", "-q"]);
}
