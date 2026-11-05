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
        let tail = t["tail_count"].as_u64().unwrap();
        assert!(tail <= 500);
        assert_eq!(t["total_count"], 500);
        assert_eq!(t["p_tail"].as_f64().unwrap(), (tail + 1) as f64 / 501.0);
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

// with alpha the test runs under pitman-yor, and reports its theta mle
#[test]
fn test_reports_alpha() {
    for alpha in [0.5, -1.0] {
        let a = alpha.to_string();
        let v = json(&["--test", "8 1 2 1", "-a", &a, "-s", "200", "-q", "--json"]);
        assert_eq!(v["alpha"], alpha);
        assert_eq!(v["theta"], theta_mle(12, 4, alpha, None));
        assert!(v.get("bias").is_none());
    }
}

#[test]
fn test_reports_human_readable() {
    let out = run(&["--test", "1 1 2 8", "-s", "100", "-q"]);
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    for line in [
        "configuration\t'1 1 2 8'",
        "slatkin\tp_tail\t",
        "watterson\tp_tail\t",
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
