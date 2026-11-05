use std::cmp::Ordering;

pub mod ewens;
pub mod pitman_yor;

// NOTE: for future me and everyone else :3
//   see how we don't have a special separate function for conditional tests
// this is because conditional test adds a constant term to all probabilities
// in our logarithmic calculation, so it ends up cancelling out.
//   this only works because we are comparing log-probabilities with '<' here.
//   this would *also* work if we compared probabilities with 1 by division
//   this would NOT work if we compared actual probabilities with '<'

pub trait Test {
    type Value;
    fn name(&self) -> &'static str;
    fn statistic(&self, uconf: &[u16]) -> Self::Value;
    // output ordering to deal with ties and tails tallies in main.rs
    fn compare(&self, tconf: &[u16]) -> Ordering;
}
