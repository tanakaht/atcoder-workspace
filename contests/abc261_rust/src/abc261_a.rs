use proconio::input;
use std::cmp;

fn main() {
    input! {
        l1: i32,
        r1: i32,
        l2: i32,
        r2: i32,
    }
    println!("{}", cmp::max(0, cmp::min(r1,  r2) - cmp::max(l1, l2)));
}
