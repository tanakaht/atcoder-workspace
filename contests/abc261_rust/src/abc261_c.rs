use std::collections::HashMap;

use proconio::{input, marker::Chars};

fn main() {
    input! {
        n: usize,
        ss: [Chars; n] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
    }
    let mut counter: HashMap<String, usize> = HashMap::new();
    for i in 0..n{
        let s: String = ss[i].iter().collect();
        match counter.get_mut(&s) {
            Some(v) => {
                println!("{}({})", s, *v);
                *v += 1;
            }
            None => {
                println!("{}", s);
                counter.insert(s, 1);
            }
        }
    }
}
