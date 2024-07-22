use proconio::{input, marker::Chars};
use std::collections::HashMap;

fn main() {
    input! {
        n: usize,
        //m: usize,
        //a: [[i32]; n], // a is Vec<i32>, n-array.
        A: [Chars; n] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
    }
    let oppo = vec![
        ('L', 'W'),
        ('W', 'L'),
        ('D', 'D'),
    ].into_iter().collect::<HashMap<_, _>>();
    for i in 0..n{
        for j in i+1..n{
            if oppo[&A[i][j]] != A[j][i]{
                println!("incorrect");
                return ;
            }
        }
    }
    println!("correct")
}
