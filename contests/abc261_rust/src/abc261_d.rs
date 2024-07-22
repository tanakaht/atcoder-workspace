use std::cmp;

use proconio::input;

fn main() {
    input! {
        n: usize,
        m: usize,
        X: [i64; n], // a is Vec<i64>, n-array.
        CY: [[i64; 2]; m] // `a` is Vec<Vec<i64>>, (m, n)-matrix.
    }
    // [0, i)を実行して、カウントがjの時の最大スコア
    let mut dp = vec![vec![-1024*1024*1024*1024*1024; n+1]; n+1];
    let mut bonus = vec![0;n+1];
    for cy in CY{
        bonus[cy[0] as usize] = cy[1];
    }
    dp[0][0] = 0;
    for i in 0..n{
        for j in 0..n{
            dp[i+1][0] = cmp::max(dp[i+1][0], dp[i][j]+bonus[0]);
            dp[i+1][j+1] = cmp::max(dp[i+1][j+1], dp[i][j]+X[i]+bonus[j+1]);
        }
    }
    println!("{}", dp[n].iter().max().unwrap())
}
