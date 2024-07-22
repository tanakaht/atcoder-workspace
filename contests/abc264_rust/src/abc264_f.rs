use proconio::{input, marker::Chars};

fn main() {
    input! {
        H: usize,
        W: usize,
        R : [i64; H], // a is Vec<i64>, n-array.
        C : [i64; W], // a is Vec<i64>, n-array.
        A_: [Chars; H] // `a` is Vec<Vec<i64>>, (m, n)-matrix.
    }
    let INF = std::i64::MAX;
    let mut dp = vec![vec![vec![INF; W+1]; H+1]; 4];
    let mut A = vec![vec![0; W+1]; H+1];
    for h in 0..H{
        for w in 0..W{
            A[h][w] = (A_[h][w]=='1') as i64;
        }
    }
    for i in 0..4{
        dp[i][H-1][W-1] = R[H-1]*((i as i64)%2)+C[W-1]*((i as i64)/2);
    }
    for h in (0..H).rev(){
        for w in (0..W).rev(){
            for i in 0..4{
                let c2 = C[w]*(i/2);
                let c3 = R[h]*(i%2);
                let j2 = (i^(((A[h][w]!=A[h][w+1]) as i64)*2)) as usize;
                let j3 = (i^((A[h][w]!=A[h+1][w]) as i64)) as usize;
                if (dp[j2][h+1][w]!=INF) && (dp[j3][h][w+1]!=INF){
                    dp[i as usize][h][w] = std::cmp::min(dp[j2][h][w+1]+c2, dp[j3][h+1][w]+c3);
                } else if dp[j2][h][w+1]!=INF{
                    dp[i as usize][h][w] = dp[j2][h][w+1]+c2
                } else if dp[j3][h+1][w]!=INF{
                    dp[i as usize][h][w] = dp[j3][h+1][w]+c3;
                }
            }
        }
    }
    let mut ans = std::i64::MAX;
    for i in 0..4{
        ans = std::cmp::min(ans, dp[i][0][0]);
    }
    println!("{}", ans)
}
