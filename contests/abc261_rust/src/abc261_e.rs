use proconio::input;

fn main() {
    input! {
        n: usize,
        c: usize,
        //m: usize,
        TA: [[i32; 2]; n], // a is Vec<i32>, n-array.
        //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
    }
    let mut x = c;
    let mut f = [[0, 1]; 30];
    let op = [
        // and
        [
            [0, 0],
            [0, 1]
        ],
        //or
        [
            [0, 1],
            [1, 1]
        ],
        //xor
        [
            [0, 1],
            [1, 0]
        ]
    ];
    for i in 0..n{
        let t = TA[i][0];
        let c = TA[i][1];
        //opの更新
        for j in 0..30{
            let flg = (c>>j)&1;
            f[j][0] = op[(t-1) as usize][f[j][0] as usize][flg as usize];
            f[j][1] = op[(t-1) as usize][f[j][1] as usize][flg as usize];
        }
        //xの更新
        let mut newx = 0;
        for j in 0..30{
            let flg = (x>>j)&1;
            newx += (f[j][flg])<<j;
        }
        x = newx;
        println!("{}", x);
    }
}
