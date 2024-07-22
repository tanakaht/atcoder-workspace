use proconio::input;
use std::collections::BinaryHeap;
use std::cmp::min;
static INF: usize = std::usize::MAX;

fn print_vec<T: std::fmt::Display>(v: &[T]) {
    if v.len() == 0 {
        println!();
    } else {
        let mut v = v.iter();
        print!("{}", v.next().unwrap());
        for v in v {
            print!(" {}", v);
        }
        println!();
    }
}
fn main() {
    input! {
        N: usize,
        M: usize,
        K: usize,
        L: usize,
        A: [usize; N],
        B: [usize; L],
        UVC: [[usize; 3]; M],
        //m: usize,
        //a: [[i32]; n], // a is Vec<i32>, n-array.
        //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
    }
    let mut g: Vec<Vec<(usize, usize)>> = vec![Vec::new(); N];
    let mut dists: Vec<Vec<Vec<usize>>> = vec![vec![vec![INF; N]; 2]; 20];
    let mut anss: Vec<i64> = Vec::new();
    for uvc in UVC.iter() {
        let u = uvc[0];
        let v = uvc[1];
        let c = uvc[2];
        g[u-1].push((v-1, c));
        g[v-1].push((u-1, c));
    }
    fn dijkstra(ninki: Vec<usize>, g: &Vec<Vec<(usize, usize)>>, N: usize) -> Vec<usize>{
        let mut dist = vec![INF; N];
        let mut appeared: Vec<bool> = vec![false; N];
        let mut q: BinaryHeap<(usize, usize)> = BinaryHeap::new();
        for u in ninki {
            q.push((INF, u));
            dist[u] = 0;
        }
        while let Some((d_inv, u)) = q.pop() {
            let d = INF - d_inv;
            if appeared[u] {
                continue;
            }
            appeared[u] = true;
            for (v, c) in g[u].iter(){
                if appeared[*v]{
                    continue;
                }
                let d_ = d+(*c);
                if d_ < dist[*v]{
                    dist[*v] = d_;
                    q.push((INF-d_, *v));
                }
            }
        }
        return dist
    }

    for i in 0..20_usize{
        for flg in 0..2{
            let mut ninki:Vec<usize> = Vec::new();
            for u_ in B.iter(){
                let u = *u_-1;
                let a = A[u]-1;
                if ((a>>i)&1)^flg==1 {
                    ninki.push(u)
                }
            }
            dists[i][flg] = dijkstra(ninki, &g, N);
        }
    }
    for u in 0..N{
        let mut ans = INF;
        for i in 0..20_usize{
            let a = A[u]-1;
            let flg = (a>>i)&1;
            ans = min(ans, dists[i][flg][u]);
        }
        if ans==INF{
            anss.push(-1);
        } else {
            anss.push(ans as i64);
        }
    }
    println!("{}", anss.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(" "))
}
