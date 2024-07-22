use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use proconio::input;



fn main() {
    pub struct FenwickTree<T> {
        n: usize,
        ary: Vec<T>,
        e: T,
    }
    impl<T: Clone + std::ops::AddAssign<T>> FenwickTree<T> {
        pub fn new(n: usize, e: T) -> Self {
            FenwickTree {
                n,
                ary: vec![e.clone(); n],
                e,
            }
        }
        pub fn accum(&self, mut idx: usize) -> T {
            let mut sum = self.e.clone();
            while idx > 0 {
                sum += self.ary[idx - 1].clone();
                idx &= idx - 1;
            }
            sum
        }
        /// performs data[idx] += val;
        pub fn add<U: Clone>(&mut self, mut idx: usize, val: U)
        where
            T: std::ops::AddAssign<U>,
        {
            let n = self.n;
            idx += 1;
            while idx <= n {
                self.ary[idx - 1] += val.clone();
                idx += idx & idx.wrapping_neg();
            }
        }
        /// Returns data[l] + ... + data[r - 1].
        pub fn sum(&self, l: usize, r: usize) -> T
        where
            T: std::ops::Sub<Output = T>,
        {
            self.accum(r) - self.accum(l)
        }
    }

    fn tentosuu(A: &Vec<i64>) -> i64 {
        let mut v2i = HashMap::new();
        let vals_: HashSet<i64> = HashSet::from_iter(A.iter().cloned());
        let mut vals = vals_.into_iter().collect::<Vec<_>>();
        vals.sort();
        for i in 0..vals.len(){
            v2i.insert(vals[i], i);
        }
        let mut bit = FenwickTree::new(vals.len()+1, 0);
        let mut ret: i64 = 0;
        for i in 0..A.len(){
            ret += (i as i64)-bit.accum(v2i[&A[i]]+1);
            bit.add(v2i[&A[i]], 1)
        }
        ret
    }


    input! {
        n: usize,
        //m: usize,
        c: [i64; n], // a is Vec<i64>, n-array.
        x: [i64; n], // a is Vec<i64>, n-array.
        //ab: [[i64; n]; m] // `a` is Vec<Vec<i64>>, (m, n)-matrix.
    }
    let mut colorwise_x: HashMap<i64, Vec<i64>> = HashMap::new();
    for i in 0..n{
        match colorwise_x.get_mut(&c[i]) {
            Some(v) => {
                v.push(x[i]);
            }
            None => {
                colorwise_x.insert(c[i], vec![x[i]]);
            }
        }
    }
    let mut ans = 0;
    ans += tentosuu(&x);
    for xx in colorwise_x.values(){
        ans -= tentosuu(xx);
    }
    println!("{}", ans);
}
