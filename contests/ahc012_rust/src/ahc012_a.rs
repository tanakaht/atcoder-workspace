use std::{f32::consts::PI, time::Duration};
use std::collections::HashMap;
use proconio::input;
use rand::Rng;
use std::time::Instant;

fn main() {
    let TS = Instant::now();
    let TIMELIMIT = Duration::new(2, 500000000);
    input! {
        N: usize,
        K: usize,
        A: [i32; 10], // a is Vec<i32>, n-array.
        XY_: [[i32; 2]; N] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
    }
    let mut anss: [(i32, i32, i32, i32); 100] = [(0, 1, 2, 3); 100];
    let mut state: Vec<[bool; 100]> = vec![[false; 100]; N];
    let mut XY: Vec<(i32, i32)> = vec![];
    for ele in XY_{
        let x = ele.get(0).unwrap();
        let y = ele.get(1).unwrap();
        XY.push((*x, *y));
    }
    fn printanss(anss: &[(i32, i32, i32, i32); 100]){
        println!("{}", 100);
        for (a, b, c, d) in anss{
            println!("{} {} {} {}", a, b, c, d)
        }
    }

    fn randint(a: i32, b:i32)->i32{
        let mut rng = rand::thread_rng();
        let i: i32 = rng.gen_range(a, b);
        return i;
    }

    fn get_random_points()->(i32, i32, i32, i32){
        let mut rng = rand::thread_rng();
        let r1: f32 = (rng.gen::<f32>())*((20000-7000) as f32)+(7000_f32);
        let theta1: f32 = rng.gen::<f32>()*PI*2_f32;
        let r2: f32 = (rng.gen::<f32>())*((20000-7000) as f32)+(7000_f32);
        let theta2: f32 = rng.gen::<f32>()*PI*2_f32;
        return ((r1*theta1.cos()) as i32, (r1*theta1.sin()) as i32, (r2*theta2.cos()) as i32, (r2*theta2.sin()) as i32);
    }

    fn is_upper(p1p2: (i32, i32, i32, i32), p: (i32, i32))->bool{
        let (x, y) = p;
        let (x1, y1, x2, y2) = p1p2;
        if (x1==x2){
            return x1>=x;
        } else {
            let y_ = ((y1*(x2-x)+y2*(x-x1)) as f32)/((x2-x1) as f32);
            return y_<=(y as f32);

        }
    }
    fn get_score(state: &Vec<[bool; 100]>, A: &Vec<i32>)->i32{
        let mut d: HashMap<[bool; 100], i32>= HashMap::new();
        for x in state{
            let counter = d.entry(*x).or_insert(0);
            *counter += 1;
        }
        let mut d2: HashMap<i32, i32>= HashMap::new();
        for x in d.values(){
            let counter = d2.entry(*x).or_insert(0);
            *counter += 1;
        }
        let mut score = 0;
        for i in 1..11{
            let x1 = *d2.get(&(i as i32)).unwrap_or(&0);
            let x2 = *A.get(i-1).unwrap_or(&0);
            if x1>x2{
                score += x2
            } else {
                score += x1
            }
        }
        return score;
    }
    for i in 0..100{
        anss[i] = get_random_points();
    }
    for i in 0..N{
        for j in 0..100{
            state[i][j] = is_upper(anss[j], *XY.get(i).unwrap());
        }
    }
    let mut cur_score = get_score(&state, &A);
    let mut cnt = 0;
    while Instant::now()-TS < TIMELIMIT {
        cnt += 1;
        let j = randint(0, 100 as i32) as usize;
        let new = get_random_points();
        for i in 0..N{
            state[i][j] = is_upper(new, *XY.get(i).unwrap());
        }
        let new_score = get_score(&state, &A);
        if cur_score<=new_score{
            cur_score = new_score;
            anss[j] = new;
            printanss(&anss);
        } else {
            for i in 0..N{
                state[i][j] = is_upper(anss[j], *XY.get(i).unwrap());
            }
        }
    }
    printanss(&anss);
    println!("{}", cnt);

}
