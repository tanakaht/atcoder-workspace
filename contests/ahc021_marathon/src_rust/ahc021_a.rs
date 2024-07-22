#![allow(non_snake_case)]


const ENDED_BONUS:i64=1100;
static mut RR:usize=30;


#[derive(Clone,Copy)]
struct Stack{
    stamp:[usize;L],
    at:usize,
    stack:[usize;L],
}
impl Stack{
    fn new()->Stack{
        Stack{
            stamp:[!0;L],
            at:0,
            stack:[!0;L]
        }
    }

    fn push(&mut self,n:usize,turn:usize){
        assert!(self.stamp[n]==!0);
        self.stamp[n]=turn;
        self.stack[self.at]=n;
        self.at+=1;
    }

    fn pop(&mut self){
        self.at-=1;
        assert!(self.stamp[self.stack[self.at]]!=!0);
        self.stamp[self.stack[self.at]]=!0;
    }

    fn contains(&self,n:usize)->bool{
        self.stamp[n]!=!0
    }

    fn to_slice(&self)->&[usize]{
        &self.stack[..self.at]
    }
}


#[derive(Clone)]
struct State{
    state:[u16;L+1],
    pos:[u16;L],
    cands:[Stack;L],
    last_swap:[usize;L],
}
impl State{
    fn new(input:&In)->State{
        let mut pos=[!0;L];
        for (i,&n) in input.grid.iter().enumerate(){
            pos[n as usize]=i as u16;
        }

        let mut state=[0;L+1];
        state[..L].copy_from_slice(&input.grid);
        let cands=[Stack::new();L];
        let last_swap=[!0;L];

        State{state,pos,cands,last_swap}
    }

    fn isok(&self,input:&In,n:usize)->bool{
        if n==465{
            return false;
        }
        let p=self.pos[n] as usize;
        let np0=input.dd[p][0];
        let np1=input.dd[p][1];

        n>=self.state[np0] as usize && n>=self.state[np1] as usize
    }

    fn score(&self,input:&In,n:usize)->i64{
        (0..L).map(|i|
            input.height[i]*self.state[i] as i64
        ).sum::<i64>()
        +n as i64*ENDED_BONUS
    }

    fn hash(&self,input:&In,n:usize)->u64{
        let mut hash=0;
        for i in 0..n{
            hash^=input.zob[0][self.pos[i] as usize];
        }
        hash
    }

    fn apply(&mut self,node:&Node){
        self.swap(node.op);
        for &(a,b) in &node.diff{
            if a==!0{
                break;
            }
            self.cands[a as usize].push(b as usize,node.turn);
        }

        if node.swap!=!0{
            let (a,b)=(self.state[node.op.0] as usize,self.state[node.op.1] as usize);
            assert!(node.swap<self.last_swap[a]);
            assert!(node.swap<self.last_swap[b]);
            self.last_swap[a]=node.swap;
            self.last_swap[b]=node.swap;
        }
    }

    fn revert(&mut self,node:&Node){
        self.swap(node.op);
        for &(a,_) in &node.diff{
            if a==!0{
                break;
            }
            self.cands[a as usize].pop();
        }

        if node.swap!=!0{
            let (a,b)=(self.state[node.op.0] as usize,self.state[node.op.1] as usize);
            assert!(node.backup!=(0,0));
            self.last_swap[a]=node.backup.1;
            self.last_swap[b]=node.backup.0;
        }
    }

    fn swap(&mut self,(p0,p1):(usize,usize)){
        self.pos.swap(self.state[p0] as usize,self.state[p1] as usize);
        self.state.swap(p0,p1);
    }
}


#[derive(Clone)]
struct Cand{
    op:(usize,usize),
    parent:usize,
    score:i64,
    n:usize,
    hash:u64,
    diff:[(usize,usize);3],
    swap:usize,
    eval_score:i64,
    backup:(usize,usize),
}
impl Cand{
    fn to_node(&self,turn:usize)->Node{
        Node{
            op:self.op,
            parent:self.parent,
            child:!0,
            prev:!0,
            next:!0,
            score:self.score,
            n:self.n,
            hash:self.hash,
            diff:self.diff,
            swap:self.swap,
            turn,
            backup:self.backup,
        }
    }
}


struct Node{
    op:(usize,usize),
    parent:usize,
    child:usize,
    prev:usize,
    next:usize,
    score:i64,
    n:usize,
    hash:u64,
    diff:[(usize,usize);3],
    swap:usize,
    turn:usize,
    backup:(usize,usize),
}
impl Node{
    fn new(score:i64,n:usize,hash:u64)->Node{
        Node{
            score,n,hash,
            op:(!0,!0),
            parent:!0,
            child:!0,
            prev:!0,
            next:!0,
            diff:[(!0,!0);3],
            swap:!0,
            turn:0,
            backup:(!0,!0)
        }
    }
}


struct Beam{
    state:State,
    latest:usize,
    nodes:Vec<Node>,
    cur_node:usize,
}
impl Beam{
    fn new(state:State,node:Node)->Beam{
        let mut nodes=Vec::with_capacity(300*2000);
        nodes.push(node);

        Beam{
            state,nodes,
            latest:0,
            cur_node:0,
        }
    }

    fn add_node(&mut self,cand:&Cand,turn:usize){
        let next=self.nodes[cand.parent].child;
        if next!=!0{
            self.nodes[next].prev=self.nodes.len();
        }
        self.nodes[cand.parent].child=self.nodes.len();

        self.nodes.push(Node{next,..cand.to_node(turn)});
    }

    fn del_node(&mut self,mut idx:usize){
        loop{
            let Node{prev,next,parent,..}=self.nodes[idx];
            assert_ne!(parent,!0);
            if prev&next==!0{
                idx=parent;
                continue;
            }

            if prev!=!0{
                self.nodes[prev].next=next;
            }
            else{
                self.nodes[parent].child=next;
            }
            if next!=!0{
                self.nodes[next].prev=prev;
            }

            break;
        }
    }

    fn restore(&self,mut idx:usize)->Vec<(usize,usize,usize)>{
        let mut ret=vec![];

        loop{
            let Node{op,swap,parent,..}=self.nodes[idx];
            if op==(!0,!0){
                break;
            }
            ret.push((op.0,op.1,swap));
            idx=parent;
        }

        ret.reverse();
        ret
    }

    fn update<'a,I:Iterator<Item=&'a Cand>>(&mut self,cands:I,turn:usize){
        let len=self.nodes.len();
        for cand in cands{
            self.add_node(cand,turn);
        }

        for i in self.latest..len{
            if self.nodes[i].child==!0{
                self.del_node(i);
            }
        }
        self.latest=len;
    }

    fn dfs(&mut self,input:&In,cands:&mut Vec<Cand>,single:bool){
        if self.nodes[self.cur_node].child==!0{
            self.append_cands(input,self.cur_node,cands);
            return;
        }

        let node=self.cur_node;
        let mut child=self.nodes[node].child;
        let next_single=single&(self.nodes[child].next==!0);

        // let prev_state=self.state.clone();
        loop{
            self.cur_node=child;
            self.state.apply(&self.nodes[child]);
            self.dfs(input,cands,next_single);

            if !next_single{
                self.state.revert(&self.nodes[child]);
                // assert!(self.state==prev_state);
            }
            child=self.nodes[child].next;
            if child==!0{
                break;
            }
        }

        if !next_single{
            self.cur_node=node;
        }
    }

    fn enum_cands(&mut self,input:&In,cands:&mut Vec<Cand>){
        self.dfs(input,cands,true);
    }

    fn append_cands(&mut self,input:&In,idx:usize,cands:&mut Vec<Cand>){
        let node=&self.nodes[idx];
        assert_eq!(node.child,!0);
        // assert_eq!(node.score,self.state.score(input,node.n));
        // assert_eq!(node.hash,self.state.hash(input,node.n));
        // assert!(!self.state.isok(input,node.n));

        let pos=self.state.pos[node.n] as usize;
        let f=pos==node.op.1;

        static mut AT:usize=0;
        static mut SEEN:[usize;L]=[0;L];

        unsafe{AT+=1;}

        // for i in 0..6{
        for &i in &[0,1,2,5]{
            let np=input.dd[pos][i];
            if node.n>=self.state.state[np] as usize || f && np==node.op.0{
                continue;
            }

            let mut score=node.score;
            self.state.swap((pos,np));

            let mut n=node.n;
            let mut hash=node.hash;

            while self.state.isok(input,n){
                let pos=self.state.pos[n] as usize;
                hash^=input.zob[0][pos];
                n+=1;
            }

            score+=ENDED_BONUS*(n-node.n) as i64;
            self.state.swap((pos,np));

            let nn=self.state.state[np] as usize;
            let old=input.height[pos]*node.n as i64+input.height[np]*nn as i64;
            let new=input.height[pos]*nn as i64+input.height[np]*node.n as i64;
            score+=new-old;

            unsafe{SEEN[nn]=AT;}

            let mut diff=[(!0,!0);3];
            if self.state.last_swap[nn]==!0{
                let mut at=0;
                for j in 0..3{
                    let mp=input.side[np][i][j];
                    let mm=self.state.state[mp] as usize;
                    if node.n<mm{
                        let (n,m)=if mm<nn{
                            (mm,nn)
                        } else {
                            (nn,mm)
                        };

                        if self.state.last_swap[n]==!0 && self.state.last_swap[m]==!0 && !self.state.cands[n].contains(m){
                            diff[at]=(n,m);
                            at+=1;
                        }
                    }
                }
            }

            let cand=Cand{
                op:(pos,np),
                parent:idx,
                score,n,hash,diff,
                swap:!0,
                eval_score:score+(rnd::next()%unsafe{RR}) as i64,
                backup:(0,0)
            };
            cands.push(cand);
        }

        if !f{
            let last=self.state.last_swap[node.n];
            for &nn in self.state.cands[node.n].to_slice(){
                let turn=self.state.cands[node.n].stamp[nn];
                if last<=turn{
                    break;
                }
                if unsafe{SEEN[nn]==AT} || self.state.last_swap[nn]<=turn{
                    continue;
                }

                let np=self.state.pos[nn] as usize;
                let mut score=node.score;

                self.state.pos.swap(self.state.state[pos] as usize,self.state.state[np] as usize);
                self.state.state.swap(pos,np);

                if !self.state.isok(input,node.n) && input.height[pos]<input.height[np]{
                    self.state.pos.swap(self.state.state[pos] as usize,self.state.state[np] as usize);
                    self.state.state.swap(pos,np);
                    continue;
                }

                let mut n=node.n;
                let mut hash=node.hash;

                while self.state.isok(input,n){
                    let pos=self.state.pos[n] as usize;
                    hash^=input.zob[0][pos];
                    n+=1;
                }

                score+=ENDED_BONUS*(n-node.n) as i64;
                self.state.pos.swap(self.state.state[pos] as usize,self.state.state[np] as usize);
                self.state.state.swap(pos,np);

                let old=input.height[pos]*node.n as i64+input.height[np]*nn as i64;
                let new=input.height[pos]*nn as i64+input.height[np]*node.n as i64;
                score+=new-old;

                assert!(turn<self.state.last_swap[node.n] && turn<self.state.last_swap[nn]);

                let cand=Cand{
                    op:(pos,np),
                    parent:idx,
                    score,n,hash,diff:[(!0,!0);3],
                    swap:turn,
                    eval_score:score+(rnd::next()%unsafe{RR}) as i64,
                    backup:(self.state.last_swap[nn],self.state.last_swap[node.n]),
                };
                cands.push(cand);
            }
        }
    }
}


use std::cmp::Reverse;


fn beam(input:&In,M:usize)->Vec<(usize,usize,usize)>{
    let mut beam={
        let state=State::new(input);
        let mut n=0;
        while state.isok(input,n){
            n+=1;
        }
        let score=state.score(input,n);
        let hash=state.hash(input,n);
        let node=Node::new(score,n,hash);
        Beam::new(state,node)
    };
    let mut cands:Vec<Cand>=vec![];
    let mut first=true;

    let mut set=FxHashSet::default();
    let mut is=vec![];
    let best;

    let mut turn=0;
    loop{
        if let Some(idx)=(0..cands.len()).find(|&i|cands[i].n==L){
            best=cands[idx].clone();
            break;
        }

        turn+=1;
        if turn&31==0 && get_time()>=1.95{
            return vec![];
        }

        const R0:f64=40.;
        const R1:f64=25.;
        unsafe{
            RR=(R0+(R1-R0)*(turn as f64/1900.)).round() as usize;
        }

        if !first{
            let MM=(M as f64*2.).round() as usize;
            is.clear();
            is.extend(0..cands.len());
            if is.len()>MM{
                nth::select_nth_unstable_by_key(&mut is,MM,|&i|Reverse(cands[i].score));
                is.truncate(MM);
            }
            is.sort_unstable_by_key(|&i|Reverse(cands[i].eval_score));

            set.clear();
            let it=is.iter().map(|&i|&cands[i]).filter(|cand|
                set.insert(cand.hash^input.zob[1][cand.op.1])
            ).take(M);
            beam.update(it,turn);
        }
        first=false;

        cands.clear();
        beam.enum_cands(input,&mut cands);
        assert_ne!(cands.len(),0);
    }

    let mut ret=beam.restore(best.parent);
    ret.push((best.op.0,best.op.1,best.swap));

    ret
}


fn restore(input:&In,res:&[(usize,usize,usize)])->Vec<(usize,usize)>{
    let mut grid=input.grid;
    let mut que=vec![];
    for &(a,b,c) in res{
        if c!=!0{
            que.push((grid[a] as usize,grid[b] as usize,c));
        }
        grid.swap(a,b);
    }

    que.sort_unstable_by_key(|t|t.2);
    let mut color=[!0;L];
    for i in 0..L{
        color[i]=i;
    }
    for i in 0..que.len(){
        let (a,b,c)=que[i];
        que[i]=(color[a],color[b],c);
        color.swap(a,b);
    }

    let isok=|p0:usize,p1:usize|input.dd[p0].iter().any(|&np|np==p1);
    let mut pos=[!0;L];
    for (i,&n) in input.grid.iter().enumerate(){
        pos[n as usize]=i;
    }
    que.reverse();
    let mut ret=vec![];
    let mut grid=input.grid;
    while let Some(&(c0,c1,_))=que.last(){
        let (p0,p1)=(pos[c0],pos[c1]);
        if isok(p0,p1){
            ret.push((p0,p1));
            que.pop().unwrap();
            grid.swap(p0,p1);
            pos.swap(grid[p0] as usize,grid[p1] as usize);
        }
        else{
            break;
        }
    }

    for &(p0,p1,c) in res{
        if c!=!0{
            continue;
        }

        grid.swap(p0,p1);
        pos.swap(grid[p0] as usize,grid[p1] as usize);
        ret.push((p0,p1));

        while let Some(&(c0,c1,_))=que.last(){
            let (p0,p1)=(pos[c0],pos[c1]);
            if isok(p0,p1){
                ret.push((p0,p1));
                que.pop().unwrap();
                grid.swap(p0,p1);
                pos.swap(grid[p0] as usize,grid[p1] as usize);
            }
            else{
                break;
            }
        }
    }

    assert!(que.is_empty());

    ret
}


fn local_search(input:&In,ans:&[(usize,usize)])->Vec<(usize,usize)>{
    let mut last=vec![!0;L];
    let mut graph=vec![vec![];ans.len()];
    let mut cnt=vec![0;ans.len()];
    for i in 0..ans.len(){
        let (p0,p1)=ans[i];
        if last[p0]!=!0{
            graph[last[p0]].push(i);
            cnt[i]+=1;
        }
        if last[p1]!=!0{
            graph[last[p1]].push(i);
            cnt[i]+=1;
        }
        last[p0]=i;
        last[p1]=i;
    }

    let mut ord=vec![];
    let mut que=(0..ans.len()).filter(|&i|cnt[i]==0).collect::<Vec<_>>();
    let mut grid=input.grid;
    while !que.is_empty(){
        let idx=rnd::next()%que.len();
        let n=que.swap_remove(idx);
        let (p0,p1)=ans[n];
        grid.swap(p0,p1);
        ord.push((ans[n],(grid[p0],grid[p1])));
        for &i in &graph[n]{
            cnt[i]-=1;
            if cnt[i]==0{
                que.push(i);
            }
        }
    }

    // let mut ord=vec![];
    // let mut grid=input.grid;
    // for &(p0,p1) in &ans{
    //     grid.swap(p0,p1);
    //     ord.push(((p0,p1),(grid[p0],grid[p1])));
    // }

    let isok=|grid:&[u16],p:usize|->bool{
        for i in 0..2{
            let np=input.dd[p][i];
            if np!=L && grid[np]<grid[p]{
                return false;
            }
        }
        for i in 3..5{
            let np=input.dd[p][i];
            if np!=L && grid[np]>grid[p]{
                return false;
            }
        }

        true
        // false
    };

    let mut pos=[!0;L];
    for (i,&n) in grid.iter().enumerate(){
        pos[n as usize]=i;
    }

    let mut ret=vec![];
    for ((p0,p1),(n0,n1)) in ord.into_iter().rev(){
        grid.swap(pos[n0 as usize],pos[n1 as usize]);
        if isok(&grid,p0) && isok(&grid,p1){
            pos.swap(n0 as usize,n1 as usize);
        }
        else{
            ret.push((p0,p1));
            grid.swap(pos[n0 as usize],pos[n1 as usize]);
        }
    }
    ret.reverse();

    ret
}


fn simulate(input:&In,ans:&[(usize,usize)]){
    let mut grid=input.grid;
    for &(p0,p1) in ans{
        grid.swap(p0,p1);
    }

    for i in 0..L{
        for &np in &input.dd[i][..2]{
            if np!=L{
                assert!(grid[i]>grid[np]);
            }
        }
    }
}


fn run(){
    get_time();
    let input=In::input();
    let mut best_ans=vec![];
    let mut best_score=!0;
    let mut M=300;
    let mut iter=0;
    loop{
        M+=1;
        iter+=1;
        let res=beam(&input,M);
        if res.is_empty(){
            break;
        }
        let ans=restore(&input,&res);
        if best_score>ans.len(){
            best_score=ans.len();
            best_ans=ans;
        }
        // break;
    }
    let ans=local_search(&input,&best_ans);
    simulate(&input,&ans);

    eprintln!("iter = {}",iter);
    eprintln!("score = {}",ans.len());
    eprintln!("time = {}",get_time());

    println!("{}",ans.len());
    for &n in &ans{
        let (i0,j0)=input.to_pos[n.0];
        let (i1,j1)=input.to_pos[n.1];
        println!("{} {} {} {}",i0,j0,i1,j1);
    }
}


fn main(){
    let _=std::thread::Builder::new().name("run".to_string()).stack_size(32*1024*1024).spawn(run).unwrap().join();
}


const N:usize=30;
const L:usize=465;


use proconio::*;
use rand::prelude::*;
use rustc_hash::*;


struct In{
    grid:[u16;L],
    zob:[[u64;L];2],
    dd:[[usize;6];L],
    side:[[[usize;3];6];L],
    height:[i64;L],
    to_pos:[(usize,usize);L],
}
impl In{
    fn input()->In{
        let mut to_id=[[!0;N];N];
        let mut it=0..;
        let mut to_pos=[(!0,!0);L];
        for i in 0..N{
            for j in 0..=i{
                to_id[i][j]=it.next().unwrap();
                to_pos[to_id[i][j]]=(i,j);
            }
        }

        input!{
            i_grid:[u16;L]
        }
        let mut grid=[!0;L];
        grid.copy_from_slice(&i_grid);

        let mut zob=[[!0;L];2];
        let mut rng=rand_pcg::Pcg64Mcg::new(0);
        for i in 0..2{
            for j in 0..L{
                zob[i][j]=rng.gen();
            }
        }

        let mut dd=[[L;6];L];
        for &n in to_pos.iter(){
            for (i,&d) in [(!0,!0),(!0,0),(0,1),(1,1),(1,0),(0,!0)].iter().enumerate(){
                let np=(n.0+d.0,n.1+d.1);
                if np.0<N && np.1<N && to_id[np.0][np.1]!=!0{
                    dd[to_id[n.0][n.1]][i]=to_id[np.0][np.1];
                }
            }
        }

        let mut height=[0;L];
        for i in 0..L{
            height[i]=(to_pos[i].0 as f64).powf(1.).round() as i64; // todo
        }

        let mut side=[[[L;3];6];L];
        for i in 0..L{
            for j in 0..6{
                for k in 0..3{
                    let d=(j+k+5)%6;
                    side[i][j][k]=dd[i][d];
                }
            }
        }

        In{grid,zob,dd,height,to_pos,side}
    }
}



#[macro_export]#[cfg(not(local))]macro_rules! eprint{($($_:tt)*)=>{}}
#[macro_export]#[cfg(not(local))]macro_rules! eprintln{($($_:tt)*)=>{}}
#[macro_export]#[cfg(not(local))]macro_rules! assert{($($_:tt)*)=>{}}
#[macro_export]#[cfg(not(local))]macro_rules! assert_eq{($($_:tt)*)=>{}}
#[macro_export]#[cfg(not(local))]macro_rules! assert_ne{($($_:tt)*)=>{}}


fn get_time()->f64{
    static mut START:f64=-1.;
    let time=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    unsafe{
        if START<0.{
            START=time;
        }

        #[cfg(local)]{
            (time-START)*1.5
        }
        #[cfg(not(local))]{
            time-START
        }
    }
}


#[allow(unused)]
mod nth{
    use std::cmp;
    use std::mem::{self, MaybeUninit};
    use std::ptr;

    struct CopyOnDrop<T> {
        src: *const T,
        dest: *mut T,
    }

    impl<T> Drop for CopyOnDrop<T> {
        fn drop(&mut self) {
            unsafe {
                ptr::copy_nonoverlapping(self.src, self.dest, 1);
            }
        }
    }

    fn shift_tail<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = v.len();
        unsafe {
            if len >= 2 && is_less(v.get_unchecked(len - 1), v.get_unchecked(len - 2)) {
                let tmp = mem::ManuallyDrop::new(ptr::read(v.get_unchecked(len - 1)));
                let v = v.as_mut_ptr();
                let mut hole = CopyOnDrop { src: &*tmp, dest: v.add(len - 2) };
                ptr::copy_nonoverlapping(v.add(len - 2), v.add(len - 1), 1);

                for i in (0..len - 2).rev() {
                    if !is_less(&*tmp, &*v.add(i)) {
                        break;
                    }

                    ptr::copy_nonoverlapping(v.add(i), v.add(i + 1), 1);
                    hole.dest = v.add(i);
                }
            }
        }
    }

    fn insertion_sort<T, F>(v: &mut [T], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        for i in 1..v.len() {
            shift_tail(&mut v[..i + 1], is_less);
        }
    }

    fn partition_in_blocks<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
    where
        F: FnMut(&T, &T) -> bool,
    {
        const BLOCK: usize = 128;

        let mut l = v.as_mut_ptr();
        let mut block_l = BLOCK;
        let mut start_l = ptr::null_mut();
        let mut end_l = ptr::null_mut();
        let mut offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

        let mut r = unsafe { l.add(v.len()) };
        let mut block_r = BLOCK;
        let mut start_r = ptr::null_mut();
        let mut end_r = ptr::null_mut();
        let mut offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

        fn width<T>(l: *mut T, r: *mut T) -> usize {
            assert!(mem::size_of::<T>() > 0);
            (r as usize - l as usize) / mem::size_of::<T>()
        }

        loop {
            let is_done = width(l, r) <= 2 * BLOCK;

            if is_done {
                let mut rem = width(l, r);
                if start_l < end_l || start_r < end_r {
                    rem -= BLOCK;
                }

                if start_l < end_l {
                    block_r = rem;
                } else if start_r < end_r {
                    block_l = rem;
                } else {
                    block_l = rem / 2;
                    block_r = rem - block_l;
                }
                debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
                debug_assert!(width(l, r) == block_l + block_r);
            }

            if start_l == end_l {
                start_l = offsets_l.as_mut_ptr() as *mut _;
                end_l = start_l;
                let mut elem = l;

                for i in 0..block_l {
                    unsafe {
                        *end_l = i as u8;
                        end_l = end_l.offset(!is_less(&*elem, pivot) as isize);
                        elem = elem.offset(1);
                    }
                }
            }

            if start_r == end_r {
                start_r = offsets_r.as_mut_ptr() as *mut _;
                end_r = start_r;
                let mut elem = r;

                for i in 0..block_r {
                    unsafe {
                        elem = elem.offset(-1);
                        *end_r = i as u8;
                        end_r = end_r.offset(is_less(&*elem, pivot) as isize);
                    }
                }
            }

            let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

            if count > 0 {
                macro_rules! left {
                    () => {
                        l.offset(*start_l as isize)
                    };
                }
                macro_rules! right {
                    () => {
                        r.offset(-(*start_r as isize) - 1)
                    };
                }

                unsafe {
                    let tmp = ptr::read(left!());
                    ptr::copy_nonoverlapping(right!(), left!(), 1);

                    for _ in 1..count {
                        start_l = start_l.offset(1);
                        ptr::copy_nonoverlapping(left!(), right!(), 1);
                        start_r = start_r.offset(1);
                        ptr::copy_nonoverlapping(right!(), left!(), 1);
                    }

                    ptr::copy_nonoverlapping(&tmp, right!(), 1);
                    mem::forget(tmp);
                    start_l = start_l.offset(1);
                    start_r = start_r.offset(1);
                }
            }

            if start_l == end_l {
                l = unsafe { l.offset(block_l as isize) };
            }

            if start_r == end_r {
                r = unsafe { r.offset(-(block_r as isize)) };
            }

            if is_done {
                break;
            }
        }

        if start_l < end_l {
            debug_assert_eq!(width(l, r), block_l);
            while start_l < end_l {
                unsafe {
                    end_l = end_l.offset(-1);
                    ptr::swap(l.offset(*end_l as isize), r.offset(-1));
                    r = r.offset(-1);
                }
            }
            width(v.as_mut_ptr(), r)
        } else if start_r < end_r {
            debug_assert_eq!(width(l, r), block_r);
            while start_r < end_r {
                unsafe {
                    end_r = end_r.offset(-1);
                    ptr::swap(l, r.offset(-(*end_r as isize) - 1));
                    l = l.offset(1);
                }
            }
            width(v.as_mut_ptr(), l)
        } else {
            width(v.as_mut_ptr(), l)
        }
    }

    fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> (usize, bool)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let (mid, was_partitioned) = {
            v.swap(0, pivot);
            let (pivot, v) = v.split_at_mut(1);
            let pivot = &mut pivot[0];

            let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
            let _pivot_guard = CopyOnDrop { src: &*tmp, dest: pivot };
            let pivot = &*tmp;

            let mut l = 0;
            let mut r = v.len();

            unsafe {
                while l < r && is_less(v.get_unchecked(l), pivot) {
                    l += 1;
                }

                while l < r && !is_less(v.get_unchecked(r - 1), pivot) {
                    r -= 1;
                }
            }

            (l + partition_in_blocks(&mut v[l..r], pivot, is_less), l >= r)
        };

        v.swap(0, mid);

        (mid, was_partitioned)
    }

    fn partition_equal<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> usize
    where
        F: FnMut(&T, &T) -> bool,
    {
        v.swap(0, pivot);
        let (pivot, v) = v.split_at_mut(1);
        let pivot = &mut pivot[0];

        let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
        let _pivot_guard = CopyOnDrop { src: &*tmp, dest: pivot };
        let pivot = &*tmp;

        let mut l = 0;
        let mut r = v.len();
        loop {
            unsafe {
                while l < r && !is_less(pivot, v.get_unchecked(l)) {
                    l += 1;
                }

                while l < r && is_less(pivot, v.get_unchecked(r - 1)) {
                    r -= 1;
                }

                if l >= r {
                    break;
                }

                r -= 1;
                let ptr = v.as_mut_ptr();
                ptr::swap(ptr.add(l), ptr.add(r));
                l += 1;
            }
        }

        l + 1
    }

    fn choose_pivot<T, F>(v: &mut [T], is_less: &mut F) -> (usize, bool)
    where
        F: FnMut(&T, &T) -> bool,
    {
        const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;
        const MAX_SWAPS: usize = 4 * 3;

        let len = v.len();

        let mut a = len / 4 * 1;
        let mut b = len / 4 * 2;
        let mut c = len / 4 * 3;

        let mut swaps = 0;

        if len >= 8 {
            let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
                if is_less(v.get_unchecked(*b), v.get_unchecked(*a)) {
                    ptr::swap(a, b);
                    swaps += 1;
                }
            };

            let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
                sort2(a, b);
                sort2(b, c);
                sort2(a, b);
            };

            if len >= SHORTEST_MEDIAN_OF_MEDIANS {
                let mut sort_adjacent = |a: &mut usize| {
                    let tmp = *a;
                    sort3(&mut (tmp - 1), a, &mut (tmp + 1));
                };

                sort_adjacent(&mut a);
                sort_adjacent(&mut b);
                sort_adjacent(&mut c);
            }

            sort3(&mut a, &mut b, &mut c);
        }

        if swaps < MAX_SWAPS {
            (b, swaps == 0)
        } else {
            v.reverse();
            (len - 1 - b, true)
        }
    }


    fn partition_at_index_loop<'a, T, F>(
        mut v: &'a mut [T],
        mut index: usize,
        is_less: &mut F,
        mut pred: Option<&'a T>,
    ) where
        F: FnMut(&T, &T) -> bool,
    {
        loop {
            const MAX_INSERTION: usize = 10;
            if v.len() <= MAX_INSERTION {
                insertion_sort(v, is_less);
                return;
            }

            let (pivot, _) = choose_pivot(v, is_less);

            if let Some(p) = pred {
                if !is_less(p, &v[pivot]) {
                    let mid = partition_equal(v, pivot, is_less);

                    if mid > index {
                        return;
                    }

                    v = &mut v[mid..];
                    index = index - mid;
                    pred = None;
                    continue;
                }
            }

            let (mid, _) = partition(v, pivot, is_less);

            let (left, right) = v.split_at_mut(mid);
            let (pivot, right) = right.split_at_mut(1);
            let pivot = &pivot[0];

            if mid < index {
                v = right;
                index = index - mid - 1;
                pred = Some(pivot);
            } else if mid > index {
                v = left;
            } else {
                return;
            }
        }
    }

    fn partition_at_index<T, F>(
        v: &mut [T],
        index: usize,
        mut is_less: F,
    ) -> (&mut [T], &mut T, &mut [T])
    where
        F: FnMut(&T, &T) -> bool,
    {
        use cmp::Ordering::Greater;
        use cmp::Ordering::Less;

        if index >= v.len() {
            panic!("partition_at_index index {} greater than length of slice {}", index, v.len());
        }

        if mem::size_of::<T>() == 0 {
        } else if index == v.len() - 1 {
            let (max_index, _) = v
                .iter()
                .enumerate()
                .max_by(|&(_, x), &(_, y)| if is_less(x, y) { Less } else { Greater })
                .unwrap();
            v.swap(max_index, index);
        } else if index == 0 {
            let (min_index, _) = v
                .iter()
                .enumerate()
                .min_by(|&(_, x), &(_, y)| if is_less(x, y) { Less } else { Greater })
                .unwrap();
            v.swap(min_index, index);
        } else {
            partition_at_index_loop(v, index, &mut is_less, None);
        }

        let (left, right) = v.split_at_mut(index);
        let (pivot, right) = right.split_at_mut(1);
        let pivot = &mut pivot[0];
        (left, pivot, right)
    }

    pub fn select_nth_unstable_by_key<T, K, F>(
        slice:&mut [T],
        index: usize,
        mut f: F,
    ) -> (&mut [T], &mut T, &mut [T])
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        let mut g = |a: &T, b: &T| f(a).lt(&f(b));
        partition_at_index(slice, index, &mut g)
    }
}



mod rnd {
    pub fn next()->usize{
        static mut SEED:usize=88172645463325252;
        unsafe{
            SEED^=SEED<<7;
            SEED^=SEED>>9;
            SEED
        }
    }
}
