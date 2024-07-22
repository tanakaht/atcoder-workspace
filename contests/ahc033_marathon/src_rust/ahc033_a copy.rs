let mut order = vec![];
// 優先度1: コンテナを持っているクレーン
for (i, crane) in cranes.iter().enumerate(){
    if crane.has_item{
        order.push(i);
    }
}
// 優先度2: メインレーンにいるクレーン
for (i, crane) in cranes.iter().enumerate(){
    if !crane.has_item && (crane.p.col==1 || crane.p.col==4 || (crane.p.row==4&&(crane.p.col==2 || crane.p.col==3))){
        order.push(i);
    }
}
// 優先度3: タスクあるクレーン
for (i, crane) in cranes.iter().enumerate(){
    if crane.task.status<3 && !order.contains(&i){
        order.push(i);
    }
}
// 優先度4: タスクないクレーン
for (i, crane) in cranes.iter().enumerate(){
    if !order.contains(&i){
        order.push(i);
    }
}
// コンフリクト解消
loop{
    let mut place2_crane_idx: HashMap<Coordinate, usize> = HashMap::new();
    let mut flg = true;
    for &i in order.iter(){
        let crane = &cranes[i];
        if next_op_candidate[i].is_empty(){
            eprintln!("cand: {:?}", next_op_candidate);
            for x in 0..5{
                eprintln!("{}:{:?} {:?} {:?}", x, cranes[x].p, cranes[x].task, cranes[x].get_op_candidate());
            }
            continue;
        }
        let op = next_op_candidate[i].pop().unwrap();
        let cd = match op{
            0 => CoordinateDiff::new(!0, 0),
            1 => CoordinateDiff::new(0, 1),
            2 => CoordinateDiff::new(1, 0),
            3 => CoordinateDiff::new(0, !0),
            _ => CoordinateDiff::new(0, 0),
        };
        let mut p = Coordinate::new(crane.p.row*2, crane.p.col*2)+cd;
        if let std::collections::hash_map::Entry::Vacant(e) = place2_crane_idx.entry(p) {
            e.insert(i);
        } else {
            let j = place2_crane_idx[&p];
            if j!=i{
                if next_op_candidate[i].is_empty(){
                    next_op_candidate[i].push(op);
                    next_op_candidate[j].pop();
                }
                flg = false;
                break;
            }
        }
        p = p+cd;
        if let std::collections::hash_map::Entry::Vacant(e) = place2_crane_idx.entry(p) {
            e.insert(i);
        } else {
            let j = place2_crane_idx[&p];
            if j!=i{
                if next_op_candidate[i].is_empty(){
                    next_op_candidate[i].push(op);
                    next_op_candidate[j].pop();
                }
                flg = false;
                break;
            }
        }
        next_op_candidate[i].push(op);
    }
    if flg{
        break;
    }
}
