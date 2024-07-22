#!/bin/bash
f=0003
cargo build --manifest-path ./Cargo.toml --bin huawei_2023_icpc_a --release || exit 1
./target/release/huawei_2023_icpc_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
python3 tester.py "./in/${f}.txt" "./out/${f}.txt"
cd ..
