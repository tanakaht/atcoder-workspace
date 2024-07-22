#!/bin/bash
ftype=A
f=0000
cargo build --manifest-path ./Cargo.toml --bin masters2024-final-open_a || exit 1
cd ./tools
cargo run -r --bin tester ../target/debug/masters2024-final-open_a  < "./in${ftype}/${f}.txt" > "./out${ftype}/${f}.txt"
