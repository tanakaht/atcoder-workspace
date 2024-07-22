#!/bin/bash
f=0005
cargo build --manifest-path ./Cargo.toml --bin ahc30_a --release || exit 1
# cargo build --manifest-path ./Cargo.toml --bin ahc30_a || exit 1
# ./target/debug/ahc30_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run -r --bin tester ../target/release/ahc30_a  < "./in/${f}.txt" > "./out/${f}.txt"
# set RUST_BACKTRACE=1
# cd ..
# RUST_BACKTRACE=1 cargo run --bin ahc30_a < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
# RUST_BACKTRACE=1 cargo run -r --bin tester ../target/release/ahc30_a   < "./in/${f}.txt" > "./out/${f}.txt"
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html

#  f=0040
# ./tools/target/release/tester cargo flamegraph --bin ahc30_a < "./tools/in/${f}.txt"
