#!/bin/bash
f=0000
cargo build --manifest-path ./Cargo.toml --bin ahc029_a --release || exit 1
./tools/target/release/tester ./target/release/ahc029_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
