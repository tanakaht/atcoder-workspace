#!/bin/bash
f=0000
cargo build --manifest-path ./Cargo.toml --bin ahc020_a --release || exit 1
./target/release/ahc020_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
