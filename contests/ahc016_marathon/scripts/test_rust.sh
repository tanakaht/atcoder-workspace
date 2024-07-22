#!/bin/bash
cargo build --manifest-path ./Cargo.toml --bin ahc016_a || exit 1
./target/debug/ahc016_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
