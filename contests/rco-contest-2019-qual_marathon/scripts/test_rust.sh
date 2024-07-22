#!/bin/bash
probname=B
cargo build --manifest-path ./Cargo.toml --bin rco-contest-2019-qual_b || exit 1
./target/debug/rco-contest-2019-qual_b  < "./input_${probname}.txt" > "./output_${probname}.txt"
