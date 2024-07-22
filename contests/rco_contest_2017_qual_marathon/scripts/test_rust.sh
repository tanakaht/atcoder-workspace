#!/bin/bash
set RUST_BACKTRACE=1
cargo run --manifest-path ./Cargo.toml --bin rco_contest_2017_qual_a --release < "./input.txt" > "./output.txt"
# cargo flamegraph --manifest-path ./Cargo.toml --bin rco_contest_2017_qual_a --release < "./input.txt" > "./output.txt"
# cargo build --manifest-path ./Cargo.toml --bin rco_contest_2017_qual_a || exit 1
# ./target/debug/rco_contest_2017_qual_a  < "./input.txt" > "./output.txt"
