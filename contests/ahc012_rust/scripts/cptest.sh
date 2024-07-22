#!/bin/bash
PATH="$HOME/.cargo/bin:$PATH"
problem_name=$1

cargo build --manifest-path ./Cargo.toml --bin ahc012_a || exit 1
./target/debug/ahc012_a < ./testcases/input.txt > ./testcases/output.txt
