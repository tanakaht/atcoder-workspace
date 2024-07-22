#!/bin/bash
PATH="$HOME/.cargo/bin:$PATH"
problem_name=$1
test_dir=./testcases/${problem_name}
base_url=${problem_name%_*}

# make test directory
if [ ! -e ${test_dir} ]; then
    oj dl -d ${test_dir} https://atcoder.jp/contests/${base_url}/tasks/${problem_name//-/_}
fi

# compile
cargo build --manifest-path ./Cargo.toml --bin ${problem_name} || exit 1
oj test -c ./target/debug/${problem_name} -d ${test_dir}  && ./scripts/submit.sh ${problem_name}

# cargo build --release --manifest-path ./Cargo.toml --bin ${problem_name} || exit 1
# oj test -c ./target/release/${problem_name} -d ${test_dir}
