#!/bin/bash
oj login https://atcoder.jp/

# rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
PATH="$HOME/.cargo/bin:$PATH"
rustup component add rls rust-analysis rust-src

sudo apt update
sudo apt upgrade -y
sudo apt install
sudo apt install linux-tools-virtual -y
cargo install flamegraph
