cargo build --manifest-path ./Cargo.toml --bin toyota_hc_2023spring_a --release || exit 1
./target/release/toyota_hc_2023spring_a < ./tools/in/0000.txt
