cargo build --manifest-path ./Cargo.toml --bin ahc018_a --release || exit 1
cd ./tools
psytester r -t 2
cd ..
