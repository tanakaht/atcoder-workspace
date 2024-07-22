cargo build --manifest-path ./Cargo.toml --bin toyota_hc_2023spring_a --release || exit 1
DATE=`date '+%Y-%m-%d-%H:%M'`
echo "enter testname"
read testname
#testname=$DATE
cp ./src_rust/toyota_hc_2023spring_a.rs ./results/src/${testname}.rs
cd ./tools
psytester r -t 100 $testname
psytester r ../results
psytester s
cd ..
