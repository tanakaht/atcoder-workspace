cargo build --manifest-path ./Cargo.toml --bin toyota_hc_2023spring_a_a --release || exit 1
echo 1
DATE=`date '+%Y-%m-%d-%H:%M'`
echo 1
#read testname
testname=$DATE
echo 1
cp ./src_rust/ahc018_a.rs ./results/src/${testname}.rs
echo 1
cd ./tools
echo 1
psytester r -t 100 $testname
psytester r ../results
psytester s
cd ..
