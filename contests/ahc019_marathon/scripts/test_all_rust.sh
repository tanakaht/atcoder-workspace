cargo build --manifest-path ./Cargo.toml --bin ahc019_a --release || exit 1
DATE=`date '+%Y-%m-%d-%H:%M'`
echo "enter testname"
read testname
#testname=$DATE
cp ./src_rust/ahc019_a.rs ./results/src/${testname}.rs
cd ./tools
psytester r -t 100 $testname
psytester r ../results
psytester s
cd ..
