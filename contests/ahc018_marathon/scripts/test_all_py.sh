cd ./tools
DATE=`date '+%Y-%m-%d-%H:%M'`
#read testname
testname=$DATE
psytester r -t 10 $testname
psytester s
cd ..
