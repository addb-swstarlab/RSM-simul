#!/bin/bash

if [ $# -ne 10 ]; then
  echo "./start_bench.sh help"
  echo "#PARAMETER < 10"
  echo "[1]num-unique-keys: 100000, ..."
  echo "[2]active-key-mode: 0,1,2"
  echo "[3]dependency-mode: 0,1,2,3"
  echo "[4]num-request: 10000000, ..."
  echo "[5]zipf-theta: 0.00,0.99, ..."
  echo "[6]compaction mode: 0,1,2,3,...,12"
  echo "[7]wb-size: 4194304, ..."
  echo "[8]enable-fsync: 0,1"
  echo "[9]use-custome-size: 0,1"
  echo "[10]episode-num: 100, ..."
  exit 1
fi

echo "num-unique-keys: ${1}"
echo "active-key-mode: ${2}"
echo "dependency-mode: ${3}"
echo "num-request: ${4}"
echo "zipf-theta: ${5}"
echo "compaction mode: ${6}"
echo "wb-size: ${7}"
echo "enable-fsync: ${8}"
echo "use-custome-size: ${9}"
echo "episode-num: ${10}"

if [ ${6} -eq 11 ]; then
  for ((i=0; i<${10}; i++))
  do
    echo "=================${i}-th episode execution================="
    ./main leveldb-sim $1 $2 $3 $4 $5 $6 $7 $8 $9 > kRSMPolicy_train.txt
  done
else
   ./main leveldb-sim $1 $2 $3 $4 $5 $6 $7 $8 $9 > kRSMPolicy_evaluate.txt
fi

