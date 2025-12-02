#!/bin/bash

cd ./src;
T=1000 ;
pb=0.0;
gtype='poisson';
datapath="../data_${gtype}_10"
M=1
radius=0.0
max_jobs=10
size=100

for M in 1 3; do
    for load_streaming in 9 8 7 6 5 4 3 2 1 ; do
        echo "submit dt, radius ${radius}, size ${size}, load_streaming ${load_streaming}";
        python spdt_sch_optimization_gating.py --datapath=${datapath} --pburst=${pb} --out=../output\
            --radius=${radius} --gtype=${gtype} --T=${T} --sizes=${size} --ls=${load_streaming} --M=${M} &
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done
done


