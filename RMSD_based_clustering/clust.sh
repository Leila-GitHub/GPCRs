#!/bin/bash


dir=10000_TMs_bootstrap_all_frames_included_1

cp clustering_cb1_TMs.ipynb $dir
cd $dir

echo 'START Date & Time:'
date +'%m/%d/%Y'
date +'%r'
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=-1 clustering_cb1_TMs.ipynb --output clustering_cb1_TMs_output.ipynb
echo 'END Date & Time:'
date +'%m/%d/%Y'
date +'%r'

rm -rf clustering_cb1_TMs.ipynb
