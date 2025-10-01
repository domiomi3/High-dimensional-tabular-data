#!/bin/bash
for i in {100..1700..100}
do
    bash experiments/generate_slurm_scripts.sh "tabpfnv2_tab" "hiva" "random_fs" -g scalability_study --run_id complete -c -- --unique_save --num_features $i
    sbatch /work/dlclarge2/matusd-toy_example/experiments/slurm_scripts/scalability_study/complete/hiva_agnostic/tabpfnv2_tab/fs_r_unique_save_num_features_$i.sh
done


