#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ltp_lstm
#SBATCH --mem=32000
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4
python ./source/lstm_with_sequence.py >> .out/log.txt
mv *.out out/
