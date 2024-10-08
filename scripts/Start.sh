#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /workspace/ls-poc/

python src/models/anomalydetection/Anomaly-Transformer/main.py --mode train --batch_size 256 --dataset LSDummyX --data_path data/01-simple-dataset --anormly_ratio 1 --input_c 6 --output_c 6 --lr 0.0001 --num_epochs 3 --pretrained_model 20 --win_size 60
python src/models/anomalydetection/Anomaly-Transformer/main.py --mode test  --batch_size 256  --dataset LSDummyX --data_path data/01-simple-dataset --anormly_ratio 1  --input_c 6 --output_c 6

python src/models/anomalydetection/Anomaly-Transformer/main.py --mode train --batch_size 256 --dataset LSDummyY --data_path data/01-simple-dataset --anormly_ratio 1 --input_c 4 --output_c 4 --lr 0.0001 --num_epochs 3 --pretrained_model 20 --win_size 60
python src/models/anomalydetection/Anomaly-Transformer/main.py --mode test  --batch_size 256  --dataset LSDummyY --data_path data/01-simple-dataset --anormly_ratio 1  --input_c 4 --output_c 4