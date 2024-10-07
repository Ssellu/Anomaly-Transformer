export CUDA_VISIBLE_DEVICES=0
cd /workspace
python ls-poc/src/models/anomalydetection/Anomaly-Transformer/main.py --mode train --batch_size 256 --dataset LSDummyX --data_path ls-poc/data/01-simple-dataset --anormly_ratio 1 --input_c 6 --output_c 6 --lr 0.0001 --num_epochs 3 --pretrained_model 20 --win_size 60
python ls-poc/src/models/anomalydetection/Anomaly-Transformer/main.py --mode test  --batch_size 256  --dataset LSDummyX --data_path ls-poc/data/01-simple-dataset --anormly_ratio 1  --input_c 6 --output_c 6

python ls-poc/src/models/anomalydetection/Anomaly-Transformer/main.py --mode train --batch_size 256 --dataset LSDummyY --data_path ls-poc/data/01-simple-dataset --anormly_ratio 1 --input_c 3 --output_c 3 --lr 0.0001 --num_epochs 3 --pretrained_model 20 --win_size 60
python ls-poc/src/models/anomalydetection/Anomaly-Transformer/main.py --mode test  --batch_size 256  --dataset LSDummyY --data_path ls-poc/data/01-simple-dataset --anormly_ratio 1  --input_c 3 --output_c 3

