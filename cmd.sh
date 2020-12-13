python create_lmdb_dataset.py \
--inputPath data/train \
--gtFile data/train/image_list.txt \
--outputPath lmdb_data/train

python create_lmdb_dataset.py \
--inputPath data/valid \
--gtFile data/valid/image_list.txt \
--outputPath lmdb_data/valid

python train_transformer.py \
--train_data lmdb_data/train \
--valid_data lmdb_data/valid \
--saved_model pretrained_model/CTC2.pth \
--batch_size 32 \
--Transformation None \
--FeatureExtraction ResNet \
--SequenceModeling Transformer \
--Prediction None \
--sensitive \
--adam \
--lr 0.01

python test.py \
--eval_data lmdb_data/valid \
--saved_model saved_models/None-ResNet-Transformer-None-Seed1111/best_accuracy.pth \
--imgH 48 \
--imgW 100 \
--data_filtering_off \
--sensitive \
--PAD \
--rgb \
--Transformation None \
--FeatureExtraction ResNet \
--SequenceModeling Transformer \
--Prediction None \
--dim_feedforward 1024 \
--pos_dropout 0.1 \
--trans_dropout 0.1 \
--d_model 512 \
--num_encoder_layers 4 \
--num_decoder_layers 4 \
--nhead 4



