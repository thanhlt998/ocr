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
--Prediction None
