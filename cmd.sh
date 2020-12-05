python train_transformer.py \
--train_data lmdb_data/train \
--valid_data lmdb_data/valid \
--batch_size 8 \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling Transformer \
--Prediction None
