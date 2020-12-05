python train_transformer.py \
--train_data lmdb_data/train \
--valid_data lmdb_data/valid \
--saved_model pretrained_model/CTC2.pth \
--batch_size 8 \
--Transformation None \
--FeatureExtraction ResNet \
--SequenceModeling Transformer \
--Prediction None
