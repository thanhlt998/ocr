import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TransformerConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from label_smoothing_loss import LabelSmoothingLoss
from test import validation
from collections import OrderedDict
import re
from tqdm import tqdm
import json
from radam import RAdam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    elif opt.Prediction == 'None':
        converter = TransformerConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    model.train()
    if opt.load_from_checkpoint:
        model.load_state_dict(torch.load(os.path.join(opt.load_from_checkpoint, 'checkpoint.pth')))
        print(f'loaded checkpoint from {opt.load_from_checkpoint}...')
    elif opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.SequenceModeling == 'Transformer':
            fe_state = OrderedDict()
            state_dict = torch.load(opt.saved_model)
            for k, v in state_dict.items():
                if k.startswith('module.FeatureExtraction'):
                    new_k = re.sub('module.FeatureExtraction.', '', k)
                    fe_state[new_k] = state_dict[k]
            model.FeatureExtraction.load_state_dict(fe_state)
        else:
            if opt.FT:
                model.load_state_dict(torch.load(opt.saved_model), strict=False)
            else:
                model.load_state_dict(torch.load(opt.saved_model))
    if opt.freeze_fe:
        model.freeze(['FeatureExtraction'])
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    elif opt.Prediction == 'None':
        criterion = LabelSmoothingLoss(classes=converter.n_classes, padding_idx=converter.pad_idx, smoothing=0.1)
        # criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.pad_idx)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        assert opt.adam in ['Adam', 'AdamW', 'RAdam'], 'adam optimizer must be in Adam, AdamW or RAdam'
        if opt.adam == 'Adam':
            optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.adam == "AdamW":
            optimizer = optim.AdamW(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            optimizer = RAdam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    if opt.load_from_checkpoint and opt.load_optimizer_state:
        optimizer.load_state_dict(torch.load(os.path.join(opt.load_from_checkpoint, 'optimizer.pth')))
        print(f'loaded optimizer state from {os.path.join(opt.load_from_checkpoint, "optimizer.pth")}')

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    if opt.load_from_checkpoint:
        with open(os.path.join(opt.load_from_checkpoint, 'iter.json'), mode='r', encoding='utf8') as f:
            start_iter = json.load(f)
            print(f'continue to train, start_iter: {start_iter}')
            f.close()

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    # i = start_iter

    bar = tqdm(range(start_iter, opt.num_iter))
    # while(True):
    for i in bar:
        bar.set_description(f'Iter {i}: train_loss = {loss_avg.val():.5f}')
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)

            # (ctc_a) For PyTorch 1.2.0 and 1.3.0. To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True

            # # (ctc_b) To reproduce our pretrained model / paper, use our previous code (below code) instead of (ctc_a).
            # # With PyTorch 1.2.0, the below code occurs NAN, so you may use PyTorch 1.1.0.
            # # Thus, the result of CTCLoss is different in PyTorch 1.1.0 and PyTorch 1.2.0.
            # # See https://github.com/clovaai/deep-text-recognition-benchmark/issues/56#issuecomment-526490707
            # cost = criterion(preds, text, preds_size, length)

        elif opt.Prediction == 'None':
            tgt_input = text['tgt_input']
            tgt_output = text['tgt_output']
            tgt_padding_mask = text['tgt_padding_mask']
            preds = model(image, tgt_input.transpose(0, 1), tgt_key_padding_mask=tgt_padding_mask,)
            cost = criterion(preds.view(-1, preds.shape[-1]), tgt_output.contiguous().view(-1))
        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (i + 1) % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')

                # checkpoint
                os.makedirs(f'./checkpoints/{opt.experiment_name}/', exist_ok=True)

                torch.save(model.state_dict(), f'./checkpoints/{opt.experiment_name}/checkpoint.pth')
                torch.save(optimizer.state_dict(), f'./checkpoints/{opt.experiment_name}/optimizer.pth')
                with open(f'./checkpoints/{opt.experiment_name}/iter.json', mode='w', encoding='utf8') as f:
                    json.dump(i + 1, f)
                    f.close()

                with open(f'./checkpoints/{opt.experiment_name}/checkpoint.log', mode='a', encoding='utf8') as f:
                    f.write(f'Saved checkpoint with iter={i}\n')
                    f.write(f'\tCheckpoint at: ./checkpoints/{opt.experiment_name}/checkpoint.pth')
                    f.write(f'\tOptimizer at: ./checkpoints/{opt.experiment_name}/optimizer.pth')

                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (i + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        # if i == opt.num_iter:
        #     print('end the training')
        #     sys.exit()
        # i += 1
        # if i == 1: break
    print('end training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    # parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--adam', type=str, default="", help='Whether to use adam (default is Adadelta), if specified: Adam|AdamW|RAdam')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/', 
                     help='select training data (default is MJ-ST, which means MJ and ST used as training data)') 
    parser.add_argument('--batch_ratio', type=str, default='1', 
                     help='assign ratio for each selected data in the batch') 
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=32, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=48, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=180, help='the width of the input image')
    parser.add_argument('--rgb', default=True, action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        # default=" :#@()'!/*,.qwertyuiopasdfghjklzxcvbnm1234567890QWERTYUIOPASDFGHJKLZXCVBNM",
                        default=" 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()-*!:#.,'/",
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM|Transformer')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn|None')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # transformer sequence modeling params
    parser.add_argument('--d_model', type=int, default=256, help='d_model of transformer sequence modeling')
    parser.add_argument('--nhead', type=int, default=8, help='nhead of transformer sequence modeling')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='num_encoder_layers of transformer sequence modeling')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='num_decoder_layers of transformer sequence modeling')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dim_feedforward of transformer sequence modeling')
    parser.add_argument('--max_seq_length', type=int, default=256, help='max_seq_length of transformer sequence modeling')
    parser.add_argument('--pos_dropout', type=float, default=0.1, help='pos_dropout of transformer sequence modeling')
    parser.add_argument('--trans_dropout', type=float, default=0.1, help='trans_dropout of transformer sequence modeling')
    parser.add_argument('--freeze_fe', action='store_true', help='freeze feature extraction module')
    parser.add_argument('--beam_search', action='store_true', help='use beam search')
    parser.add_argument('--load_optimizer_state', action='store_true', help='use beam search')
    parser.add_argument('--load_from_checkpoint', type=str, help='continue training from checkpoint')


    opt = parser.parse_args()

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
