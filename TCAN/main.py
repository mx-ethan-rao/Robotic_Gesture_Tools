import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

from model import TCANet, MultiModalNet
from utils import RawDataset, KinematicDataset, VisualDataset, TwoStreamDataset, load_data, load_dataloader, \
    save_model, load_model, save_visual_info, record, draw_attn, Wandb_logger
from config import Config, config

from IPython import embed

import logging
logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def train(args):
    logging.info("start load parameters.")
    torch.manual_seed(args.seed)
    if args.signal_type == 'kinematic':
        num_chans = [args.emsize] * args.levels
    elif args.signal_type == 'visual':
        num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
    else:
        k_num_chans = [args.emsize] * args.levels
        v_num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
    # logger = SummaryWriter(args.dir_log)
    logger = Wandb_logger(dict(args))

    # load data
    logging.info("start load {} dataset.".format(args.dataset_name))

    assert args.signal_type in ["kinematic", "visual", "both"], "signal_type must be \{'kinematic', 'visual', 'both'\}"

    if args.signal_type == 'kinematic':
        train_dataset = KinematicDataset(args.dir_data_root, args.dataset_name, 'train', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        valid_dataset = KinematicDataset(args.dir_data_root, args.dataset_name, 'valid', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        test_dataset = KinematicDataset(args.dir_data_root, args.dataset_name, 'test', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
    elif args.signal_type == 'visual':
        train_dataset = VisualDataset(args.dir_data_root, args.dataset_name, 'train', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        valid_dataset = VisualDataset(args.dir_data_root, args.dataset_name, 'valid', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        test_dataset = VisualDataset(args.dir_data_root, args.dataset_name, 'test', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
    elif args.signal_type == 'both':
        train_dataset = TwoStreamDataset(args.dir_data_root, args.dataset_name, 'train', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        valid_dataset = TwoStreamDataset(args.dir_data_root, args.dataset_name, 'valid', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
        test_dataset = TwoStreamDataset(args.dir_data_root, args.dataset_name, 'test', args.fps, args.seq_len, args.valid_len, args.train_scheme, args.permute)
    # train_dataset = RawDataset(args.dir_data_root, args.dataset_name, 'train', args.seq_len, args.valid_len, args.is_corpus, args.permute)
    # valid_dataset = RawDataset(args.dir_data_root, args.dataset_name, 'valid', args.seq_len, args.valid_len, args.is_corpus, args.permute)
    # test_dataset = RawDataset(args.dir_data_root, args.dataset_name, 'test', args.seq_len, args.valid_len, args.is_corpus, args.permute)
    train_dataloader = load_dataloader(train_dataset, args.signal_type, args.batch_size, num_workers=args.num_workers)
    valid_dataloader = load_dataloader(valid_dataset, args.signal_type, args.batch_size, num_workers=args.num_workers)
    test_dataloader = load_dataloader(test_dataset, args.signal_type, args.batch_size, num_workers=args.num_workers)
    # n_dict = train_dataset.n_dict
    logging.info("end -------------")

    # define model
    logging.info("start load model.")
    if args.signal_type != 'both':
        model = TCANet(args.emsize, args.num_gestures, num_chans, args.visual_emsize, args.valid_len, args.num_subblocks, temp_attn=args.temp_attn, nheads=args.nheads,
                en_res=args.en_res, conv=args.conv, dropout=args.dropout, emb_dropout=args.emb_dropout, key_size=args.key_size, 
                kernel_size=args.ksize, tied_weights=args.tied, signal_type=args.signal_type, visual=args.visual)
    else:
        model = MultiModalNet(args.emsize, args.num_gestures, k_num_chans, v_num_chans, args.visual_emsize, args.valid_len, args.num_subblocks, temp_attn=args.temp_attn, nheads=args.nheads,
                en_res=args.en_res, conv=args.conv, dropout=args.dropout, emb_dropout=args.emb_dropout, key_size=args.key_size, 
                kernel_size=args.ksize, tied_weights=args.tied, signal_type=args.signal_type, visual=args.visual)

    num_parameters_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of parameters = {}".format(num_parameters_train))

    if args.cuda:
        model.cuda(args.gpu_id)
    if args.is_parallel:
        model = nn.DataParallel(model)
        logging.info("The model is training with nn.DataParallel.")
    if args.continue_train:
        model = load_model(model, args)
        logging.info("Continue training, load saved model.")

    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    visual_info_all = []
    best_vloss = 1e8
    
    # start training
    logging.info("start training.")
    try:
        all_vloss = []
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            model.train() 
            loss_sum = 0
            processed_data_size = 0
            correct_total = 0
            total = 0
            for i, (train_batch, label_batch) in enumerate(tqdm(train_dataloader, ncols=80)):
                optimizer.zero_grad()
                if args.signal_type != 'both':
                    train_batch = train_batch.cuda(args.gpu_id)
                    label_batch = label_batch.cuda(args.gpu_id)
                else:
                    train_batch = (train_batch[0].cuda(args.gpu_id), train_batch[1].cuda(args.gpu_id))
                    label_batch = label_batch.cuda(args.gpu_id)
                if args.signal_type != 'both':    
                    if args.temp_attn:
                        output_batch, attn_weight_list = model(train_batch)
                        if i == 1:
                            visual_info = [train_batch, label_batch, attn_weight_list]
                            
                    else:
                        output_batch, _ = model(train_batch)
                else:
                    output_batch, k_output_batch, v_output_batch = model(train_batch, label_batch, 'train')


                # Discard the effective history part
                eff_history = args.seq_len - args.valid_len
                if eff_history < 0:
                    raise ValueError("Valid sequence length must be smaller than sequence length!")
                
                if args.signal_type != 'mnist':
                    # label_batch = label_batch[:, eff_history:].contiguous().view(-1)
                    # output_batch = output_batch[:, eff_history:].contiguous().view(-1, args.num_gestures)
                    pred = output_batch.data.max(1, keepdim=True)[1]
                    correct_total += pred.eq(label_batch.data.view_as(pred)).cpu().sum()
                    total += label_batch.size(0)
                else:
                    pred = output_batch.data.max(1, keepdim=True)[1]
                    correct_total += pred.eq(label_batch.data.view_as(pred)).cpu().sum()
                if args.signal_type != 'both':  
                    loss_i = criterion(output_batch, label_batch)
                else:
                    loss_i = criterion(output_batch, label_batch) + 0.5 * criterion(k_output_batch, label_batch) + 0.5 * criterion(v_output_batch, label_batch)

                loss_i.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                if args.signal_type != 'both':
                    loss_sum += (train_batch.size(1) - eff_history) * loss_i.item()
                    processed_data_size += train_batch.size(1) - eff_history
                else:
                    loss_sum += loss_i.item()
                    processed_data_size += 1

            # if args.dataset_name == 'visual':
            acc_train = 100*float(correct_total)/float(total)
            loss_train = round(loss_sum/processed_data_size, 6)
            ppl_train = round(np.exp(loss_train), 4)
            epoch_end_time = time.time()

            # evaluate 
            loss_val, acc_val, ppl_val = evaluate(model, valid_dataloader, criterion, args.num_gestures, args)
            loss_test, acc_test, ppl_test = evaluate(model, test_dataloader, criterion, args.num_gestures, args)
            
            # draw sequence correlation map
            if args.temp_attn and args.visual:
                visual_info_all.append(visual_info)
                if epoch == 0:
                    draw_attn(visual_info, epoch, args, train_dataset.dictionary)
                else:
                    draw_attn(visual_info, epoch, args)


            # wandb
            if args.signal_type == 'visual':
                logging.info('| Epoch {}/{} | Time: {:.2f}s | train loss {:.2f} | train acc {:.2f} | train ppl {:.2f} | test loss {:.2f}  | test acc {:.2f} | test ppl {:.2f} |'\
                    .format(epoch+1, args.epochs, epoch_end_time-epoch_start_time, loss_train, acc_train, ppl_train, loss_test, acc_test, ppl_test))
                logger.logging_with_step('Train/loss',loss_train, step=epoch)
                logger.logging_with_step('Train/acc', acc_train, step=epoch)
                logger.logging_with_step('Test/loss', loss_test, step=epoch)
                logger.logging_with_step('Test/acc', acc_test, step=epoch)
                logger.logging_with_step('Test/ppl', ppl_test, step=epoch)
                logger.logging_with_step('Val/loss', loss_val, step=epoch)
                logger.logging_with_step('Val/acc', acc_val, step=epoch)
                logger.logging_with_step('Val/ppl', ppl_val, step=epoch)
            else:
                logging.info('| Epoch {}/{} | Time: {:.2f}s | train loss {:.2f} | train acc {:.2f} | train ppl {:.2f} | test loss {:.2f}  | test acc {:.2f} | test ppl {:.2f} |'\
                    .format(epoch+1, args.epochs, epoch_end_time-epoch_start_time, loss_train, acc_train, ppl_train, loss_test, acc_test, ppl_test))
                logger.logging_with_step('Train/loss',loss_train, step=epoch)
                logger.logging_with_step('Train/acc', acc_train, step=epoch)
                logger.logging_with_step('Test/loss', loss_test, step=epoch)
                logger.logging_with_step('Test/acc', acc_test, step=epoch)
                logger.logging_with_step('Test/ppl', ppl_test, step=epoch)
                logger.logging_with_step('Val/loss', loss_val, step=epoch)
                logger.logging_with_step('Val/acc', acc_val, step=epoch)
                logger.logging_with_step('Val/ppl', ppl_val, step=epoch)

            # Save the model if the validation loss is the best we've seen so far.
            if loss_val < best_vloss:
                save_model(model, args)
                best_vloss = loss_val

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and loss_val >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(loss_val)

    except KeyboardInterrupt:
        # after Ctrl + C, print final result
        logging.info('-' * 40)
        logging.info('Exiting from training early')
        model = load_model(model, args)
        loss_test, ppl_test = evaluate(model, test_dataloader, criterion, args.num_gestures, args)
        logging.info('-' * 40)
        logging.info("log = {}".format(args.log))
        logging.info("Number of parameters = {}".format(num_parameters_train))
        if args.signal_type == 'visual':
            logging.info('| test loss {:.2f}  | test acc {:.2f}'.format(loss_test, ppl_test))
        else:
            logging.info('| test loss {:.2f}  | test ppl {:.2f}'.format(loss_test, ppl_test))
        logging.info('-' * 40)
        # logger.close()
        logger.close()
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # store result
        record(end_time, args.path_results, args.dataset_name, args.optim, args.key_size, args.vhdropout, 
        args.levels, args.batch_size, args.epochs, args.lr, args.num_subblocks, args.en_res, args.temp_attn, 
        loss_test, acc_test, ppl_test, num_parameters_train, args.log)
        logger.close()

    
    # print final result
    # logger.close()
    model = load_model(model, args)
    loss_test, acc_test, ppl_test = evaluate(model, test_dataloader, criterion, args.num_gestures, args)
    logging.info('-' * 40)
    logging.info("log = {}".format(args.log))
    logging.info("Number of parameters = {}".format(num_parameters_train))
    if args.signal_type == 'visual':
        logging.info('| test loss {:.2f}  | test acc {:.2f}'.format(loss_test, ppl_test))
    else:
        logging.info('| test loss {:.2f}  | test acc: {:.2f} | test ppl {:.2f}'.format(loss_test, acc_test, ppl_test))
    logging.info('-' * 40)
    
    # store attention weights
    if args.temp_attn:
        save_visual_info(visual_info_all, args)
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # store result
    record(end_time, args.path_results, args.dataset_name, args.optim, args.key_size, args.vhdropout, args.levels, args.batch_size, \
        args.epochs, args.lr, args.num_subblocks, args.en_res, args.temp_attn, loss_test, acc_test, ppl_test, num_parameters_train, args.log)
    logger.close()
    

def evaluate(model, dataloader, criterion, num_gestures, args):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    correct_total = 0
    total = 0
    with torch.no_grad():
        for data_batch, label_batch in dataloader:
            if args.signal_type != 'both':
                data_batch = data_batch.cuda(args.gpu_id)
                label_batch = label_batch.cuda(args.gpu_id)
            else:
                data_batch = (data_batch[0].cuda(args.gpu_id), data_batch[1].cuda(args.gpu_id))
                label_batch = label_batch.cuda(args.gpu_id)
            if args.signal_type != 'both':
                if args.temp_attn:
                    output_batch, _ = model(data_batch)
                else:
                    output_batch, _ = model(data_batch)
            else:
                output_batch, k_output_batch, v_output_batch = model(data_batch, label_batch, 'test')

            # Discard the effective history, just like in training
            if args.signal_type != 'mnist':
                eff_history = args.seq_len - args.valid_len
                # output_batch = output_batch[:, eff_history:].contiguous().view(-1, num_gestures)
                # label_batch = label_batch[:, eff_history:].contiguous().view(-1)
                pred = output_batch.data.max(1, keepdim=True)[1]
                correct_total += pred.eq(label_batch.data.view_as(pred)).cpu().sum()
                total += label_batch.size(0)
            else:
                pred = output_batch.data.max(1, keepdim=True)[1]
                correct_total += pred.eq(label_batch.data.view_as(pred)).cpu().sum()
            if args.signal_type != 'both':
                loss = criterion(output_batch, label_batch)
            else:
                loss = criterion(output_batch, label_batch) + 0.5 * criterion(k_output_batch, label_batch) + 0.5 * criterion(v_output_batch, label_batch)

            if args.signal_type != 'both':
                total_loss += (data_batch.size(1) - eff_history) * loss.item()
                processed_data_size += data_batch.size(1) - eff_history
            else:
                total_loss += loss.item()
                processed_data_size += 1
        
        if args.signal_type == 'mnist':
            acc = 100*float(correct_total)/len(dataloader.dataset)
            loss = round(total_loss / processed_data_size, 6)
            return loss, acc
        else:
            acc = 100*float(correct_total)/float(total)
            loss = round(total_loss / processed_data_size, 6)
            ppl = round(np.exp(loss), 4)
            return loss, acc, ppl


if __name__ == '__main__':
    # original tcn
    # args = Config(optim='SGD', dataset_name='penn', lr=4, epochs=150, batch_size=64, gpu_id=0, temp_attn=False, en_res=False, log="tcn_ori")
    
    # word-penn
    # args = Config(optim='Adam', key_size=600, lr=1e-4, epochs=150, gpu_id=1, num_subblocks=1, log="tcanet_test")
    
    # char_penn
    # args = Config(optim='Adam', dataset_name='char_penn', lr=1e-4, epochs=3, batch_size=256, gpu_id=1, en_res=True, 
    #     dropout=0.1, emb_dropout=0.1, levels=3, emsize=100, nhid=450, key_size=100, valid_len=320, seq_len=400, 
    #     temp_attn=False, log="tcanet_char_penn_test")
    
    # train(args)
    processes = []
    args_list = [
                ###### char_penn
                # Config(optim='Adam', dataset_name='char_penn', lr=1e-4, epochs=250, batch_size=128, gpu_id=0, en_res=False, 
                # dropout=0.1, emb_dropout=0.1, levels=6, emsize=100, nhid=450, key_size=100, valid_len=320, seq_len=400, 
                # temp_attn=True, visual=False, num_subblocks=1, continue_train=True, log="tcanet_char_penn_en_res-False_blocks-1_levels-6")
                # Config(optim='Adam', dataset_name='char_penn', lr=4, epochs=200, batch_size=128, gpu_id=1, en_res=False, 
                # dropout=0.1, emb_dropout=0.1, levels=3, emsize=100, nhid=450, key_size=100, valid_len=320, seq_len=400, 
                # temp_attn=False, num_subblocks=2, log="tcanet_char_penn_ori")
                
                ###### word_penn
                # Config(optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=1, num_subblocks=1, levels=4, en_res=True,
                # temp_attn=True, visual=True, log="tcanet_num_subblocks-1_levels-4_verti_hori_without_conv")
                # Config(optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=0, num_subblocks=1, levels=4, en_res=False,
                # temp_attn=True, visual=True, log="tcanet_en_res-False_num_subblocks-1_levels-4_v_h"),
                # Config(optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=1, num_subblocks=1, levels=6, en_res=True,
                # temp_attn=True, seq_len=40, valid_len=40, visual=True, log="tcanet_seq_len-40_valid_len-40_levels-6")

                # Config(batch_size=32, num_gestures=6, optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=6, num_subblocks=0, levels=4, en_res=False, signal_type='both',
                # temp_attn=False, seq_len=20, valid_len=20, visual_emsize=128 ,conv=True, fps=10, visual=False, wandb_mode='disabled', log="K_sig_conv-True-temp_atten-true"),

                Config(batch_size=32, num_gestures=6, optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=7, num_subblocks=0, levels=4, en_res=False, signal_type='kinematic',
                temp_attn=False, seq_len=20, valid_len=20, visual_emsize=128 ,conv=True, fps=10, visual=False, wandb_mode='disabled', log="K_sig_conv-True-temp_atten-true"),

                # Config(num_gestures=6, optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=6, num_subblocks=0, levels=4, en_res=False, signal_type='kinematic',
                # temp_attn=True, seq_len=80, valid_len=80, visual_emsize=128 ,conv=True, visual=False, wandb_mode='disabled', log="K_sig_conv-True-temp_atten-true"),

                # Config(num_gestures=10, optim='Adam', key_size=300, lr=1e-4, epochs=200, gpu_id=7, num_subblocks=0, levels=4, en_res=False, signal_type='kinematic',
                # temp_attn=False, seq_len=80, valid_len=80, conv=True, visual=False, wandb_mode='online', log="K_sig_conv-True-temp_atten-false"),

                ###### sequential visual
                # Config(optim='Adam', dataset_name='visual', key_size=25, lr=2e-3, epochs=100, gpu_id=0, num_subblocks=3, levels=4, 
                # batch_size=64, dropout=0.05, clip=-1, ksize=7, nhid=25, permute=False, num_workers=4, emsize=1, seq_len=784, 
                # temp_attn=True, en_res=True, visual=False, continue_train=True, log="tcanet_visual_num_blocks-3_levels-4_permute-False")
                
                ###### permute visual
                # Config(optim='Adam', dataset_name='visual', key_size=25, lr=1e-4, epochs=200, gpu_id=1, num_subblocks=1, levels=10, 
                # batch_size=64, dropout=0.05, clip=-1, ksize=7, nhid=25, permute=True, num_workers=4, emsize=1, seq_len=784, 
                # log="tcanet_visual_num_blocks-1_levels-10_permute-True")

                ###### wikitext-2
                # Config(optim='Adam', dataset_name='wikitext-2', key_size=600, lr=1e-4, epochs=300, gpu_id=0, num_subblocks=1, 
                # levels=6, en_res=False, temp_attn=True, visual=False, log="tcanet_wt-2_num_subblocks-1_levels-6_en_res-False")

                ###### wikitext-103
                # Config(optim='Adam', dataset_name='wikitext-103', key_size=1000, lr=1e-4, epochs=300, gpu_id=0, num_subblocks=1, 
                # levels=6, en_res=False, temp_attn=True, visual=False, log="tcanet_wt-2_num_subblocks-1_levels-6_en_res-False")
                ]

    num_processes = len(args_list)

    for i in range(num_processes):
        p = mp.Process(target=train, args=([args_list[i],]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
