import os
from random import shuffle
import sys
import argparse
import logging
import json
import numpy as np
import copy
from tqdm import tqdm
from os import listdir
import collections
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import Adafactor
import transformers
from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from utils import (
    freeze_embeds, 
    freeze_params,
    assert_all_frozen,
    check_expected_languages,
    check_correct_tasks_and_datasets_name,
    _prepare_inputs,
    get_test_dataloader,
    get_train_dataloader,
    split_batch,
    creat_dir,
    save_model,
)
from utils_trans import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    MultiDataset,
    TokenizedDataset,
    TokenizedDataCollator,
    build_compute_metrics_fn,
    check_output_dir,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
    calculate_rouge,
    calculate_bleu,
)


from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler, 
    SubsetRandomSampler,
    TensorDataset, 
    ConcatDataset
)

from model import load_model_tokenizer
from datasets import load_the_dataset
import opts as opts
#from trainer import Trainer

from transformers.trainer_utils import is_main_process
from transformers.training_args import ParallelMode
import warnings
warnings.filterwarnings("ignore")

import higher

#import apex
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

def fine_tuning(args, model, auxi_lang, train_language_dataset, tokenizer):
    model.train()
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # AdamW optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    set_seed(args.seed)

    for mt_itr in range(1, args.num_train_epochs + 1):  # outer loop learning
        train_language_iter = {}
        for language, (data_obj, dataset_collator) in train_language_dataset.items():
            lang_dataloader = get_train_dataloader(args, data_obj, dataset_collator)
            train_language_iter[language] = (iter(lang_dataloader), lang_dataloader)
        start_time = time.time()
        losses = 0
        #for uniform Sampling
        number_of_batchs = 10000000
        for auxi_lan in auxi_lang:
            batch_length = len(train_language_iter[auxi_lan][1])
            if batch_length < number_of_batchs:
               number_of_batchs = batch_length 
        logger.info("Number of Batches to run are: %d" %(number_of_batchs))
        for batch_idx in range(number_of_batchs):
            for lang, (task_loader_iter, task_loader) in train_language_iter.items():
                try:
                   batch = next(task_loader_iter)
                except StopIteration:
                   batch = next(iter(task_loader))
                _batch = _prepare_inputs(args, batch)
                optimizer.zero_grad()
                outputs = model(**_batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                losses += loss.item()

            if batch_idx != 0 and batch_idx % args.logging_steps == 0:
                iter_time = time.time() - start_time
                losses = losses / (args.logging_steps*len(auxi_lang))
                logger.info("\tEpoch: %d\t| Batch: %d/%d\t| Traing Loss: %f\t| Total Run Time of Batch : %.2f Seconds" %( mt_itr, batch_idx, number_of_batchs-1, losses, iter_time))
                start_time = time.time()
                losses= 0

        if mt_itr % args.save_steps == 0:
            save_dir = creat_dir(args, "ft_checkpoint-ep-"+str(mt_itr))
            save_model(args, tokenizer, model, save_dir)
            logger.info("Saving model checkpoint to %s after Epoch %d", save_dir ,mt_itr)   
        
    save_model(args, tokenizer, model, args.output_dir)
    logger.info("Fine-Tuning Training Completed") 


def meta_xnlg_with_maml(args, model, auxi_lang, train_language_dataset, tokenizer):
    # Loop to handle NLG tasks
    """ Meta-Train the model """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # AdamW optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # Adafactor Optimizer
    #optimizer_kwargs = {"scale_parameter": False, "relative_step": False} 
    #optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, **optimizer_kwargs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model) 

    set_seed(args.seed)

    for mt_itr in range(1, args.num_train_epochs + 1):  # outer loop learning
        train_language_iter = {}
        for language, (data_obj, dataset_collator) in train_language_dataset.items():
            lang_dataloader = get_train_dataloader(args, data_obj, dataset_collator)
            train_language_iter[language] = (iter(lang_dataloader), lang_dataloader)
        meta_step(args, model, tokenizer, mt_itr, train_language_iter, auxi_lang, optimizer)
        #valid_loss = valid_step(args, model, train_language_iter[valid_lang][1])
        #logger.info("Validation loss %f after %d epoches" %(valid_loss, mt_itr))
    save_model(args, tokenizer, model, args.output_dir)
    logger.info("Meta-Learning Training Completed") 

#def valid_step(args, model, valid_batch_loader):
#    model.eval()
#    valid_loss = 0
#    for valid_batch in tqdm(valid_batch_loader):
#        _valid_batch = _prepare_inputs(args, valid_batch)
#        outputs = model(**_valid_batch)
#        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#        if args.gradient_accumulation_steps > 1:
#            loss = loss / args.gradient_accumulation_steps
#        valid_loss += loss
#    return valid_loss/len(valid_batch_loader)
    

def meta_step(args, model, tokenizer, mt_itr, train_language_dataset, auxi_lang, optimizer):
    model.train()
    start_time = time.time()
    qry_losses = []
    #for uniform Sampling
    number_of_batchs = 10000000
    for auxi_lan in auxi_lang:
        batch_length = len(train_language_dataset[auxi_lan][1])
        if batch_length < number_of_batchs:
           number_of_batchs = batch_length 
    logger.info("Number of Batches to run are: %d and Iner Iterations are: %d" %(number_of_batchs, args.n_inner_iter))
    for batch_idx in range(number_of_batchs):
        model.zero_grad()
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.meta_lr)
        for lang, (task_loader_iter, task_loader) in train_language_dataset.items():
            try:
                batch = next(task_loader_iter)
            except StopIteration:
               batch = next(iter(task_loader))
            #print(lang, tokenizer.batch_decode(batch['input_ids']))
            _batch = _prepare_inputs(args, batch)
            support_set, query_set = split_batch(_batch,  int((args.train_batch_size)/2))
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fast_model, diffopt):
                #set_seed(args.seed)
                for _ in range(args.n_inner_iter):
                    fast_model.train()
                    st_outputs = fast_model(**support_set)
                    st_loss = st_outputs["loss"] if isinstance(st_outputs, dict) else st_outputs[0]
                    if args.gradient_accumulation_steps > 1:
                        st_loss = st_loss / args.gradient_accumulation_steps
                    diffopt.step(st_loss)

                qry_outputs = fast_model(**query_set)
                qry_loss = qry_outputs["loss"] if isinstance(qry_outputs, dict) else qry_outputs[0]  
                qry_loss.backward()
                qry_losses.append(qry_loss.detach())
            
        optimizer.step()
        if batch_idx != 0 and batch_idx % args.logging_steps == 0:
            qry_losses = sum(qry_losses)/(args.logging_steps*len(auxi_lang))
            iter_time = time.time() - start_time
            logger.info("\tEpoch: %d\t| Batch: %d/%d\t| Traing Loss: %f\t| Total Run Time of Batch : %.2f Seconds" %( mt_itr, batch_idx, number_of_batchs-1, qry_losses, iter_time))
            start_time = time.time()
            qry_losses = []       
        
    if mt_itr % args.save_steps == 0:
        save_dir = creat_dir(args, "ml_checkpoint-ep-"+str(mt_itr))
        save_model(args, tokenizer, model, save_dir)
        logger.info("Saving model checkpoint to %s after Epoch %d", save_dir ,mt_itr)

           
def _load_the_dataset(args, model, tokenizer, filter_lang, mode):
    language_data_pair = {}
    languages = [ lang.lower() for lang in listdir(args.input_dir)]
    dataset_class = Seq2SeqDataset
    dataset_collator = Seq2SeqDataCollator(tokenizer, args, None)
    
    lang_dataset_object = []
    for language in languages:
        if language in filter_lang:
            lang_dir_path = os.path.join(args.input_dir, language)
            lang_dataset = dataset_class(
                tokenizer,
                type_path='test',
                data_dir=lang_dir_path,
                n_obs=args.read_n_data_obj,
                max_target_length=args.max_target_length,
                max_source_length=args.max_source_length,
                prefix=model.config.prefix or "",
            )
            logger.info("Processed %s language and Size is: %d" %(language, len(lang_dataset)))
            lang_dataset_object.append((language, lang_dataset)) 

    if mode == 'test':
        for lang, data_obj in lang_dataset_object:
            lang_dataloader = get_test_dataloader(data_obj, dataset_collator, args.test_batch_size)
            if lang not in language_data_pair:
               language_data_pair[lang] = lang_dataloader
            else:
               logger.info("Duplicated Language found")
               return 
    if mode == 'train':
       for lang, data_obj in lang_dataset_object:
            #lang_dataloader = get_train_dataloader(args, data_obj, dataset_collator)
            if lang not in language_data_pair:
               language_data_pair[lang] = (data_obj, dataset_collator)
            else:
               logger.info("Duplicated Language found")
               return 
    return language_data_pair

def main():
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opts.add_md_help_argument(parser)
    opts.train_opts(parser)
    args = parser.parse_args()

    #Create and write config file in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f_out:
        json.dump(vars(args), f_out)
        f_out.close()   

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method="env://")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1,0] else logging.WARN, force=True)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    #Set the random seed for deterministic nature
    set_seed(args.seed)
    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
 
    # Set the verbosity to info of the Transformers logger (on main process only):
    #if is_main_process(args.local_rank):
    #transformers.utils.logging.set_verbosity_info()
    #transformers.utils.logging.enable_default_handler()
    #transformers.utils.logging.enable_explicit_format()
    
    # Load pretrained model and tokenizer
    #if args.local_rank not in [-1, 0]:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab 
    
    #loading model and tokenizer
    model, tokenizer = load_model_tokenizer(args)

    #if args.local_rank == 0:
    #    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    #Freezing model components 
    logger.info("Total Number of parameters: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.freeze_embeds:
        freeze_embeds(model)
    if args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())
    if args.freeze_embeds_and_decoder:
        freeze_embeds(model)
        freeze_params(model.get_decoder())
        assert_all_frozen(model.get_decoder())
    logger.info("Total Number of parameters after FREEZE (if any): %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # Check the expected values and tasks
    args.task_name = args.task_name.lower()
    args.task_data_name = args.task_data_name.lower()
    check_correct_tasks_and_datasets_name(args.task_name, args.task_data_name)
    
    with open(args.config_file_name) as file:
        load_lang = json.load(file)
    auxi_lang, tgt_lang = load_lang[args.task_name][args.task_data_name]["auxi_lang"], load_lang[args.task_name][args.task_data_name]["tgt_lang"]
    auxi_lang = [ lang.lower() for lang in auxi_lang]
    tgt_lang = [ lang.lower() for lang in tgt_lang]
    logger.info("Current Task : %s", args.task_name)
    logger.info("Current Task Datset Name : %s", args.task_data_name)
    logger.info("List of Auxilary Languages : %s", auxi_lang)
    logger.info("List of Target Languages : %s", tgt_lang)
    logger.info("Auxilary Languages are %s, Target Languages are %s and Total Languages are %s (English is not considered)" \
        %(len(auxi_lang), len(tgt_lang), len(auxi_lang + tgt_lang)))
    if args.config_file_name == "auxi_tgt_lang_config":
       check_expected_languages(auxi_lang, tgt_lang, args.task_name, args.task_data_name)
    
    #Added the dataloader here. Should be in the dictonary form {language: data}
    if args.do_meta_train:
        #valid_lang = tgt_lang[1]
        #auxi_lang.append(valid_lang)
        #train_language_dataset =load_the_dataset(args, model, tokenizer, auxi_lang, "test", logger) 
        train_language_dataset =_load_the_dataset(args, model, tokenizer, auxi_lang, "train")
        logger.info("***** Running Meta-Learning Script ****")
        logger.info("Do meta-learning on the following checkpoints: %s", args.model_chkpt)
        meta_xnlg_with_maml(args, model, auxi_lang, train_language_dataset, tokenizer)

    if args.do_fine_tune:
        #train_language_dataset =load_the_dataset(args, model, tokenizer, auxi_lang, "test", logger) 
        train_language_dataset =_load_the_dataset(args, model, tokenizer, auxi_lang, "train")
        logger.info("***** Running Fine-Tuning Script ****")
        logger.info("Do Fine-tuning on the following checkpoints: %s", args.model_chkpt)
        fine_tuning(args, model, auxi_lang, train_language_dataset, tokenizer)

    if args.do_test:
        test_language_dataset =_load_the_dataset(args, model, tokenizer, tgt_lang, "test")
        logger.info("***** Running Generation Script ****")
        logger.info("Starting Generation from the following checkpoints: %s", args.model_chkpt)
        model.eval()
        all_rustls_file_name = '_'.join(args.model_chkpt.split('/')[-2:])
        with open(os.path.join(args.output_dir, str(all_rustls_file_name)+"_all_results.txt"), 'w', encoding='utf8') as f_all:
            for test_lang, test_data in test_language_dataset.items():
              #if test_lang == 'bengali':
                lang_pred, lang_ref = [], []
                batch_size = test_data.batch_size
                num_examples = len(test_data.dataset)
                logger.info("****************** Start of %s language Generation ************" % test_lang)
                logger.info("  Num examples = %d", num_examples)
                logger.info("  Batch size = %d", batch_size)
                out_file_name = str(test_lang)+'_'+args.gen_file_name
                with open(os.path.join(args.output_dir, out_file_name), 'w', encoding='utf8') as f_gen:
                    for test_idx, test_instance in tqdm(enumerate(test_data), total=len(test_data)):
                        test_instance = _prepare_inputs(args, test_instance)
                        outputs = model.generate(
                            input_ids=test_instance['input_ids'],
                            attention_mask=test_instance['attention_mask'],
                            max_length=args.max_generated_seq_len,
                            num_beams=args.beam_size,
                            length_penalty=args.length_penalty,
                            no_repeat_ngram_size= args.no_repeat_ngram_size,
                            decoder_start_token_id =tokenizer.pad_token_id
                        )
                        batch_predictions=tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        batch_references =tokenizer.batch_decode(test_instance['labels'].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        batch_input = tokenizer.batch_decode(test_instance['input_ids'].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
                        assert len(batch_predictions) == len(batch_references) == len(batch_input), "Predictions and  reference lists are different size"
                        for current_idex, (ref, pred, inp) in enumerate(zip(batch_references, batch_predictions, batch_input)):
                            f_gen.write(json.dumps({"instance_id": test_idx*batch_size + (current_idex +1), "inputs" : inp, "reference": ref, "predictions":pred}, ensure_ascii=False) + "\n")
                        lang_pred += lmap(str.strip, batch_predictions)
                        lang_ref += lmap(str.strip, batch_references)
                    assert len(lang_pred) == len(lang_ref), "final length should be same"
                    logger.info("Langauge = %s, Test Length : %d" % (test_lang, len(lang_pred)))
                    cal_rouge = calculate_rouge(lang_pred, lang_ref, rouge_lang=test_lang)
                    logger.info(cal_rouge)
                    cal_bleu = calculate_bleu(lang_pred, lang_ref, bleu_lang=test_lang)
                    logger.info(cal_bleu)  
                    logger.info("*"*100)
                f_gen.close()
                f_all.write(json.dumps({"Language": test_lang, "Size": num_examples, "Rouge Score":cal_rouge, "BLEU Score": cal_bleu}, ensure_ascii=False) + "\n")
        f_all.close()


if __name__ == '__main__':
    main()



