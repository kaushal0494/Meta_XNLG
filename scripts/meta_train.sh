#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=6
source activate hpnlg_py38

export task_name="sum"
export input_data_dir_name="xlsum"

#input and output directories
export BASE_DIR='.'
export input_dir="../../mxnlg/xl-sum/seq2seq_base/XLSum_input/"
export output_dir="outputs/xlsum_14"

#model details
export model_type="t5" 
export model_chkpt="../../mxnlg/xl-sum/seq2seq_base/XLSum_output/mlqa/english/exp_18/checkpoint-2100"

export cache_dir='../cache_dir'
export config_file_name="single_auxi_lang_config" #"auxi_tgt_lang_config" 

python train.py \
    --input_dir ${input_dir}${input_data_dir_name} \
    --output_dir ${output_dir} \
    --model_type $model_type \
    --model_chkpt $model_chkpt \
    --max_source_length 512 \
    --max_target_length 84 \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --meta_lr 1e-5 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-08 \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --save_steps 1 \
    --cache_dir ${cache_dir} \
    --read_n_data_obj 1000  \
    --task_name ${task_name} \
    --freeze_embeds_and_decoder \
    --task_data_name ${input_data_dir_name} \
    --config_file_name ${config_file_name} \
    --n_inner_iter 2 \
    --do_meta_train \
