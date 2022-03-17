#!/bin/bash
cd ..
#stting GPUs to use
export CUDA_VISIBLE_DEVICES=6
#conda enviourment activate
source activate hpnlg_py38

# misc. settings
export seed=1234

#input and output directories
export task_name="sum"
export input_data_dir_name="xlsum"
export BASE_DIR='.'
export input_dir="../../mxnlg/xl-sum/seq2seq_base/XLSum_input/"
export output_dir="outputs/xlsum_14"
export gen_file_name="pred.tsv"
export cache_dir='../cache_dir'

# model settings
export model_type="t5" 

export model_chkpt="outputs/xlsum_14/ml_checkpoint-ep-10"

python train.py \
    --input_dir ${input_dir}${input_data_dir_name} \
    --output_dir ${output_dir} \
    --model_type ${model_type} \
    --model_chkpt ${model_chkpt} \
    --test_batch_size 32 \
    --max_source_length 512 \
    --max_target_length 84 \
    --length_penalty 0.6 \
    --beam_size 4 \
    --early_stopping \
    --num_of_return_seq 1 \
    --min_generated_seq_len 20 \
    --max_generated_seq_len 200 \
    --cache_dir ${cache_dir} \
    --cache_dir ${cache_dir} \
    --read_n_data_obj -1 \
    --gen_file_name ${gen_file_name} \
    --task_name ${task_name} \
    --task_data_name ${input_data_dir_name} \
    --do_test 
