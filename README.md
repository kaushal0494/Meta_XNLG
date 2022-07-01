# Meta-XNLG

Hi! This repository contains code for the paper [Meta-XNLG: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation](https://aclanthology.org/2022.findings-acl.24.pdf) published at Findiaings of ACL 2021. If you have any questions, please feel free to create a Github issue or reach out to the first author at <cs18resch11003@iith.ac.in>.

## Installation Instruction
All the dependencies can be installed with the below conda command.

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
We tested the code with ```Python=3.8```, ```PyTorch==1.8``` and ```transformers=4.11```

All dataset-specific Meta-XNLG chekpoints can be downloaded from [here](https://drive.google.com/drive/folders/1ziTVKR7j_yIGDumLRJL-4Bah-uggeuta?usp=sharing)


## Zero-shot Target Language Generation with Meta-XNLG for XL-SUM dataset
```
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source activate hpnlg_py38

export seed=1234

#input and output directories
export task_name="sum"
export input_data_dir_name="xlsum"
export BASE_DIR='.'
export input_dir="../XLSum_input/"
export output_dir="outputs/xlsum_14"
export gen_file_name="pred.tsv"
export cache_dir='../cache_dir'

# model settings
export model_type="t5" 
export model_chkpt="outputs/xlsum"

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
    --min_generated_seq_len 0 \
    --max_generated_seq_len 200 \
    --cache_dir ${cache_dir} \
    --cache_dir ${cache_dir} \
    --read_n_data_obj -1 \
    --gen_file_name ${gen_file_name} \
    --task_name ${task_name} \
    --task_data_name ${input_data_dir_name} \
    --do_test 
```
