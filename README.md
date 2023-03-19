# Meta-XNLG
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Arko98/Hostility-Detection-in-Hindi-Constraint-2021/blob/main/LICENSE)
[![others](https://img.shields.io/badge/Huggingface-Cuda%2011.1.0-brightgreen)](https://huggingface.co/)
[![others](https://img.shields.io/badge/PyTorch-Stable%20(1.8.0)-orange)](https://pytorch.org/)

## About
This repository contains the source code of the paper titled [Meta-XNLG: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation](https://aclanthology.org/2022.findings-acl.24.pdf) which is published in the Findings of the *Association of Computational Linguistics (ACL 2022)* conference. If you have any questions, please feel free to create a Github issue or reach out to the first author at <cs18resch11003@iith.ac.in>.

## Environment Setup and Downloads
To set up the environment, use the following conda commands:

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
The code was tested with `Python=3.8`, `PyTorch==1.8`, and `transformers=4.11`. You can download dataset, model-chckpoints, generation and other resources from [here](https://drive.google.com/drive/folders/1ziTVKR7j_yIGDumLRJL-4Bah-uggeuta?usp=sharing).


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

## Model Fine-tuning with Centroid Languages for XL-SUM dataset
```

export CUDA_VISIBLE_DEVICES=0
source activate hpnlg_py38

export task_name="sum"
export input_data_dir_name="xlsum"

#input and output directories
export BASE_DIR='.'
export input_dir="XLSum_input/"
export output_dir="outputs/xlsum_out"

#model details
export model_type="t5" 
export model_chkpt="ZmT5_checkpoint"

export cache_dir='../cache_dir'
export config_file_name="auxi_tgt_lang_config" 

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
```

## Citation

```
@inproceedings{maurya-desarkar-2022-meta,
    title = "Meta-X$_{NLG}$: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation",
    author = "Maurya, Kaushal  and
      Desarkar, Maunendra",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.24",
    doi = "10.18653/v1/2022.findings-acl.24",
    pages = "269--284",
}
```

The meta-learning implementations are done with [higher](https://github.com/facebookresearch/higher) library and inspired from [X-MAML](https://github.com/copenlu/X-MAML).
