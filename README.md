# Meta-XNLG:

Hi! This repository contains code for the paper [Meta-XNLG: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation](https://aclanthology.org/2022.findings-acl.24.pdf) published at Findiaings of ACL 2021. If you have any questions, please feel free to create a Github issue or reach out to the first author at <cs18resch11003@iith.ac.in>.

## Installation Instruction
All the dependencies can be installed with the below conda command.

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
We tested the code with ```Python=3.8```, ```PyTorch==1.8``` and ```transformers=4.11```

## Downloads

- Download the ZmBART chekpoint from [here](https://drive.google.com/drive/folders/1k9Usn2vc7C4SOndJ_9vjYMUEqqyLVttn?usp=sharing) and extract at ```ZmBART/checkpoints/``` location
- Also, download mBART.CC25 checkpoint and extract to ```ZmBART/``` by using below commands
```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz
```
- The raw training, validation and evaluation dataset for English, Hindi, and Japanese can be downloaded from [here](https://drive.google.com/drive/folders/1tW8BmYIa9U1KOjIIaT5VX3UUIowHoCiA?usp=sharing). The model will need an SPM tokenized dataset as input. The raw dataset should be converted into SPM token dataset with instruction provided [here](https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md#bpe-data-1), and [here](https://tmramalho.github.io/science/2020/06/10/fine-tune-neural-translation-models-with-mBART/) or can be directly downloaded in next point pre-processed by us.  
- The SPM tokenized training, validation and evaluation dataset for English, Hindi and Japanese can be downloaded from [here](https://drive.google.com/drive/folders/1tVX6VtTRadCi1bjsORw7vVSi8ygVUsao?usp=sharing).
- Extract the SPM tokenized datasets at ```ZmBART/dataset/preprocess/``` 
- Note that we added a training and validation dataset for Hindi and Japanese for few-shot training. Validation data is optional.
- For ATS, we did joint multilingual training (see the paper for more details), so 500 monolingual datasets are augmented.
