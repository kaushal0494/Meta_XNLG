# Meta-XNLG:

Hi! This repository contains code for the paper [Meta-XNLG: A Meta-Learning Approach Based on Language Clustering for Zero-Shot Cross-Lingual Transfer and Generation](https://aclanthology.org/2022.findings-acl.24.pdf) published at Findiaings of ACL 2021. If you have any questions, please feel free to create a Github issue or reach out to the first author at <cs18resch11003@iith.ac.in>.

## Installation Instruction
All the dependencies can be installed with the below conda command.

``` 
conda env create --file environment.yml
conda activate py37_ZmBART
``` 
We tested the code with ```Python=3.7``` and```PyTorch==1.8```

Install the sentence-piece (SPM) from [here](https://github.com/google/sentencepiece). The binary should be in ```/usr/local/bin/spm_encode```

