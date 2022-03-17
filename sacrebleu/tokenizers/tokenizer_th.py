# -*- coding: utf-8 -*-
#import
from nlpo3 import segment
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp

class TokenizerThai(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes Thai languega input line using 
        Thai Library : https://pythainlp.github.io/tutorials/notebooks/nlpo3ipynb.html

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence = segment(line.strip())
        sentence_str = ' '.join(sentence)
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'th' 
