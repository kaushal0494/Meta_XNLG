# -*- coding: utf-8 -*-
#import
from pyvi import ViTokenizer
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp

class TokenizerIVietnamese(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes Vietnamese language input line using 
        Library : https://github.com/trungtv/pyvi

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence_str = ViTokenizer.tokenize(line.strip())
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'vi' 
