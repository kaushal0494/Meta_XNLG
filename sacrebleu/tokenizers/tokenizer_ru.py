# -*- coding: utf-8 -*-
#import
import pymorphy2
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp
morph = pymorphy2.MorphAnalyzer()

class TokenizerRussian(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes RUSSIAN languega input line using Library : https://github.com/kmike/pymorphy2

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence_str = morph.parse(line.strip())[0].normal_form
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'ru' 
