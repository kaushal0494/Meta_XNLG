# -*- coding: utf-8 -*-
#import
from konlpy.tag import Okt
from konlpy.utils import pprint 
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp
okt = Okt()

class TokenizerIKorean(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes Korean languega input line using konlpy tokenizer.
        using konlpy Library (https://konlpy.org/en/v0.3.0/install/)

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence = okt.morphs(line.strip(), norm=True, stem=True)
        sentence_str = ' '.join(sentence)
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'ko' 
