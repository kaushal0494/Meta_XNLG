# -*- coding: utf-8 -*-
from trtokenizer.tr_tokenizer import WordTokenizer
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp
wordtoken_obj = WordTokenizer()

class TokenizerITurkish(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes Turkish language input line using 
        Library : pip install trtokenizer

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence = wordtoken_obj.tokenize(line.strip())
        sentence_str = ' '.join(sentence)
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'tr' 
