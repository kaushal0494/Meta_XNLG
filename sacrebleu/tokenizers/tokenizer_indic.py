# -*- coding: utf-8 -*-
#import
from indicnlp.tokenize import indic_tokenize 
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp

class TokenizerIndic(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes an indian languega input line using Indic tokenizer.
        using Indic NLP Library (https://anoopkunchukuttan.github.io/indic_nlp_library/)

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence = indic_tokenize.trivial_tokenize(line.strip())
        sentence_str = ' '.join(sentence)
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'indic' 
