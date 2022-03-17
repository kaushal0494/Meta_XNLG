# -*- coding: utf-8 -*-
#import
import nltk
from .tokenizer_none import NoneTokenizer
from .tokenizer_re import TokenizerRegexp

class TokenizerIArabic(NoneTokenizer):
    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """
        Tokenizes Arabic languega input line using NLTK tokenizer.
        NLTK Library : https://www.nltk.org/

        :param line: a segment to tokenize
        :return: the tokenized line
        """
        sentence = nltk.tokenize.wordpunct_tokenize(line.strip())
        sentence_str = ' '.join(sentence)
        return self._post_tokenizer(sentence_str)

    def signature(self):
        return 'ar' 
