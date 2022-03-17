# -*- coding: utf-8 -*-

from re import T
from .tokenizer_none import NoneTokenizer
from .tokenizer_13a import Tokenizer13a
from .tokenizer_intl import TokenizerV14International
from .tokenizer_zh import TokenizerZh
from .tokenizer_ja_mecab import TokenizerJaMecab
from .tokenizer_char import TokenizerChar
from .tokenizer_indic import TokenizerIndic
from .tokenizer_ar import TokenizerIArabic
from .tokenizer_th import TokenizerThai
from .tokenizer_vi import TokenizerIVietnamese
from .tokenizer_ru import TokenizerRussian
from .tokenizer_tr import TokenizerITurkish
from .tokenizer_ko import TokenizerIKorean


DEFAULT_TOKENIZER = '13a'


TOKENIZERS = {
    'none': NoneTokenizer,
    '13a': Tokenizer13a,
    'intl': TokenizerV14International,
    'zh': TokenizerZh,
    'ja-mecab': TokenizerJaMecab,
    'char': TokenizerChar,
    'indic': TokenizerIndic,
    'ar' : TokenizerIArabic,
    'th' : TokenizerThai,
    'vi' : TokenizerIVietnamese,
    'ru' : TokenizerRussian,
    'tr' : TokenizerITurkish,
    'ko' : TokenizerIKorean,
}

