import os
import torch

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def load_model_tokenizer(args):
    config_kwargs = {}
    if args.max_generated_seq_len:
        config_kwargs.update({'max_length': args.max_generated_seq_len})
    if args.beam_size:
        config_kwargs.update({'num_beams': args.beam_size})
    if args.length_penalty:
        config_kwargs.update({'length_penalty': args.length_penalty})
    if args.no_repeat_ngram_size:
        config_kwargs.update({'no_repeat_ngram_size': args.no_repeat_ngram_size})
    
    config = AutoConfig.from_pretrained(
        args.model_chkpt,
        cache_dir=args.cache_dir, **config_kwargs,
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(args, p))


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_chkpt,
        use_fast=False, cache_dir=args.cache_dir,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_chkpt,
            from_tf=".ckpt" in args.model_chkpt,
            config=config,
            cache_dir=args.cache_dir,
    )


    """
    #args.model_chkpt = args.model_chkpt.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_chkpt,
        use_fast=False, 
        cache_dir=args.cache_dir,
    )
    special_tokens = {"eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token, \
        "sep_token": tokenizer.eos_token, "unk_token": tokenizer.unk_token}
    tokenizer.add_special_tokens(special_tokens)

    config = AutoConfig.from_pretrained(
        args.model_chkpt,
        cache_dir=args.cache_dir,
        bos_token_id= tokenizer.bos_token_id,
        eos_token_id= tokenizer.eos_token_id,
        sep_token_id= tokenizer.sep_token_id,
        pad_token_id= tokenizer.pad_token_id,
        unk_token_id= tokenizer.unk_token_id,
        output_hidden_states=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_chkpt,
            config=config,
            cache_dir=args.cache_dir,
        )"""
    return model, tokenizer