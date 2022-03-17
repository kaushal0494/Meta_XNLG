""" Implementation of all available options """
from __future__ import print_function

import argparse


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add_argument('--train_data', help='path to training data')
    group.add_argument('--val_data', help='path to validation data')
    group.add_argument('--test_data', help='path to testing data')
    group.add_argument('--output_dir', help='path to output directory', required=True)
    group.add_argument('--input_dir', help='path to Input directory', required=True)
    group.add_argument('--gen_file_name', help='name of the generated file (TSV)')
    group.add_argument('--cache_dir', help='to store cache files')
    group.add_argument('--config_file_name', help='name of the config file for task and dataset', default="auxi_tgt_lang_config")
    
    group.add_argument('--train_batch_size', help='training batch size', type=int, default=32)
    group.add_argument('--eval_batch_size', help='validation batch size', type=int, default=32)
    group.add_argument('--test_batch_size', help='testing batch size', type=int, default=32)

    group.add_argument('--learning_rate', help='training learning rate', type=float, default=1e-3)
    group.add_argument('--grad_clip', help='training clip', type=float, default=5.0)
    group.add_argument('--warmup_ratio', help='training warmup ratio range 0.0 to 1.0', type=float, default=0.05)
    group.add_argument('--gradient_accumulation_steps', help='training gradient accumulation steps', type=int, default=1)
    group.add_argument('--weight_decay', help='training weight_decay', type=float, default=0.01)
    group.add_argument('--label_smoothing_factor', help='training label smoothing factor', type=float, default=0.1)
    group.add_argument('--num_train_epochs', help='training epochs', type=int, default=3)
    group.add_argument('--lr_scheduler_type', help='training lr lr_schedule_type', default='linear')
    group.add_argument('--logging_steps', help="training logging_step", type=int, default=500)
    group.add_argument('--save_steps', help="training save steps", type=int, default=500)
    group.add_argument('--n_inner_iter', help='Meta learining inner iterations', type=int, default=2)
    group.add_argument('--eval_steps', help='evaluation steps', type=int, default=500)
    group.add_argument('--num_workers', help='training data loader workers', type=int, default=2)
    group.add_argument('--save_total_limit', help='training checpoints save limit', type=int, default=3)

    group.add_argument('--max_source_length', help='maximum input length to model', type=int, default=128)
    group.add_argument('--max_target_length', help='maximum, target lenght to model', type=int, default=128)

    group.add_argument('--do_meta_train', help="start the meta-training", action='store_true')
    group.add_argument('--do_fine_tune', help="start the fine-tuning", action='store_true')
    group.add_argument('--do_test', help='start the testing', action='store_true')

    group.add_argument('--model_type', help='name of the base model', required=True)
    group.add_argument('--model_chkpt', help='path to the model checkpoint', required=True)
    group.add_argument('--resume_from_checkpoint', help='resulme from the checkpoint')
    
    group.add_argument('--meta_lr', default=1e-4, type=float, help='learning rate for inner loops')
    group.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    group.add_argument('--seed', help='random seed', type=int, default=1234)
    group.add_argument('--local_rank', type=int, default=-1, help="For distributed training: local_rank")
    group.add_argument('--no_cuda',"--no_cuda", action='store_true', help="Avoid using CUDA when available")
    group.add_argument('--fp16', action='store_true',help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    group.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    group.add_argument('--freeze_embeds', help='freeze all learned embedding weights', action='store_true')
    group.add_argument('--freeze_encoder', help='freeze all encoder weights', action='store_true')
    group.add_argument('--freeze_embeds_and_decoder', help='freeze all embed and decoder weights', action='store_true')

    group.add_argument('--read_n_data_obj', help='read n data object for train, valid and test', type=int, default=-1)

    #beam Search and generation related hyper-parameters
    group.add_argument('--length_penalty', help='length penalty for beam search', type=float, default=0.6)
    group.add_argument('--beam_size', help='Size for beam search', type=int, default=8)
    group.add_argument('--early_stopping', help='early stopping for beam search', action='store_true')
    group.add_argument('--num_of_return_seq', help='Number of return seqs for beam search', type=int, default=8)
    group.add_argument('--max_generated_seq_len', help='Maximum generated seqence length', type=int, default=100)
    group.add_argument('--min_generated_seq_len', help='Minimu generated seqence length', type=int, default=1)
    group.add_argument('--sampling', help='use sampling during test', action='store_true')
    group.add_argument('--temperature', help='Beam search  will use during test', type=float, default=1.0)
    group.add_argument('--repetition_penalty', help='Repetition penalty for beam search', type=float, default=1.0)
    group.add_argument('--no_repeat_ngram_size', help='No repeat ngram size for beam search', type=int, default=2)
    group.add_argument('--top_k', help='top_k for test generation', type=int, default=50)
    group.add_argument('--top_p', help='top_p for test generation', type=float, default=1.0)
    group.add_argument('--num_beam_groups', help='num_beam_groups for test generation', type=int, default=1)
    
    #Task name
    group.add_argument("--task_name", default='qa', type=str, help="The name of the task to train to selected")
    group.add_argument("--task_data_name", default='tydiqa', type=str, help="Name of the task-spacific data")


def add_md_help_argument(parser):
    """ md help parser """
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
