from typing import Callable, Dict, Iterable, List, Tuple, Union
import torch
import os
from torch import nn 
import collections
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Below code about freezing model parameters
def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5" or model_type == "mt5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)

def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"

# above code about freezing model parameters

def check_expected_languages(auxi_lang, tgt_lang, task_name, task_data_name):
    if task_name == "sum":
        if task_data_name == "xlsum":
           assert len(auxi_lang) == 3, f"length of the auxilary language should be 3 got {len(auxi_lang)}"
           assert len(tgt_lang) == 19, f"length of the target language should be 19 got {len(tgt_lang)}"
        if task_data_name == "wikilingua":
           assert len(auxi_lang) == 3, f"length of the auxilary language should be 3 got {len(auxi_lang)}"
           assert len(tgt_lang) == 14, f"length of the target language should be 14 got {len(tgt_lang)}"         
    
    if task_name == "qg":
        if task_data_name == "tydiqa":
            assert len(auxi_lang) == 2, f"length of the auxilary language should be 2 got {len(auxi_lang)}"
            assert len(tgt_lang) == 7, f"length of the target language should be 7 got {len(tgt_lang)}"
        if task_data_name == "xquad":
            assert len(auxi_lang) == 3, f"length of the auxilary language should be 3 got {len(auxi_lang)}"
            assert len(tgt_lang) == 8, f"length of the target language should be 8 got {len(tgt_lang)}"
        if task_data_name == "mlqa":
            assert len(auxi_lang) == 3, f"length of the auxilary language should be 3 got {len(auxi_lang)}"
            assert len(tgt_lang) == 4, f"length of the target language should be 4 got {len(tgt_lang)}"
    

def check_correct_tasks_and_datasets_name(task_name, task_data_name): 
    if task_name not in ["sum", "qg"]:
        raise ValueError("Task not found: %s" % (task_name))
    if task_name == "sum":
        if task_data_name not in ["xlsum", "wikilingua"]:
            raise ValueError("Task dataset %s  is not found task: %s" % (task_data_name, task_name))
    if task_name == "qg":
        if task_data_name not in ["mlqa", "xquad", "tydiqa"]:
            raise ValueError("Task dataset %s  is not found task: %s" % (task_data_name, task_name))

def _prepare_inputs(args, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        return inputs

def _get_eval_sampler(eval_dataset):
    return SequentialSampler(eval_dataset)

def get_test_dataloader(test_dataset, dataset_collator, test_batch_size):
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        test_sampler = _get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=test_batch_size,
            collate_fn=dataset_collator,
            drop_last=False,
        )

def _get_train_sampler(args, train_dataset):
    if args.local_rank == -1:
        return RandomSampler(train_dataset)
    else:
         DistributedSampler(train_dataset)

def get_train_dataloader(args, train_dataset, dataset_collator):
    train_sampler = _get_train_sampler(args, train_dataset)

    return DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=dataset_collator,
        drop_last=True,
        num_workers=args.num_workers,
    )
def split_batch(batch, split_point):
    support_set= {'input_ids': batch['input_ids'][:split_point, :],
                'attention_mask': batch['attention_mask'][:split_point, :],
                'decoder_input_ids':batch['decoder_input_ids'][:split_point, :],
                'labels': batch['labels'][:split_point, :]}
    query_set= {'input_ids': batch['input_ids'][split_point:, :],
                'attention_mask': batch['attention_mask'][split_point:, :],
                'decoder_input_ids':batch['decoder_input_ids'][split_point:, :],
                'labels': batch['labels'][split_point:, :]}
    return support_set, query_set

def creat_dir(args, suffix):
    save_dir = os.path.join(args.output_dir, suffix)
    if not os.path.exists(save_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_dir)
    return save_dir

def save_model(args, tokenizer, model, out_save_dir):
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(out_save_dir)
    tokenizer.save_pretrained(out_save_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(out_save_dir, 'training_args.bin')) 
   