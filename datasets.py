import os
from re import S
import numpy as np
from os import listdir
import torch
from torch.utils.data import Dataset

def load_the_dataset(args, tokenizer, filter_lang, mode, logger):
    language_data_pair = {}
    languages = [ lang.lower() for lang in listdir(args.input_dir)]
    for language in languages:
        if language in filter_lang:
            files = [os.path.join(args.input_dir, language, 'test.source'), os.path.join(args.input_dir, language, 'test.target')]
            prepared_data = PrepareDataset(args, files, tokenizer, mode)
            logger.info("Processed %s language and Size is: %d" %(language, len(prepared_data)))
            if language not in language_data_pair:
               language_data_pair[language] = prepared_data
            else:
               logger.info("Duplicated Language found")
               return
    return language_data_pair

class PrepareDataset(Dataset):
    def __init__(self, args, files, tokenizer, mode):
        self.files = files
        self.tokenizer = tokenizer
        self.mode = mode
        self.task_name = args.task_name
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.read_n_data_obj = args.read_n_data_obj
        assert isinstance(self.files, list) and len(self.files) == 2, "input file format error or either file not present"
        
        if self.read_n_data_obj != -1: 
            self.source_data = list(i.rstrip('\n').lower() for count, i in enumerate(open(self.files[0], 'r').readlines()) if count < self.read_n_data_obj )
            self.target_data = list(i.rstrip('\n').lower() for count, i in enumerate(open(self.files[1], 'r').readlines()) if count < self.read_n_data_obj )
        else:
            self.source_data = list(i.rstrip('\n').lower() for i in open(self.files[0], 'r').readlines())
            self.target_data = list(i.rstrip('\n').lower() for i in open(self.files[1], 'r').readlines())
        assert len(self.source_data) == len(self.target_data), "both source and taget file should be of same size"

    def __len__(self):
        return len(self.source_data)

    def create_input_instance(self, encoided_ids, max_sequences_length):
        encoided_ids +=[-100 for _ in range(max_sequences_length - len(encoided_ids))]
        encoided_ids = torch.LongTensor(encoided_ids)
        input_mask = (encoided_ids != -100).long()
        encoided_ids.masked_fill_(encoided_ids == -100, self.tokenizer.pad_token_id)
        return {
            'input_ids': encoided_ids,
            'attention_mask' : input_mask,
        }
    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id
        return shifted_input_ids

    def features(self, source_line, target_line):
        if not isinstance(target_line, list):
            source_lines = [source_line]
            target_lines = [target_line]
        else:
            source_lines = source_line
            target_lines = target_line
        data_instances = []
        for source_example, target_examle in zip(source_lines, target_lines):
            if self.task_name == 'qg':
                answer, passage = source_example.split('</s>')[0], source_example.split('</s>')[1]
                #encoded = self.tokenizer.encode(answer + self.tokenizer.eos_token + passage + self.tokenizer.eos_token, add_special_tokens=False)
                data_instance = self.tokenizer.encode_plus(source_example, add_special_tokens=True, \
                    max_length=self.max_source_length, padding='max_length', truncation=True, return_tensors="pt")
            if self.task_name == 'sum':
                #encoded = self.tokenizer.encode(source_example + self.tokenizer.eos_token, add_special_tokens=False)
                data_instance = self.tokenizer.encode_plus(source_example, add_special_tokens=True, \
                    max_length=self.max_source_length, padding='max_length', truncation=True, return_tensors="pt")

            #encoded = encoded[:self.max_source_length]
            #data_instance = self.create_input_instance(encoded, self.max_source_length)
            #print(data_instance)

            #encoded_label = self.tokenizer.encode(target_examle + self.tokenizer.eos_token, add_special_tokens=False)
            #encoded_label = encoded_label[:self.max_target_length]
            #data_instance_label = self.create_input_instance(encoded_label, self.max_target_length)
            data_instance_label = self.tokenizer.encode_plus(target_examle, add_special_tokens=True, \
                max_length=self.max_target_length, padding='max_length', truncation=True, return_tensors="pt")
            data_instance['decoder_input_ids'] = self._shift_right_t5(data_instance_label['input_ids'])
            data_instance['labels'] = data_instance_label['input_ids']
            if self.mode == 'test' and self.task_name == 'qa':
                data_instance['passage'] = passage
                data_instance['answer'] = answer
            if self.mode == 'test' and self.task_name == 'sum':
                data_instance['input document'] = source_example
            
            if not isinstance(source_line, list):
                return data_instance
            else:
                data_instances.append(data_instance)
        return data_instances

    def set_epoch(self, epoch):
        self.random_state= np.random.RandomState(epoch)

    def __getitem__(self, indx):
        return self.features(self.source_data[indx], self.target_data[indx])