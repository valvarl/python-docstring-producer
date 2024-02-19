# Copyright (c) 2024 Varlachev Valery

from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class CodeDataset():
    
    DATASET_NAME = 'calum/the-stack-smol-python-docstrings'
    INSTRUCTION = 'Describe what the following code does:\n```Python\n%s\n```\n# docstring\n%s'
    
    def __init__(self, model_name_or_path: str, max_seq_length: int, val_test_size=0.2, test_size=0.5, preprocessing_num_workers=19):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True)
        ds = load_dataset(self.DATASET_NAME)

        train_testvalid = ds['train'].train_test_split(test_size=0.2)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        ds = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']}
        )

        def preprocess_function(examples):
            res = []
            for body_without_docstring, docstring in zip(examples['body_without_docstring'], examples['docstring']):
                res.append(self.INSTRUCTION % (body_without_docstring, docstring))
            result = self.tokenizer(res, padding='max_length', max_length=max_seq_length, truncation=False)
            result = {
                k: [t for t in tens if len(t) <= max_seq_length] 
                for k, tens in result.items() 
            }
            
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = ds.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=ds['train'].column_names,
        )

        self.ds = tokenized_dataset

    @property
    def train(self):
        return self.ds['train']

    @property
    def valid(self):
        return self.ds['test']

    @property
    def test(self):
        return self.ds['test']
