# Copyright (c) 2024 Varlachev Valery

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


class OPTForDocstrings():
    def __init__(self, model_name_or_path: str, r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05):
        super().__init__()

        self.opt_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        
        for param in self.opt_model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        self.opt_model.gradient_checkpointing_enable()
        self.opt_model.enable_input_require_grads()
        
        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        self.opt_model.lm_head = CastOutputToFloat(self.opt_model.lm_head)

        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            init_lora_weights=False
        )

        self.peft_model = get_peft_model(self.opt_model, self.lora_config)

    def get_peft_model(self):
        return self.peft_model

    @staticmethod
    def from_pretrained(model_id: str):
        config = PeftConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, return_dict=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        peft_model = PeftModel.from_pretrained(model, model_id)
    
        return peft_model, tokenizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lora_r', type=int, default=16)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'v_proj'])
        parser.add_argument('--lora_dropout', type=float, default=0.05)
        return parser
