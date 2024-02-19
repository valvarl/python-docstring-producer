# Copyright (c) 2024 Varlachev Valery

import argparse
import math
import os

from data import CodeDataset
from model import OPTForDocstrings
from transformers import Trainer, TrainingArguments

if __name__ == '__main__':
    
    # ----------
    # args
    # ----------
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser.add_argument('--model_name_or_path', type=str,
                       default='facebook/opt-125m')
    parser.add_argument('--preprocessing_num_workers', type=int, default=4)
    parser.add_argument('--max_seq_lenghth', type=int, default=128)
    parser.add_argument('--train_barch_size', type=int, default=32)
    parser.add_argument('--valid_barch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser = OPTForDocstrings.add_model_specific_args(parser)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(d) for d in args.devices])

    # ----------
    # data
    # ----------
    data_module = CodeDataset(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_lenghth,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )

    # ----------
    # model
    # ----------
    model = OPTForDocstrings(
        model_name_or_path=args.model_name_or_path,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
    ).get_peft_model()

    # ----------
    # training
    # ----------
    output_dir = os.path.basename(args.model_name_or_path) + '-fine-tuned'
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=args.train_barch_size,
        per_device_eval_batch_size=args.valid_barch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        logging_dir='./logs',
        push_to_hub=False,    
    )

    print(data_module.train)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module.train,
        eval_dataset=data_module.valid,
    )
    
    trainer.train()

    save_path = f'{output_dir}/peft-model'
    model.save_pretrained(save_path)
    print(f'Model saved at {save_path}')

    eval_results = trainer.evaluate()
    if 'eval_loss' in eval_results:
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
