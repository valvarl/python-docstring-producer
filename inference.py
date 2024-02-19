# Copyright (c) 2024 Varlachev Valery

import argparse
from model import OPTForDocstrings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser.add_argument('--checkpoint_path', type=str,
                       default='opt-125m-fine-tuned/peft-model')
    parser.add_argument('--input_code_file', type=str,
                       default='./data/input.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--doc_max_length', type=int, default=128)
    parser.add_argument('--repetition_penalty', type=float, default=1)
    args = parser.parse_args()
    
    model, tokenizer = OPTForDocstrings.from_pretrained(args.checkpoint_path)
    model = model.to(args.device)

    with open(args.input_code_file) as inf:
        input_code = inf.read()

    inputs = tokenizer(f'Describe what the following code does:\n```Python\n{input_code}\n```\n# docstring\n', return_tensors='pt')

    inputs['input_ids'] = inputs['input_ids'].to(args.device)
    inputs['attention_mask'] = inputs['attention_mask'].to(args.device)

    generated_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + args.doc_max_length,
        do_sample=False,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        num_return_sequences=1,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id  # 
    )

    ret = tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=False)
    print(ret)
