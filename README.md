# Project Name: Code Summarization with Transformer-based Models
## Introduction
This project aims to summarize code snippets using transformer-based models. The task involves generating descriptive summaries for given code snippets, which can be beneficial for code documentation and understanding. We leverage state-of-the-art transformer architectures and fine-tune them on a dataset containing code snippets along with corresponding docstrings.

## Dataset
We use the `calum/the-stack-smol-python-docstrings` dataset, which contains pairs of Python code snippets and their associated docstrings. Each code snippet is accompanied by a docstring that describes its functionality. The dataset is split into training, validation, and test sets.

## Setup
To set up the project environment, follow these steps:

Clone the repository from GitHub:

```bash
git clone https://github.com/valvarl/python-docstring-producer.git
cd python-docstring-producer
```

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
## Training
To train the transformer-based model for code summarization, execute the train.py script with appropriate arguments. The script loads the dataset, initializes the model, and fine-tunes it on the training data. Below are the configurable arguments:

- **--model_name_or_path**: Pre-trained model to use for fine-tuning (default: 'facebook/opt-125m').
- **--max_seq_length**: Maximum sequence length for tokenization (default: 128).
- **--train_batch_size**: Batch size for training (default: 32).
- **--valid_batch_size**: Batch size for validation (default: 32).
- **--learning_rate**: Learning rate for training (default: 2e-5).
- **--weight_decay**: Weight decay for regularization (default: 0.01).
- **--num_train_epochs**: Number of training epochs (default: 10).
- **--devices**: List of GPU devices to use (default: [0]).
- **--lora_r**: LoRA parameter r for fine-tuning (default: 16).
- **--lora_alpha**: LoRA parameter alpha for fine-tuning (default: 32).
- **--lora_target_modules**: List of target modules for LoRA fine-tuning.
- **--lora_dropout**: Dropout probability for LoRA fine-tuning (default: 0.05).
Example command:

```bash
python train.py --model_name_or_path facebook/opt-125m --num_train_epochs 5
```

## Inference
To generate summaries for new code snippets, use the inference.py script. Provide the path to the trained model checkpoint and the input code file. The script generates descriptive summaries for the input code using the fine-tuned model.

Arguments for inference:

- **--checkpoint_path**: Path to the trained model checkpoint.
- **--input_code_file**: Path to the file containing input code snippets.
- **--device**: Device for inference (default: 'cuda:0').
- **--doc_max_length**: Maximum length of the generated docstring (default: 128).
- **--repetition_penalty**: Penalty for repeating tokens during generation (default: 1).
Example command:

````bash
python inference.py --checkpoint_path opt-125m-fine-tuned/peft-model --input_code_file ./data/input.txt

</s>Describe what the following code does:
```Python
def sum_two_numbers(self, a, b):
    return a + b
```
# docstring
Sum two numbers of two-numbers.
````

## Results
The trained model achieves competitive performance compared to existing solutions in code summarization tasks. We conducted experiments to optimize hyperparameters and fine-tune the model, resulting in improved summarization quality. We used OPT 125m, OPT 350m, and Falcon 1b for testing, and fine-tuned the models using LoRA. The best results are obtained with a larger number of epochs, which may require significant computational resources.

## Areas of Improvement
While the current approach demonstrates promising results in code summarization, there are several areas where improvements could be made:

1. Large Language Models (LLMs)
Instead of fine-tuning transformer-based models primarily pre-trained on text corpora, a dedicated LLM trained predominantly on code could potentially yield better performance. Training a large language model specifically on code-related tasks can enhance the model\'s understanding of programming syntax, semantics, and context, leading to more accurate code summarization.

2. Data Augmentation
Augmenting the dataset with diverse code snippets and corresponding docstrings can improve the model\'s ability to generalize across different coding styles, languages, and domains. Techniques such as paraphrasing, code transformation, and code synthesis can be employed to generate additional training examples, thereby enhancing the model\'s robustness.

## References
[OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Contributors
Varlachev Valery (@valvarl)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.