# TinyZero-TRL

TinyZero-TRL is designed to provide an easy-to-use framework for reproducing R1-Zero, built on the popular HuggingFace-compatible TRL (Reinforcement Learning Training) library. Primarily educational, it enables users to implement and develop their own Zero models quickly. By minimizing the construction effort, TinyZero-TRL accelerates the development process, making it easier for anyone to create their own "zero-cost" models.

## News

- Support multimodal training for GRPO with number of iterations of $\mu$.
- scale for multi-node and 70B training.

## Installation

To install TinyZero-TRL, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/TinyZero-TRL.git
cd TinyZero-TRL
conda create -n zero-trl python=3.10
pip install -r requirements.txt
```

## Countdown task

### Data Preparation

Format your data for the countdown task by using the function `format_countdown` in `scripts/prepare_countdown_data.py`. This function processes raw countdown task data into the format required for training. The data should include a list of numbers (`nums`) and a target value (`target`). The function returns formatted data that can be directly used for model training.

```py
def format_countdown(prompt, tokenizer):
    numbers = prompt["nums"]
    target = prompt["target"]
  
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) one or multiple times but each number can only be used once. Show your work in <think> </think> tags. And return the final equation in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think step by step inside <think> tags."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target, "nums": numbers}
```

You can also custom your own format function in `format_countdown_data` to process the data according to your specific needs. For example, you can define how to format the countdown data, add additional fields, or modify the output structure.

### Train

To train the model for the countdown task, use the following command:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 29501 \
--num_processes=3 \
--config_file recipes/accelerate_configs/zero1.yaml \
src/tinyzero_trl/grpo.py \
+countdown=grpo-qwen-2.5-3b-base
```

### Result

After training, the results will be saved in the `outputs/countdown` directory. You can evaluate the model performance using:

```bash
python evaluate_countdown.py --model_path outputs/your-saved-model
```

## Mathematics Competition

### Data Preparation

Format your data for the mathematics competition task by using the function `format_math` in `scripts/prepare_math_data.py`. This function takes a math problem example, system prompt, instruction, and tokenizer as input and formats it into the required structure for training:

```py
def format_math(example, system_prompt, instruction, tokenizer):
    """Format Math500 dataset examples for training"""
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    if instruction is not None:
        example["problem"] += instruction
  
    prompt.append({"role": "user", "content": example["problem"]})
  
    return {"prompt": prompt}
```

### Train

To train the model for the mathematics competition task, use the following command:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 29501 \
--num_processes=3 \
--config_file recipes/accelerate_configs/zero1.yaml \
src/tinyzero_trl/grpo.py \
+math=grpo-deepseek-distill-1.5b-inst
```

### Result

After training, the results will be saved in the `outputs/math` directory. You can evaluate the model performance using:

```bash
python evaluate_math.py --model_path outputs/your-saved-model
```

## Evaluation

To be continue ...

## Acknowledge

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [Open R1](https://github.com/huggingface/open-r1)
- [TRL](https://github.com/huggingface/trl)
