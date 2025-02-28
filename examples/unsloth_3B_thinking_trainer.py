import os
import transformers
import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model_state_dict, PeftConfig, PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print debug info
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Patch FastLanguageModel
PatchFastRL("GRPO", FastLanguageModel)

# Model configuration
max_seq_length = 512  # Reduced from 1024 for memory
lora_rank = 32       # Reduced from 64 for memory
gpu_memory_utilization = 0.3  # Reduced from 0.5 for memory

try:
    print("Starting model loading...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # Disable vLLM
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )
    print("Model loaded successfully")
    
    if model is None:
        raise ValueError("Model is None after loading")
    
    print("Model type:", type(model))
    print("Model config:", model.config)
        
except Exception as e:
    print(f"Detailed model loading error: {str(e)}")
    print(f"Error type: {type(e)}")
    raise

# Apply PEFT
if model is None:
    raise ValueError("Cannot proceed - model not properly loaded")

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# System prompt and format
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Training configuration
training_args = GRPOConfig(
    use_vllm=False,  # Disable vLLM for training
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

def save_model(model, tokenizer, save_path: str):
    """Save model, config, tokenizer, and PEFT adapter config"""
    try:
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save the PEFT adapter config
        model.peft_config.save_pretrained(save_path)
        
        # Save the LoRA weights
        peft_state_dict = get_peft_model_state_dict(model)
        torch.save(peft_state_dict, os.path.join(save_path, "adapter_model.bin"))
        
        # Save base model config
        model.config.save_pretrained(save_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved successfully to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

def load_lora_weights(base_model, lora_path: str):
    """Load LoRA weights into base model"""
    try:
        config = PeftConfig.from_pretrained(lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path)
        logger.info(f"LoRA weights loaded successfully from {lora_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {str(e)}")
        return None

def main():
    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # Save model
    save_path = "grpo_saved_lora"
    if save_model(model, tokenizer, save_path):
        print(f"Model saved to {save_path}")
    else:
        print("Failed to save model")

    # Test generation
    test_text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How many r's are in strawberry?"},
    ], tokenize=False, add_generation_prompt=True)

    # Load and test model
    loaded_model = load_lora_weights(model, save_path)
    if loaded_model is not None:
        output = loaded_model.generate(
            tokenizer(test_text, return_tensors="pt").input_ids.cuda(),
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95,
        )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\nTest generation output:", output_text)
    else:
        print("Failed to load model for testing")

if __name__ == "__main__":
    main()