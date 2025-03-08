import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import colorlog
import torch
from accelerate.state import PartialState
from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from peft import PeftConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor)
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.utils import is_peft_available
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from tinyzero_trl.utils.import_utils import is_unsloth_available
from tinyzero_trl.utils.script_utils import ScriptArguments

logger = colorlog.getLogger(__name__)

INTERNET_CONNECTION = True
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


@dataclass
class ALLModelConfig(ModelConfig):
    """
    Arguments pertaining to model for training and eval.
    """
    
    max_pixels: Optional[int] = field(
        default=12845056,  # 1280 * 28 * 28
        metadata={"help": "Maximum number of pixels for the image used in Qwen VL series"},
    )
    min_pixels: Optional[int] = field(
        default=3136,  # 256 * 28 * 28
        metadata={"help": "Minimum number of pixels for the image used in Qwen VL series"},
    )
    patch_size: Optional[int] = field(
        default=14,
        metadata={"help": "Patch size for the image used in Qwen VL series"},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM parameters during training"},
    )
    freeze_vision: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the vision model parameters during training"},
    )


# Add multimodal model support
def is_multimodal_model(model_id: str) -> bool:
    """Check if a model is multimodal by its config.
    
    Args:
        model_id: Model ID or path to check
    
    Returns:
        bool: True if model is multimodal, False otherwise
    """
    return AutoConfig.from_pretrained(model_id).model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return PartialState().local_process_index if torch.cuda.is_available() else "cpu"


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    if INTERNET_CONNECTION:
        try:
            
            # Try first if model on a Hub repo
            repo_files = list_repo_files(model_name_or_path, revision=revision)
        except (HFValidationError, RepositoryNotFoundError):
            # If not, check local repo
            repo_files = os.listdir(model_name_or_path)
    else:
        repo_files = os.listdir(model_name_or_path)
        
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_processing_class(
    script_args: ScriptArguments, 
    training_args, 
    model_args: ALLModelConfig, 
    auto_set_chat_template=True
):
    # Add multimodal model support
    processing_class = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=script_args.padding_side if script_args.padding_side is not None else "left",  # for generation
        truncation_side=script_args.truncation_side if script_args.truncation_side is not None else "left",  # Truncate from left to ensure we don't lose labels in final turn
    )
    # Add multimodal model support
    if is_multimodal_model(model_args.model_name_or_path):
        if "qwen" in model_args.model_name_or_path.lower():
            processing_class.image_processor.max_pixels = model_args.max_pixels
            processing_class.image_processor.min_pixels = model_args.min_pixels
            processing_class.image_processor.patch_size = model_args.patch_size

        if processing_class.tokenizer.pad_token_id is None:
            processing_class.tokenizer.pad_token_id = processing_class.tokenizer.unk_token_id
        processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    else:
        if processing_class.pad_token_id is None:
            processing_class.pad_token_id = processing_class.unk_token_id
    # Set reasonable default for models without max length
    # if processing_class.model_max_length > 100_000:
    #     processing_class.model_max_length = training_args.max_prompt_length + training_args.max_completion_length
    # assert processing_class.chat_template is not None, "Needs chat template!"
    if script_args.chat_template is not None:
        processing_class.chat_template = script_args.chat_template
    elif auto_set_chat_template and processing_class.chat_template is None:
        processing_class.chat_template = DEFAULT_CHAT_TEMPLATE

    # Add multimodal model support
    if "Qwen2.5-VL" in model_args.model_name_or_path or "Qwen2.5-VL" in model_args.model_name_or_path:
        processing_class.image_processor.max_pixels = model_args.max_pixels
        processing_class.image_processor.min_pixels = model_args.min_pixels
        processing_class.image_processor.patch_size = model_args.patch_size
    
    return processing_class


def get_model(script_args: ScriptArguments, training_args, model_args: ALLModelConfig):
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    model = model_args.model_name_or_path
    
    if script_args.use_unsloth:
        if not is_unsloth_available():
            raise ImportError(
                "Unsloth is not available and required for this reward function. Please install Unsloth with "
                "`pip install unsloth`"
            )
        from unsloth import FastLanguageModel, PatchFastRL
        algorithm = training_args.__class__.__name__.replace("TrainingArguments", "")
        PatchFastRL(algorithm, FastLanguageModel)
        
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model,
            **script_args.unsloth_configs,
            # max_seq_length=script_args.unsloth_configs['max_seq_length'],
            # load_in_4bit=script_args.unsloth_configs['load_in_4bit'],
            # fast_inference=script_args.unsloth_configs['fast_inference'], # uses vLLM
            # gpu_memory_utilization=script_args.unsloth_configs['gpu_memory_utilization'],
            # trust_remote_code=script_args.unsloth_configs['trust_remote_code'],
            # max_lora_rank=model_args.lora_r,
        )
        
        if model_args.use_peft:
            if not is_peft_available():
                raise ValueError(
                    "You need to have PEFT library installed in your environment, make sure to install `peft`. "
                    "Make sure to run `pip install -U peft`."
                )
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_args.lora_r,
                target_modules=model_args.lora_target_modules,
                lora_alpha=model_args.lora_alpha,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
                **script_args.unsloth_configs,
                # random_state=script_args.unsloth_configs['random_state'],
                # use_dora=script_args.unsloth_configs['use_dora'],
            )
        
    else:
        if is_adapter_model(model, model_args.model_revision) is True:
            logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
            peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
            if is_multimodal_model(model_args.model_name_or_path):
                base_model = AutoModelForImageTextToText.from_pretrained(
                    peft_config.base_model_name_or_path,
                    **model_kwargs,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    **model_kwargs,
                )
            model = get_peft_model(base_model, peft_config)
            # Use a standalone base model
            model = model.merge_and_unload()
            model_kwargs = None
        
    return model, model_kwargs