import os
import torch
import colorlog
from huggingface_hub import list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from accelerate.state import PartialState
from peft import get_peft_model, PeftConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from trl import (
    get_kbit_device_map,
    get_quantization_config,
)
from transformers.utils import (
    is_peft_available,
)
from .import_utils import is_unsloth_available
from .data_utils import DataArguments


logger = colorlog.getLogger(__name__)


INTERNET_CONNECTION = True
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


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


def get_tokenizer(
    data_args: DataArguments, 
    training_args, 
    model_args, 
    auto_set_chat_template=True
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    tokenizer.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    # Set reasonable default for models without max length
    # if tokenizer.model_max_length > 100_000:
    #     tokenizer.model_max_length = training_args.max_prompt_length + training_args.max_completion_length
    # assert tokenizer.chat_template is not None, "Needs chat template!"
    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    
    return tokenizer


def get_model(data_args: DataArguments, training_args, model_args):
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
    
    if data_args.use_unsloth:
        if not is_unsloth_available():
            raise ImportError(
                "E2B is not available and required for this reward function. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )
        from unsloth import FastLanguageModel, PatchFastRL
        algorithm = training_args.__class__.__name__.replace("TrainingArguments", "")
        PatchFastRL("algorithm", FastLanguageModel)
        
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model,
            **data_args.unsloth_configs,
            # max_seq_length=data_args.unsloth_configs['max_seq_length'],
            # load_in_4bit=data_args.unsloth_configs['load_in_4bit'],
            # fast_inference=data_args.unsloth_configs['fast_inference'], # uses vLLM
            # gpu_memory_utilization=data_args.unsloth_configs['gpu_memory_utilization'],
            # trust_remote_code=data_args.unsloth_configs['trust_remote_code'],
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
                **data_args.unsloth_configs,
                # random_state=data_args.unsloth_configs['random_state'],
                # use_dora=data_args.unsloth_configs['use_dora'],
            )
        
    else:
        if is_adapter_model(model, model_args.model_revision) is True:
            logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
            peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                **model_kwargs,
            )
            model = get_peft_model(base_model, peft_config)
            # Use a standalone base model
            model = model.merge_and_unload()
            model_kwargs = None
        
    return model, model_kwargs