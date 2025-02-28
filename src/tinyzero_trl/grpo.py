# import debugpy; debugpy.connect(("localhost", 9501))
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import datasets
import torch
import random


from omegaconf import  OmegaConf, DictConfig
from dataclasses import dataclass, field
from transformers.trainer_utils import set_seed, get_last_checkpoint
from transformers import (
    HfArgumentParser
)
from accelerate.state import PartialState
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    get_peft_config,
)
from trl.trainer.utils import empty_cache
from utils import (
    setup_logger,
    is_adapter_model,
    get_current_device,
    get_datasets,
    DataArguments,
    get_tokenizer,
    get_model
)
from rewards import get_reward_functions


@dataclass
class GRPOTrainingArguments(GRPOConfig):
    """
    Training arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": (
                "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', "
                "'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'."
            )
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    


@hydra.main(version_base='1.1', config_path="../../recipes", config_name="hydra_default")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # print(cfg)
    
    parser = HfArgumentParser((DataArguments, GRPOTrainingArguments, ModelConfig))
    data_args, training_args, model_args = parser.parse_dict(cfg, allow_extra_keys=True)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    ###############
    # Setup logging
    ###############
    logger = setup_logger(training_args)
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ###################
    # Model & Tokenizer
    ###################
    logger.info("*** Loading pretrained model and tokenizer ***")
    # MODEL
    model, model_kwargs = get_model(training_args, model_args)
    # TOKENIZER
    tokenizer = get_tokenizer(data_args, training_args, model_args)
    
    
    ###############
    # Load Dataset
    ###############
    logger.info("*** Loading datasets ***")
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=None,
    )
    
    # # split the dataset into train and test
    # if all(["test" not in split for split in data_args.dataset_splits]):
    #     raw_datasets = raw_datasets.train_test_split(test_size=0.01)
    
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    if training_args.debug:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(100))
    
    
    #################
    # Format Dataset
    #################
    # def formatting_datasets(example, tokenizer):
    #     prompt = [{"role":"user", "content": example["prompt"]}]
    #     example["prompt"] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    #     return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}
    from utils.data_utils import format_countdown
    
    with PartialState().main_process_first():
        raw_datasets = raw_datasets.map(
            format_countdown,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=1 if training_args.debug else data_args.preprocessing_num_workers,
            remove_columns=column_names if training_args.remove_unused_columns else None,
            desc="Formatting training datasets"
        )
    
    # Log a few random samples from the training set:
    if PartialState().is_main_process:
        for index in random.sample(range(len(raw_datasets["train"])), 2):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")

    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("test", None)
    
    
    ############################
    # Instantiate GRPO trainer
    ############################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    trainer = GRPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=get_reward_functions(training_args),
        peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)        
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")

    
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Saving model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


    ##########
    # Evaluate
    ##########
    empty_cache()
    if training_args.do_eval and "test" in raw_datasets:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Evaluating complete ***")



if __name__ == "__main__":
    main()
    
