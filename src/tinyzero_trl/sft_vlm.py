# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vlm_smol_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir sft-smol-vlm-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random

import hydra
from accelerate.state import PartialState
from omegaconf import DictConfig, OmegaConf
from transformers import HfArgumentParser, LlavaForConditionalGeneration, Idefics3ForConditionalGeneration, Qwen2VLProcessor
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl import SFTConfig, SFTTrainer, get_peft_config
from trl.trainer.utils import empty_cache

from tinyzero_trl.utils import (ALLModelConfig, ScriptArguments, get_datasets,
                                get_model, get_processing_class, setup_logger)


@hydra.main(version_base='1.1', config_path="../../recipes", config_name="hydra_default")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # print(cfg)

    parser = HfArgumentParser((ScriptArguments, SFTConfig, ALLModelConfig))
    script_args, training_args, model_args = parser.parse_dict(cfg, allow_extra_keys=True)

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
    logger.info(f"Data parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ###################
    # Model & processing_class
    ###################
    logger.info("*** Loading pretrained model and processing_class ***")
    # MODEL
    model, model_kwargs = get_model(script_args, training_args, model_args)
    # processing_class
    processing_class = get_processing_class(script_args, training_args, model_args)
    
    
    ###############
    # Load Dataset
    ###############
    logger.info("*** Loading datasets ***")
    raw_datasets = get_datasets(
        script_args,
        splits=script_args.dataset_splits,
        configs=script_args.dataset_configs,
        columns_to_keep=None,
    )
    
    # split the dataset into train and test if requires evaluation
    if training_args.do_eval and "test" not in raw_datasets:
        raw_datasets = raw_datasets.train_test_split(test_size=0.1)
    
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
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    from tinyzero_trl.utils.script_utils import (format_vlm_sft)

    with PartialState().main_process_first():
        raw_datasets = raw_datasets.map(
            format_vlm_sft,
            fn_kwargs={
                "processing_class": processing_class,
                "system_prompt": script_args.system_prompt,
            },
            num_proc=1 if training_args.debug else script_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Formatting training datasets"
        )
    
    # Log a few random samples from the training set:
    if PartialState().is_main_process:
        print(raw_datasets)
        for index in random.sample(range(len(raw_datasets["train"])), 2):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")

    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("test", None)


    ########################################################
    # Create a data collator to encode text and image pairs
    ########################################################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [example["texts"] for example in examples]
        images = [example["images"] for example in examples]
        videos = [example["videos"] for example in examples]

        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        # Tokenize the texts and process the images
        batch = processing_class(text=texts, images=images, videos=videos, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processing_class.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(model, Idefics3ForConditionalGeneration):
            image_token_ids = [processing_class.tokenizer.additional_special_tokens_ids[
                    processing_class.tokenizer.additional_special_tokens.index("<image>")
                ]
            ]
        elif isinstance(processing_class, Qwen2VLProcessor):
            image_token_ids = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_token_ids = [processing_class.tokenizer.convert_tokens_to_ids(processing_class.image_token)]
        # Mask image token IDs in the labels
        for image_token_id in image_token_ids:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels  # Add labels to the batch

        return batch


    #############################
    # Instantiate VLM-SFT trainer
    #############################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class.tokenizer,
        peft_config=get_peft_config(model_args) if not script_args.use_unsloth else None,
    )
    # Add multimodal model support
    if model_args.freeze_vision:
        trainer.model.visual.requires_grad_(requires_grad=False)
    elif model_args.freeze_llm:
        trainer.model.model.requires_grad_(requires_grad=False)


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