#!/usr/bin/env python
"""
Fine-tuning OpenAI gpt-oss-20b for multilingual reasoning using TRL.

This script demonstrates how to fine-tune OpenAI's open-weight reasoning model
to reason effectively in multiple languages by adding a new "reasoning language"
option to the model's system prompt and applying supervised fine-tuning.

Based on: https://huggingface.co/docs/trl/tutorials/multilingual_reasoner

Usage:
    # Full training
    python multilingual_reasoner.py \
        --model_name_or_path openai/gpt-oss-20b \
        --dataset_name HuggingFaceH4/Multilingual-Thinking \
        --learning_rate 2.0e-4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing \
        --max_length 2048 \
        --output_dir gpt-oss-20b-multilingual-reasoner \
        --use_peft \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_target_modules all-linear \
        --push_to_hub

    # Inference only
    python multilingual_reasoner.py \
        --inference_only \
        --peft_model_id gpt-oss-20b-multilingual-reasoner

Requirements:
    - torch
    - trl>=0.20.0
    - peft>=0.17.0
    - transformers>=4.55.0
    - trackio
    - datasets
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, get_peft_config
from peft import get_peft_model


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    """
    Arguments specific to the multilingual reasoner training script.
    """

    dataset_name: str = field(
        default="HuggingFaceH4/Multilingual-Thinking",
        metadata={"help": "Dataset name from Hugging Face Hub"},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"},
    )
    inference_only: bool = field(
        default=False,
        metadata={"help": "Only run inference with a pre-trained model"},
    )
    peft_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "PEFT model ID for inference (if different from output_dir)"},
    )
    reasoning_language: str = field(
        default="German",
        metadata={"help": "Language for reasoning during inference"},
    )
    inference_prompt: str = field(
        default="¿Cuál es el capital de Australia?",
        metadata={"help": "User prompt for inference testing"},
    )


########################
# Helper functions
########################
def load_and_prepare_model(model_config, training_args):
    """
    Load and prepare the model for training.

    Args:
        model_config: ModelConfig object with model parameters
        training_args: SFTConfig object with training arguments

    Returns:
        Loaded model
    """
    print(f"Loading model: {model_config.model_name_or_path}")

    # Configure model loading with Mxfp4Config for OpenAI gpt-oss models
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation=model_config.attn_implementation or "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map="auto",
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs
    )

    return model


def generate_sample_response(model, tokenizer, prompt="¿Cuál es el capital de Australia?"):
    """
    Generate a sample response to test the model.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use
        prompt: User prompt
    """
    print("\n" + "="*80)
    print("Generating sample response...")
    print("="*80)

    messages = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(output_ids)[0]

    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")
    print("="*80 + "\n")


def run_inference(script_args, model_config):
    """
    Run inference with the fine-tuned model.

    Args:
        script_args: ScriptArguments object
        model_config: ModelConfig object
    """
    print("\n" + "="*80)
    print("Running inference with fine-tuned model...")
    print("="*80)

    # Load tokenizer
    model_name = model_config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model
    model_kwargs = dict(
        attn_implementation=model_config.attn_implementation or "eager",
        torch_dtype="auto",
        use_cache=True,
        device_map="auto"
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Merge fine-tuned weights
    peft_model_id = script_args.peft_model_id
    if peft_model_id:
        print(f"Loading PEFT model from: {peft_model_id}")
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model = model.merge_and_unload()
    else:
        print("No PEFT model specified, using base model")
        model = base_model

    # Prepare messages
    system_prompt = f"reasoning language: {script_args.reasoning_language}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": script_args.inference_prompt},
    ]

    # Tokenize and generate
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": None,
        "top_k": None
    }

    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]

    print(f"\nReasoning Language: {script_args.reasoning_language}")
    print(f"User Prompt: {script_args.inference_prompt}")
    print(f"\nResponse:\n{response}")
    print("="*80 + "\n")

    # Try other languages
    print("\nTesting with other languages...")
    for lang, prompt in [
        ("Chinese", "What is the national symbol of Canada?"),
        ("Hindi", "What is the tallest mountain in the world?"),
        ("French", "Qui a écrit 'Les Misérables'?"),
    ]:
        print(f"\n--- {lang} ---")
        system_prompt = f"reasoning language: {lang}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(input_ids, **gen_kwargs)
        response = tokenizer.batch_decode(output_ids)[0]
        print(f"Prompt: {prompt}")
        print(f"Response:\n{response}\n")
        print("="*80 + "\n")


def main(script_args, training_args, model_config):
    """Main training and inference pipeline."""

    # Enable logging in a Hugging Face Space
    os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

    # Inference only mode
    if script_args.inference_only:
        run_inference(script_args, model_config)
        return

    # Load dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_train_split)
    print(f"Dataset loaded with {len(dataset)} examples")

    # Display a sample
    print("\nSample from dataset:")
    print(f"Number of messages: {len(dataset[0]['messages'])}")
    print(dataset[0])
    
    # split train and eval
    train_dataset = dataset.train_test_split(test_size=0.1)
    eval_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    

    # Load model
    model = load_and_prepare_model(model_config, training_args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    # Generate a sample response before training (optional)
    print("\n" + "="*80)
    print("Before Training - Sample Generation")
    print("="*80)
    generate_sample_response(model, tokenizer)

    # Get PEFT config
    peft_config = get_peft_config(model_config)
    print(f"PEFT config: {peft_config}")
    
    # get peft model
    peft_model = get_peft_model(model, peft_config)
    print(peft_model.print_trainable_parameters())  
    
    # Initialize the SFT trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # peft_config=peft_config,
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()
    print("\nTraining completed!")

    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

    # Push to Hub if requested
    if training_args.push_to_hub:
        print("Pushing model to Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        print(f"Model pushed to the Hub")

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"\nTo run inference, restart your kernel and run:")
    print(f"python multilingual_reasoner.py --inference_only --peft_model_id {training_args.output_dir}")


if __name__ == "__main__":
    # Parse arguments using TrlParser
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # Set default model if not specified
    if model_config.model_name_or_path is None:
        model_config.model_name_or_path = "openai/gpt-oss-20b"

    # Set default LoRA target modules for gpt-oss if using PEFT
    if model_config.use_peft and model_config.lora_target_modules is None:
        model_config.lora_target_modules = "all-linear"

    # Run main function
    main(script_args, training_args, model_config)
