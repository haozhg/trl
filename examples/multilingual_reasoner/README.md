# Multilingual Reasoner

Fine-tune OpenAI's gpt-oss-20b to reason in multiple languages.

## Quick Start

### Training with YAML Config (Recommended)

**Multi-GPU training:**
```bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config.yaml
```

**Single GPU training:**
```bash
accelerate launch --config_file examples/accelerate_configs/single_gpu.yaml \
    examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config_single_gpu.yaml
```

### Training with Command-Line Arguments

```bash
python examples/multilingual_reasoner/multilingual_reasoner.py \
    --model_name_or_path openai/gpt-oss-20b \
    --dataset_name HuggingFaceH4/Multilingual-Thinking \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --output_dir gpt-oss-20b-multilingual-reasoner \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules all-linear
```

### Inference

**With YAML config:**
```bash
python examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config_inference.yaml
```

**With command-line arguments:**
```bash
python examples/multilingual_reasoner/multilingual_reasoner.py \
    --inference_only \
    --peft_model_id gpt-oss-20b-multilingual-reasoner \
    --reasoning_language German \
    --inference_prompt "¿Cuál es el capital de Australia?"
```

## Available Configs

See [recipes/](recipes/) directory:

- **[config.yaml](recipes/config.yaml)** - Multi-GPU training (8 GPUs)
- **[config_single_gpu.yaml](recipes/config_single_gpu.yaml)** - Single GPU training
- **[config_inference.yaml](recipes/config_inference.yaml)** - Inference only
- **[config_full.yaml](recipes/config_full.yaml)** - Full config with all options
- **[README.md](recipes/README.md)** - Detailed documentation

## What This Does

This script fine-tunes OpenAI's gpt-oss-20b model to:
1. Accept a "reasoning language" in the system prompt
2. Generate chain-of-thought reasoning in that language
3. Provide final answers in the user's language

### Example Output

**Input:**
- System: "reasoning language: German"
- User: "¿Cuál es el capital de Australia?" (Spanish)

**Output:**
- Reasoning (German): "Okay, der Benutzer fragt nach der Hauptstadt Australiens..."
- Response (Spanish): "La capital de Australia es Canberra..."

## Requirements

```bash
pip install torch trl>=0.20.0 peft>=0.17.0 transformers>=4.55.0 datasets trackio
```

## Key Features

- ✅ Uses TRL's `TrlParser` for unified argument handling
- ✅ Supports YAML configs and command-line args
- ✅ LoRA fine-tuning for memory efficiency
- ✅ Special handling for MoE architecture (gpt-oss-20b)
- ✅ Multi-language reasoning support
- ✅ Accelerate integration for distributed training
- ✅ Mxfp4 quantization support

## Arguments

The script accepts three types of arguments:

1. **ScriptArguments** - Script-specific settings
   - `dataset_name`, `dataset_train_split`
   - `inference_only`, `peft_model_id`
   - `reasoning_language`, `inference_prompt`

2. **ModelConfig** - Model and LoRA settings
   - `model_name_or_path`, `attn_implementation`
   - `use_peft`, `lora_r`, `lora_alpha`, `lora_target_modules`
   - Quantization options

3. **SFTConfig** - Training parameters
   - `learning_rate`, `num_train_epochs`
   - `per_device_train_batch_size`, `gradient_accumulation_steps`
   - `max_length`, `output_dir`
   - All standard TrainingArguments

## Training Time

- Single A100 80G: ~2 hours

## Resources

- 📚 [Tutorial](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
- 🗃️ [Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- 🤖 [Base Model](https://huggingface.co/openai/gpt-oss-20b)
- 📖 [TRL Documentation](https://huggingface.co/docs/trl)
