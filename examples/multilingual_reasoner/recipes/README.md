# Multilingual Reasoner Training Recipes

This directory contains YAML configuration files for training and running inference with the multilingual reasoning model based on OpenAI's gpt-oss-20b.

## Configuration Files

### `config.yaml` - Multi-GPU Training
Default configuration optimized for multi-GPU training (8 GPUs).

**Usage:**
```bash
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config.yaml
```

**Key settings:**
- Batch size: 4 per device
- Gradient accumulation: 4 steps
- Effective batch size: 4 × 4 × 8 = 128
- LoRA rank: 8


**Usage:**
```bash
# With accelerate
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config.yaml

# Or without accelerate
python examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config.yaml
```

**Key settings:**
- Batch size: 2 per device
- Gradient accumulation: 8 steps
- Effective batch size: 2 × 8 = 16
- LoRA rank: 8

### `config_inference.yaml` - Inference Only
Configuration for running inference with a fine-tuned model.

**Usage:**
```bash
python examples/multilingual_reasoner/multilingual_reasoner.py \
    --config examples/multilingual_reasoner/recipes/config_inference.yaml
```

**Key settings:**
- `inference_only: true`
- `peft_model_id`: Path to fine-tuned model
- `reasoning_language`: Language for internal reasoning
- `inference_prompt`: Question to ask

## Customization

### Adjusting Memory Usage

If you encounter OOM errors, try:

1. **Reduce batch size:**
   ```yaml
   per_device_train_batch_size: 2  # or 1
   ```

2. **Increase gradient accumulation:**
   ```yaml
   gradient_accumulation_steps: 8  # or 16
   ```

3. **Reduce sequence length:**
   ```yaml
   max_length: 1024  # default is 2048
   ```

4. **Reduce LoRA rank:**
   ```yaml
   lora_r: 4  # default is 8
   ```

### Pushing to Hub

To push your trained model to the Hugging Face Hub:

```yaml
push_to_hub: true
hub_model_id: your-username/gpt-oss-20b-multilingual-reasoner
```

### Changing Dataset

To use a different dataset:

```yaml
dataset_name: your-username/your-dataset
dataset_train_split: train
```

### Advanced LoRA Configuration

```yaml
use_peft: true
lora_r: 16                    # Higher rank = more parameters, better performance
lora_alpha: 32                # Scaling factor
lora_dropout: 0.1             # Dropout for regularization
lora_target_modules: all-linear  # Target all linear layers
use_rslora: true              # Use Rank-Stabilized LoRA
use_dora: false               # Use Weight-Decomposed LoRA (slower but better)
```


## Inference Examples

### German Reasoning, Spanish Question
```yaml
reasoning_language: German
inference_prompt: "¿Cuál es el capital de Australia?"
```

### French Reasoning, English Question
```yaml
reasoning_language: French
inference_prompt: "What is the tallest mountain in the world?"
```

### Chinese Reasoning, French Question
```yaml
reasoning_language: Chinese
inference_prompt: "Qui a écrit 'Les Misérables'?"
```

## Expected Training Time

On H100 80GB:
- Single GPU: ~18 minutes
- 8 GPUs: ~3-4 minutes

On A100 40GB:
- Single GPU: ~25-30 minutes

## Monitoring Training

The training progress is logged to Trackio by default. You can change this:

```yaml
report_to: wandb  # or tensorboard, none
```

## Troubleshooting

### OOM Errors
See "Adjusting Memory Usage" section above.

### Slow Training
- Enable `tf32: true` (already enabled by default)
- Ensure `gradient_checkpointing: true`
- Use `attn_implementation: flash_attention_2` if available

### Dataset Loading Issues
Make sure you're logged in to Hugging Face:
```bash
huggingface-cli login
```

## References

- [Tutorial](https://huggingface.co/docs/trl/tutorials/multilingual_reasoner)
- [Dataset](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
- [Base Model](https://huggingface.co/openai/gpt-oss-20b)
