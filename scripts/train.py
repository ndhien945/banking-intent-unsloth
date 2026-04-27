import yaml
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import os

PROMPT_TEMPLATE = """You are an AI assistant for a banking customer service. Classify the user's intent into exactly one of the known banking categories.

User Query: {}
Intent: {}"""

def format_prompts(examples):
    texts = []
    for text, intent in zip(examples['text'], examples['intent_name']):
        full_text = PROMPT_TEMPLATE.format(text, intent) + " <|end_of_text|>"
        texts.append(full_text)
    return {"text": texts}


def main():
    print("1. Read configs/train.yaml...")
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"2. Initializing model: {config['model']['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['base_model'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
    )

    print("3. Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        target_modules=config['lora']['target_modules'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing="unsloth",
    )

    print("4. Loading and formatting training data...")
    df_train = pd.read_csv(config['data']['train_path'])
    dataset = Dataset.from_pandas(df_train, preserve_index=False)       
    dataset = dataset.map(format_prompts, batched=True)
    
    dataset = dataset.select_columns(["text"])

    print("5. Initializing SFT Trainer and starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text", 
        max_seq_length=config['model']['max_seq_length'],
        args=TrainingArguments(
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            num_train_epochs=config['training']['epochs'],
            learning_rate=config['training']['learning_rate'],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=config['training']['optimizer'],
            output_dir=config['training']['output_dir'],
            seed=3407, 
        ),
    )
    
    trainer.train()

    print("6. Saving model...")
    
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    print(f"Done")

if __name__ == "__main__":
    main()