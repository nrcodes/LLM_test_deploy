#!/usr/bin/env python3
"""
Simple, reliable Ollama JSONL trainer with better error handling
"""

import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOllamaTrainer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.output_dir = "simple_fine_tuned_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_jsonl(self, file_path: str):
        """Load JSONL data"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def prepare_data(self, jsonl_data):
        """Prepare data for training"""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formatted_texts = []
        for item in jsonl_data:
            prompt = item['prompt']
            response = item['response']
            
            text = f"Human: {prompt}\nAssistant: {response}{tokenizer.eos_token}"
            formatted_texts.append(text)

        dataset = Dataset.from_dict({"text": formatted_texts})

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=1024,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        self.tokenizer = tokenizer
        return tokenized_dataset
    
    def setup_model(self):
        """Setup model with minimal configuration"""
        logger.info(f"Loading model: {self.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        
        try:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            logger.info("LoRA applied successfully")
        except Exception as e:
            logger.warning(f"LoRA failed: {e}. Using base model.")
        
        return model
    
    def train(self, dataset, model):
        """Train the model"""
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            learning_rate=5e-5,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            optim="adamw_torch",
            report_to="none",
            dataloader_num_workers=0,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        return trainer
    
    def run_pipeline(self, jsonl_file: str):
        """Run the complete pipeline"""
        try:
            logger.info("Step 1: Loading data...")
            data = self.load_jsonl(jsonl_file)
            
            logger.info("Step 2: Preparing dataset...")
            dataset = self.prepare_data(data)
            
            logger.info("Step 3: Setting up model...")
            model = self.setup_model()
            
            logger.info("Step 4: Training...")
            trainer = self.train(dataset, model)
            
            logger.info("Pipeline completed successfully!")
            return trainer
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    trainer = SimpleOllamaTrainer()
    
    jsonl_file = "training_data.jsonl"    
    trainer.run_pipeline(jsonl_file)

if __name__ == "__main__":
    main()