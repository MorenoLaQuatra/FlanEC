import comet_ml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import transformers
import datasets
import evaluate
from datasets import load_dataset
from yaml_config_override import add_arguments
from addict import Dict
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from peft import LoraConfig, LoraModel, TaskType, get_peft_model
import accelerate
from data_classes.hyporadise_dataset import HyporadiseDataset

# Removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load configuration from yaml file
config = add_arguments()
config = Dict(config)

# Define model and tokenizer
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.model.model_tag)
tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.model_tag)
tokenizer.save_pretrained(config.training.checkpoint_dir + "/tokenizer/")

print (model)

# add lora
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=16,
    target_modules='all-linear',
    bias="none",
    lora_dropout=0.05,
)

lora_model = get_peft_model(model, lora_config)
lora_model.save_pretrained(config.training.checkpoint_dir + "/base_lora_model/")

# Some info about the model
print(f"Model name: {lora_model.config.model_type}")
print(f"Model parameters: {lora_model.num_parameters() / 1e6:.2f}M")
# sum parameters
total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
percentage_trainable = trainable_params / total_params * 100
print(f"Model parameters (total): {total_params / 1e6:.2f}M")
print(f"Model parameters (trainable): {trainable_params / 1e6:.2f}M")
print(f"Percentage of trainable parameters: {percentage_trainable:.2f}%")

# Instantiate the dataset objects for each split
s2s_train_dataset = HyporadiseDataset(
    json_file_path=config.data.train_file,
    tokenizer_name_or_path=config.model.model_tag,
    max_length=config.data.max_input_length,
    truncation=config.data.truncation,
    prefix_prompt=config.data.prefix_prompt,
    suffix_prompt=config.data.suffix_prompt,
    return_scores=config.data.return_scores,
    is_test=False
)

# Split the training dataset into training and validation sets using torch.utils.data.random_split
train_size = int(config.data.train_val_split * len(s2s_train_dataset))
val_size = len(s2s_train_dataset) - train_size
# debug just 10 samples in validation set
s2s_train_dataset, s2s_val_dataset = torch.utils.data.random_split(s2s_train_dataset, [train_size, val_size])

print(f"*"*50)
print(f"Training dataset size: {len(s2s_train_dataset)}")
print(f"Validation dataset size: {len(s2s_val_dataset)}")
print(f"*"*50)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=lora_model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)


# Creating training arguments
training_arguments = transformers.Seq2SeqTrainingArguments(
    output_dir=config.training.checkpoint_dir,
    num_train_epochs=config.training.epochs,
    per_device_train_batch_size=config.training.batch_size,
    per_device_eval_batch_size=config.training.batch_size,
    warmup_ratio=config.training.warmup_ratio,
    weight_decay=config.training.weight_decay,
    logging_dir=config.training.log_dir,
    logging_steps=config.training.logging_steps,
    evaluation_strategy=config.training.evaluation_strategy,
    save_strategy=config.training.evaluation_strategy,
    eval_steps=config.training.eval_steps,
    save_steps=config.training.save_steps,
    load_best_model_at_end=config.training.load_best_model_at_end,
    learning_rate=config.training.learning_rate,
    dataloader_num_workers=config.training.dataloader_num_workers,
    save_total_limit=config.training.save_total_limit,
    no_cuda=not config.training.use_cuda,
    fp16=config.training.fp16,
    metric_for_best_model=config.training.metric_for_best_model,
    greater_is_better=config.training.greater_is_better,
    hub_model_id=config.training.hub_model_id,
    push_to_hub=config.training.push_to_hub,
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    label_names=["labels"],
)

# Define the compute_metrics function
wer = evaluate.load("wer")
cer = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    wer_output = wer.compute(predictions=pred_str, references=label_str)
    cer_output = cer.compute(predictions=pred_str, references=label_str)
    print("\n")

    return {
        "wer": round(wer_output, 4),
        "cer": round(cer_output, 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids

early_stopping = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0001)

# Instantiate the Trainer object
trainer = transformers.Seq2SeqTrainer(
    model=lora_model,
    args=training_arguments,
    train_dataset=s2s_train_dataset,
    eval_dataset=s2s_val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    data_collator=data_collator,
    callbacks=[early_stopping]
)

# Train the model
trainer.train()

# Save the model and tokenizer
lora_model = trainer.model
lora_model.save_pretrained(config.training.checkpoint_dir + "/best_model/")
tokenizer.save_pretrained(config.training.checkpoint_dir + "/best_model/")

# Push model to Hugging Face Hub if specified
if config.training.push_to_hub:
    trainer.push_to_hub()
    tokenizer.push_to_hub(repo_id=config.training.hub_model_id)
