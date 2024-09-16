import os
import torch
import transformers
import datasets
import evaluate
from yaml_config_override import add_arguments
from addict import Dict
from tqdm import tqdm

import editdistance

from data_classes.hyporadise_dataset import HyporadiseDataset

# Load configuration from yaml file
config = add_arguments()
config = Dict(config)

# Load model and tokenizer
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.model.model_tag)
tokenizer = transformers.AutoTokenizer.from_pretrained(config.inference.specific_checkpoint_dir)
model.load_adapter(config.inference.specific_checkpoint_dir)
num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_parameters}")

if config.inference.specific_test_file != "":
    data_file = config.inference.specific_test_file
else:
    data_file = config.data.test_file

# Load the test dataset
s2s_test_dataset = HyporadiseDataset(
    json_file_path=data_file,
    tokenizer_name_or_path=config.model.model_tag,
    max_length=config.data.max_input_length,
    truncation=config.data.truncation,
    prefix_prompt=config.data.prefix_prompt,
    suffix_prompt=config.data.suffix_prompt,
    return_scores=config.data.return_scores,
)

# Define metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Define function for evaluation
def compute_metrics(predictions, references):
    wer_output = wer_metric.compute(predictions=predictions, references=references)
    cer_output = cer_metric.compute(predictions=predictions, references=references)
    
    # Compute edit distance
    splitted_predictions = [pred.split() for pred in predictions]
    splitted_references = [ref.split() for ref in references]
    
    ed_dist_scores = []
    for pred, ref in zip(splitted_predictions, splitted_references):
        ed_dist_scores.append(editdistance.eval(pred, ref) / len(ref))
    wer_score = sum(ed_dist_scores) / len(ed_dist_scores)

    return {
        "wer": round(wer_output, 4),
        "cer": round(cer_output, 4),
        "wer_editdistance": round(wer_score, 4),
    }

# Inference function
def evaluate_model(model, tokenizer, dataset):
    model.eval()
    predictions = []
    references = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = tokenizer(batch['input_text'], return_tensors='pt', truncation=True, padding=True)
        # device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=config.data.max_output_length)
        pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(pred_str)
        output_str = batch['output_text']
        output_str = [item.replace("<s>", "").replace("</s>", "").strip() for item in output_str]
        references.extend(output_str)
        
    metrics = compute_metrics(predictions, references)
    return metrics, predictions, references

# Compute and print metrics on the test set
metrics, preds, refs = evaluate_model(model, tokenizer, s2s_test_dataset)

print(f"Word Error Rate (WER): {metrics['wer'] * 100 :.1f}")
print(f"Character Error Rate (CER): {metrics['cer'] * 100 :.1f}")
print(f"Wer Editdistance: {metrics['wer_editdistance'] * 100 :.1f}")
print("-"*50 + "\n\n")

