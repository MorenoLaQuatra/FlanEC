import json
import random
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


source_mapping = {
    "test_atis": "ATIS",
    "test_chime4": "CHiME-4",
    "test_coraal": "CORAAL",
    "test_cv": "Common Voice",
    "test_lrs2": "LRS2",
    "test_ls_clean": "LibriSpeech Clean",
    "test_ls_other": "LibriSpeech Other",
    "test_swbd": "Switchboard",
    "test_td3": "TED-LIUM 3",
    "test_wsj_score": "Wall Street Journal",
    
    "train_atis": "ATIS",
    "train_chime4": "CHiME-4",
    "train_coraal": "CORAAL",
    "train_cv": "Common Voice",
    "train_lrs2": "LRS2",
    "train_other_500": "LibriSpeech Other",
    "train_swbd": "Switchboard",
    "train_td3": "TED-LIUM 3",
    "train_wsj_score": "Wall Street Journal",
}

class HyporadiseDataset(Dataset):
    '''
    This class is a wrapper around the Hyporadise dataset in the Hugging Face datasets library.
    '''
    def __init__(
        self,
        json_file_path: str,
        tokenizer_name_or_path: str,
        max_length: int = 1024,
        truncation: str = "only_first",
        prefix_prompt: str = "The following is a n-best list of ASR hypotheses for the given audio file:",
        suffix_prompt: str = "The correct transcription is:",
        return_scores: bool = False,
        is_test: bool = False,
        use_source: bool = False
    ):
        '''
        Args:
            json_file_path (str): The path to the JSON file containing the dataset.
            prefix_prompt (str): The prefix prompt to be used for the dataset.
            suffix_prompt (str): The suffix prompt to be used for the dataset.
            return_scores (bool): Whether to return the scores for the ASR hypotheses.
        '''
        # Load the dataset from the JSON file
        with open(json_file_path, 'r') as f:
            self.dataset = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.truncation = truncation
        
        self.prefix_prompt = prefix_prompt
        self.suffix_prompt = suffix_prompt
        self.return_scores = return_scores
        self.is_test = is_test
        self.use_source = use_source
            
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # create "text" field containing the prompt
            inputs = self.dataset[idx]['input']
            correct_output = self.dataset[idx]['output']
            if self.return_scores: am_scores = self.dataset[idx]['am_score']
        except Exception as e:
            print(f"Error at index {idx}: {self.dataset[idx].keys()} - {e}")
            # return another sample - call with random index
            return self[random.randint(0, len(self.dataset)-1)]
        
        if self.use_source:
            current_source = self.dataset[idx]['data_source'].split(".")[0]
            current_source = source_mapping[current_source]
            input_text = self.prefix_prompt.replace(":", f" from {current_source} dataset:\n\n")
            # print(input_text)
        else: input_text = self.prefix_prompt + "\n\n"
        for i, sentence in enumerate(inputs):
            if self.return_scores: text += f"{i+1}. {sentence} ({am_scores[i]})\n"
            else: input_text += f"{i+1}. {sentence}\n"
        input_text += f"{self.suffix_prompt}"
        
        # encode the text
        input_encoded = self.tokenizer(
            input_text,
            max_length=self.max_length, 
            truncation=self.truncation, 
            return_tensors='pt', 
            return_attention_mask=True, 
            padding='max_length'
        )
        
        if not self.is_test:
            output_text = correct_output + self.tokenizer.eos_token
    
            output_encoded = self.tokenizer(
                output_text, 
                max_length=self.max_length, 
                truncation=self.truncation, 
                return_tensors='pt', 
                return_attention_mask=True, 
                padding='max_length'
            )
        
            return {
                'input_ids': input_encoded['input_ids'].squeeze(),
                'attention_mask': input_encoded['attention_mask'].squeeze(),
                'labels': output_encoded['input_ids'].squeeze(),
                'input_text': input_text,
                'output_text': output_text,
            }
            
        else:
            return {
                'input_ids': input_encoded['input_ids'].squeeze(),
                'attention_mask': input_encoded['attention_mask'].squeeze(),
            }
            
            

class HyporadiseDatasetForCausalLM(Dataset):
    '''
    This class is a wrapper around the Hyporadise dataset in the Hugging Face datasets library, adapted for CausalLM.
    '''
    def __init__(
        self,
        json_file_path: str,
        tokenizer_name_or_path: str,
        max_length: int = 1024,
        truncation: str = "only_first",
        padding: bool = "max_length",
        prefix_prompt: str = "The following is a n-best list of ASR hypotheses for the given audio file:",
        suffix_prompt: str = "The correct transcription is:",
        return_scores: bool = False,
        is_test: bool = False,
        use_source: bool = False
    ):
        '''
        Args:
            json_file_path (str): The path to the JSON file containing the dataset.
            prefix_prompt (str): The prefix prompt to be used for the dataset.
            suffix_prompt (str): The suffix prompt to be used for the dataset.
            return_scores (bool): Whether to return the scores for the ASR hypotheses.
        '''
        # Load the dataset from the JSON file
        with open(json_file_path, 'r') as f:
            self.dataset = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side="left")
        self.max_length = max_length
        self.truncation = truncation
        
        self.prefix_prompt = prefix_prompt
        self.suffix_prompt = suffix_prompt
        self.return_scores = return_scores
        self.is_test = is_test
        self.use_source = use_source
        self.padding = padding
            
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # create "text" field containing the prompt
            inputs = self.dataset[idx]['input']
            correct_output = self.dataset[idx]['output']
            if self.return_scores: am_scores = self.dataset[idx]['am_score']
        except Exception as e:
            print(f"Error at index {idx}: {self.dataset[idx].keys()} - {e}")
            # return another sample - call with random index
            return self[random.randint(0, len(self.dataset)-1)]
        
        
        if self.use_source:
            current_source = self.dataset[idx]['data_source'].split(".")[0]
            current_source = source_mapping[current_source]
            prompt_text = self.prefix_prompt.replace(":", f" from {current_source} dataset:\n\n")
            input_text = self.prefix_prompt.replace(":", f" from {current_source} dataset:\n\n")
        else:
            prompt_text = self.prefix_prompt + "\n\n"
            input_text = self.prefix_prompt + "\n\n"
            
        for i, sentence in enumerate(inputs):
            if self.return_scores: current_text = f"{i+1}. {sentence} ({am_scores[i]})\n"
            else: current_text = f"{i+1}. {sentence}\n"
            prompt_text += current_text
            input_text += current_text
            
        input_text += f"{self.suffix_prompt} {correct_output} {self.tokenizer.eos_token}"
        prompt_text += f"{self.suffix_prompt}"
        
        # encode the text
        input_encoded = self.tokenizer(
            input_text,
            max_length=self.max_length, 
            truncation=self.truncation, 
            return_tensors='pt', 
            return_attention_mask=True, 
            padding=self.padding
        )
        
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_length, 
            truncation=self.truncation, 
            return_tensors='pt', 
            return_attention_mask=True, 
            padding=self.padding
        )
        
        return {
            'input_ids': input_encoded['input_ids'].squeeze(),
            'attention_mask': input_encoded['attention_mask'].squeeze(),
            'prompt_ids': prompt_encoded['input_ids'].squeeze(),
            'prompt_attention_mask':  prompt_encoded['attention_mask'].squeeze(),
            'labels': input_encoded['input_ids'].squeeze(),  # For CausalLM, labels are the same as input_ids
            'input_text': input_text,
            'prompt_text': prompt_text,
            'output_text': correct_output,
        }

# ------------------------------------
# In-context learning
# ------------------------------------

import json
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class HyporadiseDatasetICL(Dataset):
    '''
    This class is a wrapper around the Hyporadise dataset in the Hugging Face datasets library for In-context Learning,
    specifically designed for encoder-decoder models like Flan-T5.
    '''
    def __init__(
        self,
        json_file_path: str,
        json_train_file_path: str,
        tokenizer_name_or_path: str,
        max_length: int = 1024,
        truncation: str = "only_first",
        prefix_prompt: str = "The following is a n-best list of ASR hypotheses for the given audio file:",
        suffix_prompt: str = "The correct transcription is:",
        return_scores: bool = False,
        is_test: bool = False,
        use_source: bool = False
    ):
        '''
        Args:
            json_file_path (str): The path to the JSON file containing the dataset.
            json_train_file_path (str): The path to the JSON file containing the training dataset.
            prefix_prompt (str): The prefix prompt to be used for the dataset.
            suffix_prompt (str): The suffix prompt to be used for the dataset.
            return_scores (bool): Whether to return the scores for the ASR hypotheses.
        '''
        # Load the dataset from the JSON file
        with open(json_file_path, 'r') as f:
            self.dataset = json.load(f)
            
        with open(json_train_file_path, 'r') as f:
            self.train_dataset = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_length = max_length
        self.truncation = truncation
        
        self.prefix_prompt = prefix_prompt
        self.suffix_prompt = suffix_prompt
        self.return_scores = return_scores
        self.is_test = is_test
        self.use_source = use_source
        
    def find_top_n_similar(self, test_hypothesis, n=5):
        '''
        Function to find top-n similar hypotheses using word matching
        '''
        def hypothesis_similarity(hyp1, hyp2):
            words1 = set(hyp1.split())
            words2 = set(hyp2.split())
            return len(words1 & words2)

        similarities = []
        test_first_hypothesis = test_hypothesis.split('\n')[0]
        
        for i, train_example in enumerate(self.train_dataset):
            train_first_hypothesis = train_example['input'][0]
            sim = hypothesis_similarity(test_first_hypothesis, train_first_hypothesis)
            similarities.append((sim, i))

        # Sort by similarity score in descending order and select top-n
        top_n = sorted(similarities, key=lambda x: x[0], reverse=True)[:n]
        top_n_indices = [idx for _, idx in top_n]
        
        return top_n_indices
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # create "text" field containing the prompt
            inputs = self.dataset[idx]['input']
            correct_output = self.dataset[idx]['output']
            if self.return_scores: am_scores = self.dataset[idx]['am_score']
        except Exception as e:
            print(f"Error at index {idx}: {self.dataset[idx].keys()} - {e}")
            # return another sample - call with random index
            return self[random.randint(0, len(self.dataset)-1)]
        
        if self.use_source:
            current_source = self.dataset[idx]['data_source'].split(".")[0]
            current_source = source_mapping[current_source]
            input_text = self.prefix_prompt.replace(":", f" from {current_source} dataset:\n\n")
        else:
            input_text = self.prefix_prompt + "\n\n"
        
        N_SAMPLES = 0
        if N_SAMPLES > 0:
            # Find top-n similar examples from the training set
            example_indices = self.find_top_n_similar("\n".join(inputs) + '\n', n=3)
            
            input_text += f"Here are some similar examples from the training set:\n\n"
            
            # Add examples to input_text
            for example_idx in example_indices:
                example = self.train_dataset[example_idx]
                example_inputs = example['input']
                example_output = example['output']
                for i, sentence in enumerate(example_inputs[:5]):
                    if self.return_scores:
                        input_text += f"- {sentence} ({example['am_score'][i]})\n"
                    else:
                        input_text += f"- {sentence}\n"
                input_text += f"{self.suffix_prompt} \"{example_output}\"\n\n"
        
            input_text += f"Now, here is the test example:\n\n"
        
        for i, sentence in enumerate(inputs[:5]):
            if self.return_scores:
                input_text += f"- {sentence} ({am_scores[i]})\n"
            else:
                input_text += f"- {sentence}\n"
        input_text += f"{self.suffix_prompt}"
        
        # encode the text
        input_encoded = self.tokenizer(
            input_text,
            max_length=self.max_length, 
            truncation=self.truncation, 
            return_tensors='pt', 
            return_attention_mask=True, 
            padding='max_length'
        )
        
        if not self.is_test:
            output_text = correct_output + self.tokenizer.eos_token
    
            output_encoded = self.tokenizer(
                output_text, 
                max_length=self.max_length, 
                truncation=self.truncation, 
                return_tensors='pt', 
                return_attention_mask=True, 
                padding='max_length'
            )
        
            return {
                'input_ids': input_encoded['input_ids'].squeeze(),
                'attention_mask': input_encoded['attention_mask'].squeeze(),
                'labels': output_encoded['input_ids'].squeeze(),
                'input_text': input_text,
                'output_text': output_text,
            }
            
        else:
            return {
                'input_ids': input_encoded['input_ids'].squeeze(),
                'attention_mask': input_encoded['attention_mask'].squeeze(),
            }