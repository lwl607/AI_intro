'''
HW3: Sentiment Analysis with Deep Learning

In this homework, we will explore the fascinating field of sentiment analysis using deep learning techniques. 
Specifically, we will focus on multi-class classification, where the goal is to predict each sentence from social media 
as belonging to the label.

Label definition:
0 -> Negative
1 -> Neutral
2 -> Positive
'''

import os
import re
import gc
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FLOPs estimation
def estimate_flops(hidden_size: int, num_layers: int, seq_len: int, batch_size: int) -> float:
    '''
    Roughly estimate the number of floating point operations (FLOPs) 
    per training step for models

    Args:
        hidden_size: model embedding dimension
        num_layers: number of encoder layers
        seq_len: number of tokens per input
        batch_size: number of samples processed per step

    Returns:
        Estimated FLOPs per training step (in GFLOPs)
    '''
    pass


# Dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        """
        Step1. Load the CSV file using pandas -> HINT: use pd.read_csv(csv_path)
        Step2. Extract text and label columns -> HINT: df["text"].tolist(), df["label"].tolist()
        Step3. Store tokenizer and max_length for later use

        Args:
            csv_path: Path to the CSV file (with columns 'text' and 'label')
            tokenizer: Pre-trained tokenizer from Hugging Face
            max_length: Maximum token length for padding/truncation
        """
        # Load CSV and extract columns
        df = pd.read_csv(csv_path)
        # Expecting columns 'text' and 'label'
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

        # Store tokenizer and max length for later tokenization
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns:
            Total number of samples in the dataset -> HINT: len(self.texts)
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Step1. Select text and label by index -> HINT: text = self.texts[idx]; label = self.labels[idx]
        Step2. Tokenize the text -> HINT: use self.tokenizer with truncation, padding, max_length, return_tensors="pt"
        Step3. Convert results to proper tensor format -> HINT: enc["input_ids"].squeeze(0)
        Step4. Return a dictionary

        Returns:
            One sample (tokenized text and label)
        """
        # Select the raw text and label
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text. Return tensors as PyTorch tensors and squeeze the batch dim.
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


# Model Architecture Components
class CustomBlock(nn.Module): 
    def __init__(self, hidden_size, num_labels):
        """, 
        Initialize the layers and parameters of this block.

        HINTS:
        - Always call super().__init__() first to inherit from nn.Module.
        - Define any sub-layers you need (e.g., Linear, Conv1d, Dropout).
        - Store any configuration parameters (e.g., hidden size, kernel size).
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)
        self.hidden_size = hidden_size
        self.num_labels = num_labels

    def forward(self, x): 
        """
        Define how data moves through the block.

        Args:
            ... : input tensor(s)

        Returns:
            The transformed output tensor.

        HINTS:
        - The forward pass describes the actual computation.
        - Use the layers defined in __init__ to process the input.
        - Make sure to return the final output tensor.
        """

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Example of Custom Block
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Model Config
class SentimentConfig(PretrainedConfig):
    model_type = "..."  # describe this model type

    def __init__(
        self,
        model_name="bert-base-uncased", # name of pre-trained model backbone
        num_labels=3,     # number of output classes (Negative, Neutral, Positive)
        head="mlp",       # classifier head
                          # other hyperparameters
        dropout=0.2,
        lr_encoder=2e-5,
        lr_head=1e-3,
        warmup_ratio=0.1,
        **kwargs,
    ):
        # Always call the parent class initializer first
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.dropout = dropout
        
        '''
        Save all hyperparameters to self
        
        Example:
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        ...
        self.other_hyperparam = other_hyperparam
        
        These attributes will be automatically saved in config.json
        when you call `config.save_pretrained("./path")`.
        '''
        pass


# Model (DO NOT change the name "SentimentClassifier")
class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig # Which config class to use

    def __init__(self, config: Optional[SentimentConfig] = None):
        super().__init__(config)
        """
        Initialize the model components using the configuration.

        HINTS: 
        - Use AutoModel.from_pretrained(config.model_name)
        - Get hidden size from encoder config
        - Conduct layer normalization
        - Define classifier head
        - Use dropout layer for regularization
        - Apply loss function
        - Store any other hyperparameters from config

        Example:
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.head_type = config.head
        self.head = CustomBlock(self.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        ...
        """
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.head_type = config.head
        self.head = CustomBlock(self.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        pass

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Defines how the input data flows through the model.

        Args:
            input_ids: tokenized input sequences
            attention_mask: masks for padding tokens
            labels: ground-truth labels (optional, for training)

        Returns:
            Dictionary with "logits" (and optionally loss)
        
        HINTS:
        - Pass inputs through the encoder
        - Apply dropout and classifier head
        - Compute loss if labels are provided
        - Return logits (and loss if computed)

        Example:
        outputs = self.encoder(...)
        feat = outputs.last_hidden_state
        feat = self.dropout(self.norm(feat))
        logits = self.head(feat)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        feat = outputs.last_hidden_state[:, 0, :]
        feat = self.dropout(self.norm(feat))

        logits = self.head(feat)
        result = {"logits": logits}
        # print("feat:", feat.shape)
        # print("logits:", logits.shape)
        # print("labels:", labels.shape)

        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels)
        return result


# Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model accuracy on a given dataset.

    Args:
        model: the trained PyTorch model
        dataloader: DataLoader for validation or test set

    Returns:
        acc: overall accuracy
        all_y: true labels
        all_pred: predicted labels
    """
    model.eval()  
    all_y, all_pred = [], []
    with torch.inference_mode():  
        for batch in dataloader:
            '''
            HINTS:
            - Move the batch to the correct device (GPU/CPU)
            - Run a forward pass through the model
            - Get predicted class from logits
            - Save ground-truth and predicted labels
            '''
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            all_y.extend(batch["labels"].cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(all_y, all_pred)
    return acc, np.array(all_y), np.array(all_pred)


# Training Loop
def train(
    model_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    max_length: int,
                     # any other hyperparameters you want to add (e.g., learning rate, dropout, etc.)
    head: str,
    dropout: float,
    lr_encoder: float,
    lr_head: float,
    warmup_ratio: float,
    seed: int = 42,
):
    '''
    HINTS:
    - Setup & Reproductibility
    - Prepare datasets and dataloaders
    - Initialize the model
    - Set up optimizer and learning rate scheduler
    - Run the training loop (and save the best checkpoint)
    - Evaluation and save results and metrics
    '''

    # 1. Setup & Reproducibility
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Prepare datasets and dataloaders (train, val, test)
    '''
    Example:
    tokenizer = AutoTokenizer.from_pretrained(...)
    ds = SentimentDataset(...)
    dl = DataLoader(...)
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ds_train = SentimentDataset(train_csv, tokenizer, max_length)
    ds_val   = SentimentDataset(val_csv,   tokenizer, max_length)
    ds_test  = SentimentDataset(test_csv,  tokenizer, max_length)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print("DataLoader created successfully.")
    
    # 3. Initialize the model
    '''
    Example:
    config = SentimentConfig(...)
    model = SentimentClassifier(...).to(DEVICE)
    '''
    config = SentimentConfig()
    model = SentimentClassifier(config).to(DEVICE)

    # 4. Set up optimizer and learning rate scheduler
    '''
    Example:
    optimizer = optim.AdamW(...)
    scheduler = get_linear_schedule_with_warmup(...)
    '''
    optimizer = optim.AdamW(
        model.parameters(),   # parameters to update
        lr=lr_encoder,              # learning rate
        betas=(0.9, 0.999),   # coefficients for moving averages
        eps=1e-8,             # small constant to avoid division by zero
        weight_decay=0.01     # decoupled L2 regularization strength
    )

    num_batches_per_epoch = len(dl_train)
    total_steps = epochs * num_batches_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_ratio*total_steps, 
        num_training_steps=total_steps
    )

    # 5. Run the training loop
    best_val = -1.0
    ckpt_dir = os.path.join(out_dir, "checkpoint") # DO NOT change the file name
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)

    for epoch in range(1, epochs + 1):
        model.train()  
        running_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            '''
            HINTS:
            - Move data to GPU/CPU (the same when doing evaluation)
              -> batch = ...

            - Reset gradients 
              -> optimizer.zero_grad(...)

            - Forward pass
              -> outputs = model(...)
              -> loss = outputs[...]
            
            - Backpropagation
              -> loss.backward()
              -> torch.nn.utils.clip_grad_norm_(...)
            
            - Optimizer step and scheduler update
              -> optimizer.step()
              -> scheduler.step()

            - Update running loss
              -> running_loss += loss.item()
            '''

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
    print("train complete")
    val_acc   = eval("val", dl_val)
    test_acc  = eval("test", dl_test)

    # Save Summary in json format
    summary = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "params_trainable": int(sum(p.numel() for p in best.parameters() if p.requires_grad))
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Cleanup
    try:
        best.to("cpu"); model.to("cpu")
    except Exception:
        pass
    del best, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Main

def main():
    parser = argparse.ArgumentParser()
    # file paths
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--out_dir", type=str, default="./saved_models/") # DO NOT change the file name

    # model / data
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)

    # architecture
    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.2)

    # optimization
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Setup
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    '''
    HINTS:
    - Load the dataset and split into training / validation 
    - Save split data
    - Start Training

    Example:
    # train/val split
    full = pd.read_csv(...)

    train_df, val_df = train_test_split(...)
    os.makedirs(arg.out_dir, exist_ok=True)
    train_split = os.path.join(...)
    val_split = os.path.join(...)
    train_df.to_csv(...)
    val_df.to_csv(...)

    # Start training
    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,

                                     # any other hyperparameters you want to add (e.g., learning rate, dropout, etc.)
        seed=args.seed,
    )
    '''
    # train/val split
    full = pd.read_csv("./dataset/dataset.csv")

    train_df, temp_df = train_test_split(full, test_size=0.2, random_state=args.seed, stratify=full["label"])

    # --- Second split: 10% val, 10% test ---
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["label"])
    os.makedirs(args.out_dir, exist_ok=True)

    train_split = os.path.join(args.out_dir, "train_split.csv")
    val_split = os.path.join(args.out_dir, "val_split.csv")
    test_df.to_csv(args.test_csv, index=False)

    train_df.to_csv(train_split, index=False)
    val_df.to_csv(val_split, index=False)
    print("split complete")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val  : {len(val_df)} samples")
    print(f"  Test : {len(test_df)} samples")

    # Start training
    train(
        model_name=args.model_name,
        train_csv=train_split,
        val_csv=val_split,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
                                     # any other hyperparameters you want to add (e.g., learning rate, dropout, etc.)
        head=args.head,
        dropout=args.dropout,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        warmup_ratio=args.warmup_ratio,

        seed=args.seed,
    )

if __name__ == "__main__":
    main()

