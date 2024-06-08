from datasets import load_dataset
from transformers import set_seed, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, MultitaskPromptTuningConfig, TaskType, MultitaskPromptTuningInit
import json
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim.adamw import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--total_epochs", type=int, default=5)
args = parser.parse_args()

model_name = args.model_name


num_tasks = int(os.getenv('NUM_TASKS', '2')) 
prompt_tuning_init_text = os.getenv('PROMPT_TUNING_INIT_TEXT', 'edit text from source to target:') 

peft_config = MultitaskPromptTuningConfig(
    tokenizer_name_or_path=model_name,
    num_tasks=num_tasks,
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=MultitaskPromptTuningInit.TEXT,
    num_virtual_tokens=100,
    num_transformer_submodules=1,
    prompt_tuning_init_text=prompt_tuning_init_text,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = get_peft_model(model, peft_config)

model = model.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def send_to_device(batch):
    for i in batch:
        batch[i] = batch[i].cuda()
    return batch

def get_task1(split):
    return load_task_data("task1", split)

def get_task2(split):
    return load_task_data("task2", split)

def get_task3(split):
    return load_task_data("task3", split)

def get_task4(split):
    return load_task_data("task4", split)

def load_task_data(task_name, split):
    file_path = f"/path/to/your/multitask_dataset/{task_name}_{split}.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result_examples = []
    for example in data:
        input_text = example["text"].strip() + tokenizer.eos_token
        output_text = example["summary"].strip() + tokenizer.eos_token
        result_examples.append({
            "input": input_text,
            "output": output_text,
            "task_id": int(task_name[-1]) - 1
        })
    return result_examples

class MyDataset(Dataset):
    def __init__(self, split: str, mode: str = "source") -> None:
        super().__init__()

        if split == "train":
            if mode == "source":
                self.examples = get_task1(split) + get_task2(split) + get_task3(split) + get_task4(split)
            elif mode == "target":
                self.examples = get_task1(split)
        if split == "val":
            self.examples = get_task1("validation")
        if split == "test":
            self.examples = get_task1("test")

    def __getitem__(self, index) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

def collate_fn(batch):
    input = [i["input"] for i in batch]
    output = [i["output"] for i in batch]

    input_dict = tokenizer(input, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_dict = tokenizer(output, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512)

    output_dict.input_ids[output_dict.input_ids == tokenizer.pad_token_id] = -100

    task_ids = torch.tensor([i["task_id"] for i in batch])

    return {
        "input_ids": input_dict.input_ids,
        "attention_mask": input_dict.attention_mask,
        "labels": output_dict.input_ids,
        "task_ids": task_ids,
    }

train = DataLoader(MyDataset("train"), shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
val = DataLoader(MyDataset("val"), shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)


model_save_path = "/path/to/your/prompt_source/checkpoints_source"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

metrics = {
    "train_loss": [],
    "eval_loss": [],
    "learning_rate": [],
    "bleu": [],
    "meteor": [],
    "rouge": []
}

results_file = f"{model_save_path}/training_evaluation_metrics.json"
total_epochs = 5
n = 1000
step = 0

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=len(train) * total_epochs)

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

import numpy as np

def evaluate(model, data_loader, tokenizer, device="cuda"):
    model.eval()
    predictions = []
    references = []
    total_eval_loss = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        task_ids = batch['task_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task_ids=task_ids)
            loss = outputs.loss
            total_eval_loss += loss.item()
            generated_tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, task_ids=task_ids)

        decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_tokens]

        labels = labels.cpu().numpy()
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        references.extend([[label] for label in decoded_labels])
        predictions.extend(decoded_preds)

    avg_eval_loss = total_eval_loss / len(data_loader)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return {
        "eval_loss": avg_eval_loss,
        "bleu": bleu_score["bleu"],
        "meteor": meteor_score["meteor"],
        "rouge": rouge_score
    }

# Main training loop
for epoch in range(total_epochs):
    model.train()
    total_train_loss = 0
    train_loader = DataLoader(MyDataset("train"), batch_size=8, collate_fn=collate_fn, shuffle=True)
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = send_to_device(batch)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        total_train_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)
    metrics["train_loss"].append(avg_train_loss)

    eval_loader = DataLoader(MyDataset("val"), batch_size=8, collate_fn=collate_fn, shuffle=False)
    eval_metrics = evaluate(model, eval_loader, tokenizer)
    metrics["eval_loss"].append(eval_metrics["eval_loss"])
    metrics["bleu"].append(eval_metrics["bleu"])
    metrics["meteor"].append(eval_metrics["meteor"])
    metrics["rouge"].append(eval_metrics["rouge"])

    # Save model after each epoch
    epoch_model_path = os.path.join(model_save_path, f"checkpoint_epoch_{epoch+1}")
    model.save_pretrained(epoch_model_path)

    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)

print("Training completed and model saved.")
