from datasets import load_dataset
import numpy as np
from datasets import load_metric
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import sentencepiece

#Sample dataaset kde4
#raw_datasets = load_dataset("kde4", lang1="en", lang2 = "fr")

# Tokenize input
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


#Preprocess dasta
def Preprocess_function(input, max_length = 128):
    inputs = [ex["en"] for ex in input["translation"]]
    targets = [ex["vi"] for ex in input["translation"]]

    model_inputs = tokenizer(
        inputs, text_target=targets, max_length = max_length, truncation = True
    )
    return model_inputs

#Load dataset
def LoadData(input_data, config_name):
    data = load_dataset(input_data,config_name)
    return data


# Evaluate model with the metric BLEU
def Compute_metric(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds,tuple):
        preds = preds[0]

    decoded_preds = tokenizer(preds, return_tensor = "pt").batch_decode(preds, skip_special_tokens= True)

    # Replace -100s as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer(preds, return_tensor = "pt").pad_token_id )
    decoded_labels = tokenizer(labels, return_tensors="pt").batch_decode(labels, skip_special_tokens=True)
    # Post processing
    decoded_preds = [pred.strip() for pred in decoded_preds ]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu" :  result["score"]}

