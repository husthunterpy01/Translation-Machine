# TODO 3
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import data
if __name__ == "__main__":
    parser = ArgumentParser()
    
    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--per-device-train-batch-size", default=32, type=int)
    parser.add_argument("--per-device-eval-batch-size", default=64, type=int)
    parser.add_argument("--num-train-epochs", default=3, type=int)
    parser.add_argument("--learning-rate", default = 2e-5, type = float)
    parser.add_argument("--weight-decay", default = 0.01, type = float)
    parser.add_argument("--save-total-limit", default = 3, type = int)
    parser.add_argument("--predict-with-generate", default = False, type = bool)
    home_dir = os.getcwd()
    args = parser.parse_args()

    #num_encoder_layers = args.num_encoder_layers
    #d_model = args.d_model
    data_source = "mt_eng_vietnamese"
    config_name = 'iwslt2015-en-vi'
    # FIXME
    # Project Description

    print('---------------------Welcome to ProtonX Translation Machine project-------------------')
    print('Github: husthunterpy01')
    print('Email: dangminhhust193231@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Translation Machine model with hyper-params:')
    print('===========================')


    # Process data
    dataset = data.LoadData(data_source,config_name)
    split_datasets = dataset["train"].train_test_split(train_size=0.8, seed = 20)
    split_datasets["validation"] = split_datasets.pop("test")
    tokenized_datasets = split_datasets.map(
        data.Preprocess_function,
        batched = True,
        remove_columns = split_datasets["train"].column_names
    )
    # Instantiate the model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path= 'Helsinki-NLP/opus-mt-en-fr'
    )

    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size= args.per_device_eval_batch_size,
        weight_decay = args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate= True,
        fp16=False, #Set to True when you train with GPU
        logging_dir="./VietnameseTranslationMachine/logs",
        logging_steps= 1000,
        load_best_model_at_end=True,
        output_dir= './VietnameseTranslationMachine/Result'
    )

    #Trainer setup
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset =tokenized_datasets["train"],
        eval_dataset= tokenized_datasets["validation"],
        data_collator= DataCollatorForSeq2Seq(data.tokenizer, model = model),
        tokenizer = data.tokenizer,
        compute_metrics = data.Compute_metric
    )
# Train the model
    for epoch in range(training_args.num_train_epochs):
        trainer.train()
        eval_result = trainer.evaluate()
        print(f"Epoch {epoch + 1} - Evaluation result: {eval_result}")

        # Save metrics, logs, and model checkpoint
        trainer.save_metrics(f"./VietnameseTranslationMachine/Result/training_metrics_epoch_{epoch + 1}.json")
        trainer.log_metrics("epoch", eval_result)
        trainer.save_model(f"./VietnameseTranslationMachine/Result/checkpoint_epoch_{epoch + 1}")
        trainer.save_state()

        if eval_result["eval_bleu"] > best_bleu:
            best_bleu = eval_result["eval_bleu"]
            print(f"New best BLEU score: {best_bleu}")

    # Evaluate the model
    val_result = trainer.evaluate(tokenized_datasets["validation"])
    print(f"Test set evaluation result: {val_result}")