import transformers
import pandas as pd
import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch

transformers.set_seed(42)


class ToxicBertModel:
    def __init__(self):
        # Load the model and tokenizer
        toxic_val_model_name = "unitary/toxic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(toxic_val_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            toxic_val_model_name
        )

    # Function to predict toxicity
    def predict_toxicity(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        predicted_labels = predictions.float()
        return predicted_labels


class DetoxificationModel:
    def __init__(self, model_checkpoint, prefix):
        self.model_checkpoint = model_checkpoint
        self.prefix = prefix
        self.max_input_length = 256
        self.max_target_length = 256
        self.toxicity_model = ToxicBertModel()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.metric = load_metric("sacrebleu")
        self.data = None

    def preprocess_function(self, examples):
        inputs = [self.prefix + ex for ex in examples["reference"]]
        targets = [ex for ex in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True
        )

        # Setup the tokenizer for targets
        labels = self.tokenizer(
            targets, max_length=self.max_target_length, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Simple postprocessing for text
    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # compute metrics function to pass to trainer
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels
        )

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}
        (
            toxic_sum,
            severe_toxic_sum,
            obscene_sum,
            threat_sum,
            insult_sum,
            identity_hate_sum,
        ) = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        for text in decoded_preds:
            toxic, severe_toxic, obscene, threat, insult, identity_hate = tuple(
                *self.toxicity_model.predict_toxicity(text).tolist(),
            )
            toxic_sum += toxic
            severe_toxic_sum += severe_toxic
            obscene_sum += obscene
            threat_sum += threat
            insult_sum += insult
            identity_hate_sum += identity_hate

        result["toxic_average"] = toxic_sum / len(decoded_preds)
        result["severe_toxic_average"] = severe_toxic_sum / len(decoded_preds)
        result["obscene_average"] = obscene_sum / len(decoded_preds)
        result["threat_average"] = threat_sum / len(decoded_preds)
        result["insult_average"] = insult_sum / len(decoded_preds)
        result["identity_hate_average"] = identity_hate_sum / len(decoded_preds)

        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def train(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        # Defining the parameters for training
        batch_size = 128
        model_name = self.model_checkpoint.split("/")[-1]
        args = Seq2SeqTrainingArguments(
            model_name,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=5,
            predict_with_generate=True,
            # fp16=True,
        )

        # Instead of a custom data collation function, this works fine as well!
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)

        # Instead of writing train loop we will use Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        # Saving my precious model
        trainer.save_model("best-t5-last")

    def preprocess(self, data):
        dataset = Dataset(pa.Table.from_pandas(data.reset_index(drop=True)))

        train_test_split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

        # Split the train set further into training and validation sets
        train_val_split = train_test_split_dataset["train"].train_test_split(
            test_size=0.1, seed=42
        )

        # Combine the splits into a single DatasetDict
        final_splits = DatasetDict(
            {
                "train": train_val_split["train"],
                "validation": train_val_split["test"],
                "test": train_test_split_dataset["test"],
            }
        )

        tokenized_datasets = final_splits.map(self.preprocess_function, batched=True)

        self.dataset = tokenized_datasets


def main(path="../data/training_data.csv"):
    # Loading the dataset
    data = pd.read_csv(path, index_col=False)
    model = DetoxificationModel('t5-small', "Make this text less toxic:")
    print('Preprocessing...')
    model.preprocess(data)
    print('Training...')
    model.train()


if __name__ == "__main__":
    main()
