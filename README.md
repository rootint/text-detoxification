# Text Detoxification

This is an implementation of text detoxification using a fine-tuned [T5 model from HuggingFace.](https://huggingface.co/t5-small)<br>
Notebooks in notebooks/testing don't have extensive comments as they were used as drafts to check hypotheses.
## Training the model
To train the model, simply run:
> `sh train.sh`<br>

It will download the necessary data, preprocess it, and train the model on a CPU.
## Running the model
To run the model, simply run:
> `sh predict.sh <toxic text>`<br>

At first, it will download the model from a bucket, and then will de-toxify the text that you provided in the arguments.
Example:
> `sh predict.sh "Most toxic text ever"`<br>

Output:
> `Least toxic text ever`<br>
