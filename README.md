# Text De-toxification

## Training the model
To train the model, simply run:
> `sh train.sh`<br>

It will download the necessary data, preprocess it, and train the model.
## Running the model
To run the model, simply run:
> `sh predict.sh <toxic text>`<br>

It will de-toxify the text that you provided in the arguments.
Example:
> `sh predict.sh "Most toxic text ever"`<br>

Output:
> `Least toxic text ever`<br>