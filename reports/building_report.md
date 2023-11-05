# Report 1: Building

# Baseline: Dictionary based
...
# Hypothesis 1: Custom embeddings
...
# Hypothesis 2: More rnn layers
...
# Hypothesis 3: Pretrained embeddings
...
# Results
...

Tried using BERT, but ran out of memory
Tried building my own architecture, but couldn't))
Used T5 and it worked. The biggest problem was making sure that the dataset was in the same format as the required one.
Tried 3 epochs - took 2 hours to train.
Also, made all the text lowercase and without special symbols, that was a mistake, will redo tomorrow.
The threshold might be experimented with - 0.6 is an arbitrary number
Also, 256 can maybe be changed to 128
