from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import warnings

warnings.filterwarnings("ignore")


def detoxify(model, inference_request, tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0))


def main():
    model_checkpoint = "t5-small"
    prefix = "Make this text less toxic:"
    # Check if an argument is provided
    if len(sys.argv) <= 1:
        print("No text was provided!")
        raise ValueError("No text was provided to detoxify")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained("models/t5-best-model")
    model.eval()
    model.config.use_cache = False

    inference_request = prefix + sys.argv[1]
    detoxify(model, inference_request, tokenizer)


if __name__ == "__main__":
    main()
