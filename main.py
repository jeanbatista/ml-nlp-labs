
from transformers import pipeline

def main():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    text = "I love this product!"

    result = classifier(text)

    print(result)


    text = "I love this product!"

    tokenizer = classifier.tokenizer

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    inputs = tokenizer(text, return_tensors="pt")

    print("Input IDs:", inputs["input_ids"])
    print("Tokens completos:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    print("Attention Mask:", inputs["attention_mask"])

if __name__ == "__main__":
    main()