
# from transformers import pipeline

# def main():
#     classifier = pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english"
#     )

#     text = "I love this product!"

#     result = classifier(text)

#     print(result)


#     text = "I love this product!"

#     tokenizer = classifier.tokenizer

#     tokens = tokenizer.tokenize(text)
#     token_ids = tokenizer.convert_tokens_to_ids(tokens)

#     print("Tokens:", tokens)
#     print("Token IDs:", token_ids)

#     inputs = tokenizer(text, return_tensors="pt")

#     print("Input IDs:", inputs["input_ids"])
#     print("Tokens completos:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
#     print("Attention Mask:", inputs["attention_mask"])


#     texts = [
#         "I love this product!",
#         "Bad."
#     ]

#     inputs = tokenizer(
#         texts,
#         padding=True,
#         return_tensors="pt"
#     )

#     print("Input IDs:", inputs["input_ids"])
#     print("Attention Mask:", inputs["attention_mask"])

# if __name__ == "__main__":
#     main()


from transformers import pipeline


def main():
    # Carregar o modelo pré-treinado
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Dataset de exemplos para inferência
    texts = [
        "I love this product!",
        "This is the worst experience ever.",
        "The service was excellent.",
        "I am very disappointed.",
        "The product is good, but delivery was terrible."
    ]

    # Executar inferência em lote
    results = classifier(texts)

    # Exibir classificações
    for text, result in zip(texts, results):
        label = result["label"]
        score = result["score"]

        print(f"Texto: {text}")
        print(f"Sentimento: {label}")
        print(f"Score: {score:.2%}")
        print("-" * 50)


if __name__ == "__main__":
    main()