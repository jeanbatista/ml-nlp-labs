# Lab 04 — Similaridade de Embeddings

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

frase = "O cliente pagou a conta do restaurante."

embedding = model.encode(frase)

print("Frase:", frase)
print("Tipo:", type(embedding))
print("Dimensões:", embedding.shape)
print("Primeiros valores:", embedding[:5])
