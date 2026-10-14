# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim

# model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# frases = [
#     "O cliente pagou a conta do restaurante.",
#     "Ele gastou dinheiro com alimentação.",
#     "A taxa de juros subiu este mês.",
# ]

# embeddings = model.encode(frases)

# similaridade_ab = cos_sim(embeddings[0], embeddings[1])
# similaridade_ac = cos_sim(embeddings[0], embeddings[2])

# print("A ↔ B:", similaridade_ab.item())
# print("A ↔ C:", similaridade_ac.item())

# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim

# model = SentenceTransformer(
#     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# frases = [
#     "O cliente pagou a conta do restaurante.",
#     "Ele gastou dinheiro com alimentação.",
#     "A taxa de juros subiu este mês.",
#     "O banco aumentou os juros do financiamento.",
# ]

# embeddings = model.encode(frases)

# matriz = cos_sim(embeddings, embeddings)

# print(matriz)

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

frases = [
    "O cliente pagou a conta do restaurante.",
    "Ele gastou dinheiro com alimentação.",
    "A taxa de juros subiu este mês.",
    "O banco aumentou os juros do financiamento.",
]

embeddings = model.encode(frases)

consulta = "Quero informações sobre juros de empréstimos."
embedding_consulta = model.encode(consulta)

similaridades = cos_sim(embedding_consulta, embeddings)[0]

melhor_indice = similaridades.argmax().item()

print("Consulta:", consulta)
print()

for i, score in enumerate(similaridades):
    print(f"{i}: {score.item():.4f} -> {frases[i]}")

print()
print("Resultado mais relevante:")
print(frases[melhor_indice])
print("Similaridade:", similaridades[melhor_indice].item())