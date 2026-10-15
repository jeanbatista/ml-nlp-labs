# ============================================================
# 1. Documentos (chunks) financeiros
# ============================================================

documents = [
    "Categoria Alimentação: inclui restaurantes, delivery e supermercado.",
    "Categoria Transporte: inclui combustível, ônibus, metrô e aplicativos de corrida.",
    "Categoria Saúde: inclui medicamentos, consultas médicas e exames.",
    "Política: despesas devem possuir valor maior que zero.",
    "Política: a categoria informada deve existir na lista de categorias permitidas.",
]


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
document_embeddings = model.encode(documents)

query = "gastei com Uber ontem"
query_embedding = model.encode(query)
scores = util.cos_sim(query_embedding, document_embeddings)[0]


def main():
    print(len(documents))
    for doc in documents:
        print(doc)

    print("Quantidade de embeddings:", len(document_embeddings))
    print("Dimensão de cada embedding:", len(document_embeddings[0]))
    print("Shape:", document_embeddings.shape)

    for i, score in enumerate(scores):
        print(f"{score:.4f} -> {documents[i]}")

    top_k = 2
    top_results = scores.topk(k=top_k)
    retrieved_chunks = []
    for score, idx in zip(top_results.values, top_results.indices):
        chunk = documents[idx]
        retrieved_chunks.append(chunk)
        print(f"{score:.4f} -> {chunk}")

    context = "\n".join(retrieved_chunks)
    print("\nCONTEXTO RECUPERADO:")
    print(context)


if __name__ == "__main__":
    main()
