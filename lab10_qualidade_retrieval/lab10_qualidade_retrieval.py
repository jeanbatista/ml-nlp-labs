import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================
# 1. Base de documentos (chunks) financeiros
# ============================================================

documents = [
    {
        "id": "cat_transporte",
        "tipo": "categoria",
        "texto": (
            "Categoria Transporte: inclui combustível, ônibus, metrô, "
            "táxi e aplicativos de corrida como Uber."
        ),
    },
    {
        "id": "cat_alimentacao",
        "tipo": "categoria",
        "texto": (
            "Categoria Alimentação: inclui restaurantes, supermercados, "
            "lanches e aplicativos de entrega de comida."
        ),
    },
    {
        "id": "cat_moradia",
        "tipo": "categoria",
        "texto": (
            "Categoria Moradia: inclui aluguel, condomínio, energia elétrica, "
            "água e manutenção residencial."
        ),
    },
    {
        "id": "politica_valor",
        "tipo": "politica",
        "texto": "Política: despesas devem possuir valor maior que zero.",
    },
    {
        "id": "politica_comprovante",
        "tipo": "politica",
        "texto": (
            "Política: despesas reembolsáveis devem possuir comprovante "
            "quando exigido."
        ),
    },
]

# ============================================================
# 2. Casos de teste (query -> id do documento esperado)
# ============================================================

test_cases = [
    {
        "query": "gastei com Uber ontem",
        "expected_id": "cat_transporte",
    },
    {
        "query": "paguei o aluguel",
        "expected_id": "cat_moradia",
    },
    {
        "query": "comprei comida no supermercado",
        "expected_id": "cat_alimentacao",
    },
]

# ============================================================
# 3. Geração dos embeddings dos documentos
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = [doc["texto"] for doc in documents]
document_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# ============================================================
# 4. Retrieval com controle de top-k e threshold
# ============================================================

def retrieve(query, top_k=3, threshold=None, tipo=None):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    scores = document_embeddings @ query_embedding
    ranking = np.argsort(scores)[::-1]

    results = []
    for idx in ranking:
        doc = documents[idx]
        score = float(scores[idx])
        if tipo is not None and doc["tipo"] != tipo:
            continue
        if threshold is not None and score < threshold:
            continue
        results.append({
            "id": doc["id"],
            "tipo": doc["tipo"],
            "texto": doc["texto"],
            "score": score
        })
        if len(results) == top_k:
            break
    return results


def evaluate_reranker(test_cases):
    correct_before = 0
    correct_after = 0

    for case in test_cases:
        query = case["query"]
        expected_id = case["expected_id"]

        candidates = retrieve(
            query,
            top_k=3,
            threshold=None,
            tipo="categoria"
        )
        before_top1 = candidates[0]["id"]

        pairs = [
            [query, candidate["texto"]]
            for candidate in candidates
        ]
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        after_top1 = reranked[0][0]["id"]

        if before_top1 == expected_id:
            correct_before += 1
        if after_top1 == expected_id:
            correct_after += 1

        print(f"\nQuery: {query}")
        print(f"Esperado: {expected_id}")
        print(f"Top-1 antes do reranking: {before_top1}")
        print(f"Top-1 depois do reranking: {after_top1}")

    print(f"\nAcertos antes do reranking: {correct_before}/{len(test_cases)}")
    print(f"Acertos depois do reranking: {correct_after}/{len(test_cases)}")


def main():
    print(document_embeddings.shape)

    results = retrieve(
        "gastei com Uber ontem",
        top_k=3
    )
    for result in results:
        print(
            result["id"],
            "-",
            round(result["score"], 4)
        )

    print()

    results = retrieve(
        "gastei com Uber ontem",
        top_k=3,
        threshold=0.10
    )
    for result in results:
        print(
            result["id"],
            "-",
            round(result["score"], 4)
        )

    for case in test_cases:
        results = retrieve(
            case["query"],
            top_k=3,
            threshold=0.10
        )
        print("\nQuery:", case["query"])
        print("Esperado:", case["expected_id"])
        for result in results:
            print(
                result["id"],
                "-",
                round(result["score"], 4)
            )

    thresholds = [0.10, 0.20, 0.30]
    for threshold in thresholds:
        print(f"\nTHRESHOLD = {threshold}")
        for case in test_cases:
            results = retrieve(
                case["query"],
                top_k=3,
                threshold=threshold
            )
            print("\nQuery:", case["query"])
            for result in results:
                print(
                    result["id"],
                    "-",
                    round(result["score"], 4)
                )

    for case in test_cases:
        results = retrieve(
            case["query"],
            top_k=3,
            threshold=0.10,
            tipo="categoria"
        )
        print("\nQuery:", case["query"])
        for result in results:
            print(
                result["id"],
                "-",
                round(result["score"], 4)
            )

    query = "qual é a cotação do dólar hoje?"
    results = retrieve(
        query,
        top_k=3,
        threshold=0.30,
        tipo="categoria"
    )
    print("Query:", query)
    if not results:
        print("FALLBACK: nenhum contexto confiável encontrado.")
    else:
        for result in results:
            print(
                result["id"],
                "-",
                round(result["score"], 4)
            )

    query = "paguei o aluguel"
    candidates = retrieve(
        query,
        top_k=3,
        threshold=None,
        tipo="categoria"
    )
    pairs = [
        [query, candidate["texto"]]
        for candidate in candidates
    ]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("ANTES DO RERANKING")
    for candidate in candidates:
        print(
            candidate["id"],
            "-",
            round(candidate["score"], 4)
        )

    print("\nDEPOIS DO RERANKING")
    for candidate, rerank_score in reranked:
        print(
            candidate["id"],
            "-",
            round(float(rerank_score), 4)
        )

    evaluate_reranker(test_cases)


if __name__ == "__main__":
    main()
