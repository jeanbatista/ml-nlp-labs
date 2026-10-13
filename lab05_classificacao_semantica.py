import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Carregamento do modelo de embeddings
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 2. Definição das categorias e exemplos
# ============================================================

categorias = {
    "Alimentação": [
        "almocei em um restaurante",
        "paguei o jantar",
        "comprei um lanche",
        "tomei café com um cliente",
    ],

    "Transporte": [
        "peguei um Uber",
        "paguei uma corrida de táxi",
        "abasteci o carro",
        "paguei passagem de ônibus",
    ],

    "Hospedagem": [
        "paguei uma diária de hotel",
        "fiquei hospedado em uma pousada",
        "paguei duas noites no hotel",
        "reservei uma acomodação",
    ],

    "Software": [
        "paguei a assinatura do sistema",
        "mensalidade do software",
        "renovação da licença",
        "assinatura de ferramenta SaaS",
    ],
}


# ============================================================
# 3. Criação dos protótipos semânticos das categorias
# ============================================================

nomes_categorias = list(categorias.keys())

embeddings_categorias = []

for nome, exemplos in categorias.items():

    # Gera um embedding para cada exemplo da categoria
    embeddings_exemplos = model.encode(exemplos)

    # Calcula o vetor médio da categoria
    embedding_medio = np.mean(
        embeddings_exemplos,
        axis=0
    )

    embeddings_categorias.append(embedding_medio)


# Converte a lista de vetores médios para uma matriz NumPy
embeddings_categorias = np.array(embeddings_categorias)


print("Categorias:", nomes_categorias)
print("Shape:", embeddings_categorias.shape)


# ============================================================
# 4. Classificador
# ============================================================

def classificar_despesa(
    relato,
    threshold=0.40,
    margem_minima=0.04
):

    # Gera embedding do relato
    embedding_relato = model.encode([relato])

    # Compara o relato com os protótipos das categorias
    scores = cosine_similarity(
        embedding_relato,
        embeddings_categorias
    )[0]

    # Ranking do maior score para o menor
    ranking = sorted(
        zip(nomes_categorias, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Primeiro colocado
    melhor_categoria, melhor_score = ranking[0]

    # Segundo colocado
    _, segundo_score = ranking[1]

    # Diferença entre primeiro e segundo colocado
    margem = melhor_score - segundo_score

    # Score absoluto muito baixo
    if melhor_score < threshold:
        return "Não classificado", melhor_score, margem

    # Decisão muito ambígua
    if margem < margem_minima:
        return "Não classificado", melhor_score, margem

    return melhor_categoria, melhor_score, margem


# ============================================================
# 5. Função de diagnóstico
# ============================================================

def analisar_despesa(relato):

    embedding_relato = model.encode([relato])

    scores = cosine_similarity(
        embedding_relato,
        embeddings_categorias
    )[0]

    ranking = sorted(
        zip(nomes_categorias, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print("ANÁLISE")
    print("=" * 60)
    print(f"Relato: {relato}")
    print()

    for categoria, score in ranking:
        print(f"{categoria}: {score:.4f}")

    melhor_categoria, melhor_score = ranking[0]
    _, segundo_score = ranking[1]

    margem = melhor_score - segundo_score

    print()
    print(f"Melhor categoria: {melhor_categoria}")
    print(f"Melhor score: {melhor_score:.4f}")
    print(f"Margem: {margem:.4f}")


# ============================================================
# 6. Testes do classificador
# ============================================================

relatos = [
    "Peguei um Uber para ir ao aeroporto",
    "Almocei com um cliente",
    "Paguei a mensalidade do sistema",
    "Fiquei duas noites em um hotel",
    "Comprei uma cadeira para o escritório",
]


print("\n")
print("=" * 60)
print("CLASSIFICAÇÃO DAS DESPESAS")
print("=" * 60)


for relato in relatos:

    categoria, score, margem = classificar_despesa(relato)

    print("-" * 60)
    print("Relato:", relato)
    print("Categoria:", categoria)
    print("Score:", round(float(score), 4))
    print("Margem:", round(float(margem), 4))


# ============================================================
# 7. Diagnóstico detalhado
# ============================================================

print("\n")
print("=" * 60)
print("DIAGNÓSTICO DETALHADO")
print("=" * 60)


for relato in relatos:
    analisar_despesa(relato)