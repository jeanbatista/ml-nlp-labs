import json
import os
import sys

from groq import Groq
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 1. Modelo da saída estruturada (schema + validação)
# ============================================================

class Despesa(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: str = Field(description="Tipo do lançamento: 'despesa' ou 'receita'")
    valor: float = Field(description="Valor numérico em reais, sem o símbolo da moeda")
    categoria: str = Field(
        description="Categoria da despesa, ex: Transporte, Alimentação, Hospedagem, Software, Outros"
    )


# ============================================================
# 2. Configuração da API da Groq
# ============================================================

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "Você extrai dados estruturados de relatos de despesas e receitas em português. "
    "Classifique 'tipo' como 'despesa' ou 'receita', extraia o valor numérico em reais "
    "e escolha a categoria mais adequada ao relato."
)


# ============================================================
# 3. Pipeline: frase -> API da LLM -> JSON estruturado -> validação
# ============================================================

def extrair_despesa(client: Groq, frase: str) -> Despesa:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": frase},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "despesa",
                "strict": True,
                "schema": Despesa.model_json_schema(),
            },
        },
    )

    conteudo = response.choices[0].message.content

    # Validação explícita da resposta do modelo contra o schema do Pydantic
    return Despesa.model_validate(json.loads(conteudo))


# ============================================================
# 4. Execução
# ============================================================

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Defina a variável de ambiente GROQ_API_KEY antes de executar este script.")
        print("Exemplo: export GROQ_API_KEY='sua-chave-aqui'")
        sys.exit(1)

    frase = sys.argv[1] if len(sys.argv) > 1 else "Gastei 85 reais de Uber hoje"

    print("=" * 60)
    print("ENTRADA")
    print("=" * 60)
    print(frase)

    client = Groq()
    despesa = extrair_despesa(client, frase)

    print()
    print("=" * 60)
    print("JSON ESTRUTURADO E VALIDADO")
    print("=" * 60)
    print(json.dumps(despesa.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
