import json
import os
import sys
import time

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    RateLimitError,
)
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
TEMPERATURE = 0.0
TIMEOUT_SEGUNDOS = 30.0
MAX_TENTATIVAS = 3
ESPERA_ENTRE_TENTATIVAS_SEGUNDOS = 2.0

SYSTEM_PROMPT = (
    "Você extrai dados estruturados de relatos de despesas e receitas em português. "
    "Classifique 'tipo' como 'despesa' ou 'receita', extraia o valor numérico em reais "
    "e escolha a categoria mais adequada ao relato."
)


# ============================================================
# 3. Cliente da LLM (isola a integração HTTP/SDK do restante do script)
# ============================================================

class ErroLLM(Exception):
    """Erro de alto nível com mensagem compreensível para falhas na chamada à LLM."""


class LLMClient:
    """Monta a requisição, chama a API da Groq com retry limitado e devolve
    apenas o conteúdo de texto da resposta. Quem usa este cliente não precisa
    conhecer detalhes de HTTP, códigos de status ou do formato da resposta."""

    def __init__(
        self,
        model: str = MODEL,
        temperature: float = TEMPERATURE,
        timeout: float = TIMEOUT_SEGUNDOS,
        max_tentativas: int = MAX_TENTATIVAS,
    ):
        # max_retries=0: o retry é feito explicitamente abaixo, para a política ficar visível.
        self._client = Groq(timeout=timeout, max_retries=0)
        self._model = model
        self._temperature = temperature
        self._max_tentativas = max_tentativas

    def chamar(self, system_prompt: str, mensagem: str, schema: dict) -> str:
        requisicao = self._montar_requisicao(system_prompt, mensagem, schema)

        for tentativa in range(1, self._max_tentativas + 1):
            try:
                resposta = self._client.chat.completions.create(**requisicao)
                return self._extrair_conteudo(resposta)

            except AuthenticationError as exc:
                # 401: credencial inválida, repetir não resolve.
                raise ErroLLM(
                    "Falha de autenticação (401) ao chamar a LLM. "
                    "Verifique a variável de ambiente GROQ_API_KEY."
                ) from exc

            except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as exc:
                # 429, 5xx, timeout e erro de conexão são falhas transitórias: vale tentar novamente.
                if tentativa == self._max_tentativas:
                    raise ErroLLM(
                        f"Falha ao chamar a LLM após {self._max_tentativas} tentativas "
                        f"({self._descrever_falha(exc)})."
                    ) from exc
                time.sleep(ESPERA_ENTRE_TENTATIVAS_SEGUNDOS * tentativa)

            except APIStatusError as exc:
                # Demais erros 4xx (ex.: requisição inválida) não devem ser repetidos automaticamente.
                raise ErroLLM(
                    f"A API da LLM retornou um erro ({exc.status_code}): {exc.message}"
                ) from exc

        raise ErroLLM("Falha inesperada ao chamar a LLM.")

    def _montar_requisicao(self, system_prompt: str, mensagem: str, schema: dict) -> dict:
        return {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": mensagem},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "resposta_estruturada",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    @staticmethod
    def _extrair_conteudo(resposta) -> str:
        return resposta.choices[0].message.content

    @staticmethod
    def _descrever_falha(exc: Exception) -> str:
        if isinstance(exc, RateLimitError):
            return "limite de requisições excedido (429)"
        if isinstance(exc, InternalServerError):
            return "erro no servidor da API (5xx)"
        if isinstance(exc, APITimeoutError):
            return "timeout"
        return "erro de conexão"


# ============================================================
# 4. Pipeline: frase -> API da LLM -> JSON estruturado -> validação
# ============================================================

def extrair_despesa(client: LLMClient, frase: str) -> Despesa:
    conteudo = client.chamar(SYSTEM_PROMPT, frase, Despesa.model_json_schema())

    # Validação explícita da resposta do modelo contra o schema do Pydantic
    return Despesa.model_validate(json.loads(conteudo))


# ============================================================
# 5. Execução
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

    client = LLMClient()

    try:
        despesa = extrair_despesa(client, frase)
    except ErroLLM as exc:
        print()
        print(f"Erro ao consultar a LLM: {exc}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("JSON ESTRUTURADO E VALIDADO")
    print("=" * 60)
    print(json.dumps(despesa.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
