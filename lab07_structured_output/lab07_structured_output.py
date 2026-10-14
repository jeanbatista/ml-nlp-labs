import json
import os
import sys
import time
from datetime import date
from json import JSONDecodeError

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from jsonschema import ValidationError, validate

# ============================================================
# 1. Schema da saída estruturada
# ============================================================

EXPENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intencao": {
            "type": "string",
            "enum": [
                "registrar_despesa",
                "consultar_despesas"
            ]
        },
        "valor": {
            "type": ["number", "null"]
        },
        "data": {
            "type": ["string", "null"]
        },
        "categoria": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "intencao",
        "valor",
        "data",
        "categoria"
    ],
    "additionalProperties": False
}


# ============================================================
# 2. Configuração da API da Groq
# ============================================================

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.0
TIMEOUT_SEGUNDOS = 30.0
MAX_TENTATIVAS = 3
ESPERA_ENTRE_TENTATIVAS_SEGUNDOS = 2.0

DATA_REFERENCIA = date.today().isoformat()

SYSTEM_PROMPT = f"""Você extrai dados estruturados de mensagens financeiras em português.
Data de referência: {DATA_REFERENCIA}.
Regras:
- Nunca invente valores.
- Se um campo não estiver presente ou não puder ser inferido com segurança, retorne null.
- Converta datas relativas como "hoje" e "ontem" para YYYY-MM-DD usando a data de referência."""


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

def extrair_despesa(client: LLMClient, frase: str) -> dict:
    conteudo = client.chamar(SYSTEM_PROMPT, frase, EXPENSE_SCHEMA)

    try:
        dados = json.loads(conteudo)

        validate(instance=dados, schema=EXPENSE_SCHEMA)

        if dados["valor"] is None:
            return {
                "ok": False,
                "erro": "dados_incompletos",
                "campo": "valor",
            }

        return {
            "ok": True,
            "dados": dados,
        }

    except JSONDecodeError:
        return {
            "ok": False,
            "erro": "json_invalido",
            "mensagem": "A LLM não retornou um JSON válido.",
        }

    except ValidationError as exc:
        return {
            "ok": False,
            "erro": "schema_invalido",
            "mensagem": exc.message,
        }


# ============================================================
# 5. Execução
# ============================================================

FRASES_TESTE = [
    "Gastei 85 reais de Uber hoje",
    "Paguei 120 reais no supermercado ontem",
    "Paguei Uber ontem",
]


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Defina a variável de ambiente GROQ_API_KEY antes de executar este script.")
        print("Exemplo: export GROQ_API_KEY='sua-chave-aqui'")
        sys.exit(1)

    frases = [sys.argv[1]] if len(sys.argv) > 1 else FRASES_TESTE

    client = LLMClient()

    for frase in frases:
        print("=" * 60)
        print("ENTRADA")
        print("=" * 60)
        print(frase)

        try:
            resultado = extrair_despesa(client, frase)
        except ErroLLM as exc:
            print()
            print(f"Erro ao consultar a LLM: {exc}")
            print()
            continue

        print()
        print("=" * 60)
        print("RESULTADO")
        print("=" * 60)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
