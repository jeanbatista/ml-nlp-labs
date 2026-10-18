from enum import Enum


class Estado(Enum):
    INICIO = "inicio"
    EXTRAINDO_DADOS = "extraindo_dados"
    AGUARDANDO_DADOS = "aguardando_dados"
    VALIDANDO_CATEGORIA = "validando_categoria"
    AGUARDANDO_CONFIRMACAO = "aguardando_confirmacao"
    REGISTRANDO_DESPESA = "registrando_despesa"
    CONCLUIDO = "concluido"
    CANCELADO = "cancelado"
    ERRO = "erro"


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Despesa:
    valor: Optional[float] = None
    categoria: Optional[str] = None
    data: Optional[str] = None


@dataclass
class Sessao:
    estado: Estado = Estado.INICIO
    despesa: Despesa = field(
        default_factory=Despesa
    )
    erro: Optional[str] = None
    confirmado: bool = False


TRANSICOES_PERMITIDAS = {
    Estado.INICIO: {
        Estado.EXTRAINDO_DADOS
    },
    Estado.EXTRAINDO_DADOS: {
        Estado.AGUARDANDO_DADOS,
        Estado.VALIDANDO_CATEGORIA,
        Estado.ERRO
    },
    Estado.AGUARDANDO_DADOS: {
        Estado.EXTRAINDO_DADOS,
        Estado.CANCELADO
    },
    Estado.VALIDANDO_CATEGORIA: {
        Estado.AGUARDANDO_CONFIRMACAO,
        Estado.AGUARDANDO_DADOS,
        Estado.ERRO
    },
    Estado.AGUARDANDO_CONFIRMACAO: {
        Estado.REGISTRANDO_DESPESA,
        Estado.CANCELADO,
        Estado.EXTRAINDO_DADOS
    },
    Estado.REGISTRANDO_DESPESA: {
        Estado.CONCLUIDO,
        Estado.ERRO
    },
    Estado.ERRO: {
        Estado.EXTRAINDO_DADOS,
        Estado.CANCELADO
    },
    Estado.CONCLUIDO: set(),
    Estado.CANCELADO: set()
}


def transicionar(sessao: Sessao, novo_estado: Estado):
    estado_atual = sessao.estado
    permitidos = TRANSICOES_PERMITIDAS[estado_atual]

    if novo_estado not in permitidos:
        raise ValueError(
            f"Transição inválida: "
            f"{estado_atual.value} -> {novo_estado.value}"
        )

    if novo_estado == Estado.REGISTRANDO_DESPESA:
        despesa = sessao.despesa
        dados_validos = (
            despesa.valor is not None
            and isinstance(despesa.valor, (int, float))
            and not isinstance(despesa.valor, bool)
            and 0 < despesa.valor < float("inf")
            and isinstance(despesa.categoria, str)
            and bool(despesa.categoria.strip())
            and isinstance(despesa.data, str)
            and bool(despesa.data.strip())
        )

        if not dados_validos:
            raise ValueError(
                "Registro bloqueado: dados incompletos ou inválidos"
            )

        if sessao.confirmado is not True:
            raise ValueError(
                "Registro bloqueado: confirmação obrigatória"
            )

    sessao.estado = novo_estado
    print(
        f"Transição realizada: "
        f"{estado_atual.value} -> {novo_estado.value}"
    )


def cancelar_despesa(sessao: Sessao):
    transicionar(sessao, Estado.CANCELADO)
    sessao.confirmado = False
    print("Despesa cancelada.")


def registrar_erro(sessao: Sessao, mensagem: str):
    transicionar(sessao, Estado.ERRO)
    sessao.erro = mensagem
    print("Erro registrado:", sessao.erro)


def atualizar_despesa(
    sessao: Sessao,
    valor=None,
    categoria=None,
    data=None
):
    if sessao.estado not in {
        Estado.EXTRAINDO_DADOS,
        Estado.AGUARDANDO_DADOS
    }:
        raise ValueError(
            "Não é permitido alterar dados neste estado"
        )

    if valor is not None:
        sessao.despesa.valor = valor
    if categoria is not None:
        sessao.despesa.categoria = categoria
    if data is not None:
        sessao.despesa.data = data

    # Uma alteração invalida qualquer confirmação anterior
    sessao.confirmado = False
    print("Despesa atualizada:", sessao.despesa)


if __name__ == "__main__":
    sessao = Sessao()

    print("\n--- MENSAGEM 1 ---")
    transicionar(sessao, Estado.EXTRAINDO_DADOS)
    atualizar_despesa(
        sessao,
        valor=85.0,
        data="2026-10-17"
    )
    transicionar(sessao, Estado.AGUARDANDO_DADOS)
    print("Estado:", sessao.estado.value)

    print("\n--- MENSAGEM 2 ---")
    # Usuário informou a categoria
    transicionar(sessao, Estado.EXTRAINDO_DADOS)
    atualizar_despesa(
        sessao,
        categoria="Transporte"
    )
    transicionar(sessao, Estado.VALIDANDO_CATEGORIA)
    # Simula categoria aprovada pela validação
    transicionar(sessao, Estado.AGUARDANDO_CONFIRMACAO)

    print("Estado final:", sessao.estado.value)
    print("Despesa final:", sessao.despesa)

    print("\n--- TESTE FINAL: CONFIRMACAO INVALIDADA ---")
    sessao.confirmado = True
    transicionar(sessao, Estado.EXTRAINDO_DADOS)
    atualizar_despesa(sessao, valor=850.0)

    print("Confirmado:", sessao.confirmado)
    print("Novo valor:", sessao.despesa.valor)
    print("Estado:", sessao.estado.value)
