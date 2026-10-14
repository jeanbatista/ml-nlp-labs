from pydantic import BaseModel, Field, ValidationError

# ============================================================
# 1. Tools do agente financeiro
# ============================================================

CATEGORIES = [
    "alimentacao",
    "transporte",
    "moradia",
    "saude",
    "lazer",
]


def get_categories() -> list[str]:
    return CATEGORIES


def register_expense(
    valor: float,
    data: str,
    categoria: str,
) -> dict:
    if valor <= 0:
        raise ValueError("valor deve ser maior que zero")
    if categoria not in CATEGORIES:
        raise ValueError("categoria inválida")
    return {
        "status": "ok",
        "valor": valor,
        "data": data,
        "categoria": categoria,
    }


# ============================================================
# 2. Validação dos argumentos antes da execução da tool
# ============================================================

class RegisterExpenseArgs(BaseModel):
    valor: float = Field(gt=0)
    data: str
    categoria: str


# ============================================================
# 3. Dispatcher — executa apenas tools permitidas
# ============================================================

def execute_tool(tool_name: str, tool_args: dict):
    if tool_name == "get_categories":
        return get_categories()
    if tool_name == "register_expense":
        args = RegisterExpenseArgs(**tool_args)
        return register_expense(**args.model_dump())
    raise ValueError("tool não permitida")


# ============================================================
# 4. Execução
# ============================================================

def main():
    print(get_categories())

    print(
        register_expense(
            valor=85.0,
            data="2026-10-14",
            categoria="transporte",
        )
    )

    try:
        print(
            register_expense(
                valor=85.0,
                data="2026-10-14",
                categoria="viagem",
            )
        )
    except ValueError as exc:
        print(f"Erro: {exc}")

    args = RegisterExpenseArgs(
        valor=85,
        data="2026-10-14",
        categoria="transporte",
    )
    print(args.model_dump())
    print(register_expense(**args.model_dump()))

    try:
        args = RegisterExpenseArgs(
            valor=-10,
            data="2026-10-14",
            categoria="transporte",
        )
    except ValidationError as e:
        print(e)

    print(
        execute_tool(
            "get_categories",
            {}
        )
    )

    print(
        execute_tool(
            "register_expense",
            {
                "valor": 85,
                "data": "2026-10-14",
                "categoria": "transporte",
            }
        )
    )

    try:
        print(
            execute_tool(
                "delete_database",
                {}
            )
        )
    except ValueError as e:
        print("Erro:", e)


if __name__ == "__main__":
    main()
