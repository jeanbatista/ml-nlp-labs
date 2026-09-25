
from transformers import AutoTokenizer

modelo = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(modelo)

frase = "Gastei R$ 150 no supermercado."

tokens = tokenizer.tokenize(frase)
ids = tokenizer.encode(frase, add_special_tokens=False)

print("Frase original:", frase)
print("Tokens:", tokens)
print("Token IDs:", ids)
print("Quantidade de tokens:", len(ids))

print("\nDecodificação:")
print(tokenizer.decode(ids))
  

# Execução do modelo Qwen-2.5B-Instruct para gerar uma resposta a partir de uma mensagem do usuário.

import torch
from transformers import AutoModelForCausalLM

# 1. Carregar o modelo na GPU da Apple
device = "mps"

model = AutoModelForCausalLM.from_pretrained(
    modelo,
    dtype=torch.float16
).to(device)

model.eval()

# 2. Criar uma mensagem para o modelo
# mensagens = [ # Exemplo de mensagem para o modelo
#     {
#         "role": "user",
#         "content": "Explique em uma frase o que é uma despesa financeira."
#     }
# ]

mensagens = [ # Exemplo de mensagem para o modelo
    {
        "role": "user",
        "content": (
            "Responda em português, em apenas uma frase: "
            "o que é uma despesa em finanças pessoais?"
        )
    }
]

# 3. Preparar a entrada
inputs = tokenizer.apply_chat_template(
    mensagens,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)

# 4. Gerar a resposta
with torch.inference_mode():
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=100,
    #     do_sample=False,
    #     pad_token_id=tokenizer.eos_token_id
    # )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# 5. Decodificar apenas os tokens gerados
novos_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

resposta = tokenizer.decode(
    novos_tokens,
    skip_special_tokens=True
)

print("\n--- RESPOSTA DO QWEN ---")
print(resposta)