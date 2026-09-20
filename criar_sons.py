import wave
import math
import struct
import os


# =========================
# CONFIGURAÇÕES
# =========================

pasta = "sons"

os.makedirs(pasta, exist_ok=True)

notas = {
    "do": 262,
    "re": 294,
    "mi": 330,
    "fa": 349,
    "sol": 392
}

duracao = 0.8
taxa_amostragem = 44100


# =========================
# CRIAR OS SONS
# =========================

for nome, frequencia in notas.items():

    caminho = os.path.join(pasta, nome + ".wav")

    quantidade_amostras = int(taxa_amostragem * duracao)

    dados = []

    for i in range(quantidade_amostras):

        tempo = i / taxa_amostragem

        onda = math.sin(
            2 * math.pi * frequencia * tempo
        )

        # Pequeno fade para evitar estalos
        if tempo < 0.03:
            onda *= tempo / 0.03

        if tempo > duracao - 0.05:
            onda *= (duracao - tempo) / 0.05

        valor = int(onda * 16000)

        dados.append(
            struct.pack("<h", valor)
        )


    with wave.open(caminho, "wb") as arquivo:

        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa_amostragem)

        arquivo.writeframes(
            b"".join(dados)
        )


    print("Criado:", caminho)


print()
print("Todos os sons foram criados!")