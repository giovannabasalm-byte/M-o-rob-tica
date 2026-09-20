import cv2
import mediapipe as mp
import winsound
import time


# =========================
# SONS
# =========================

sons = {
    "DÓ": "pianinho/Sons/do.wav",
    "RÉ": "pianinho/Sons/re.wav",
    "MI": "pianinho/Sons/mi.wav",
    "FÁ": "pianinho/Sons/fa.wav",
    "SOL": "pianinho/Sons/sol.wav"
}


# =========================
# CADA DEDO TEM UMA NOTA FIXA
# =========================

nota_do_dedo = {
    4: "DÓ",     # polegar
    8: "RÉ",     # indicador
    12: "MI",    # médio
    16: "FÁ",    # anelar
    20: "SOL"    # mínimo
}


# =========================
# MEDIAPIPE
# =========================

MODEL_PATH = "models/hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


# =========================
# CÂMERA
# =========================

camera = cv2.VideoCapture(0)
frame_timestamp = 0


# =========================
# CONTROLE DOS DEDOS
# =========================

y_anterior = {
    4: None,
    8: None,
    12: None,
    16: None,
    20: None
}

dedo_pressionando = {
    4: False,
    8: False,
    12: False,
    16: False,
    20: False
}


# =========================
# EFEITO VISUAL DAS NOTAS
# =========================

efeito_tecla = {
    "DÓ": 0,
    "RÉ": 0,
    "MI": 0,
    "FÁ": 0,
    "SOL": 0
}


notas = ["DÓ", "RÉ", "MI", "FÁ", "SOL"]


# =========================
# LOOP PRINCIPAL
# =========================

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)

    altura, largura, _ = frame.shape


    # =========================
    # PREPARAR IMAGEM
    # =========================

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    # =========================
    # TIMESTAMP
    # =========================

    novo_timestamp = int(time.monotonic() * 1000)

    if novo_timestamp <= frame_timestamp:
        novo_timestamp = frame_timestamp + 1

    frame_timestamp = novo_timestamp


    resultado = landmarker.detect_for_video(
        imagem_mp,
        frame_timestamp
    )


    # =========================
    # DETECTAR PONTAS DOS DEDOS
    # =========================

    pontas_dedos = [4, 8, 12, 16, 20]

    dedos_detectados = []
    maos_detectadas = []


    if resultado.hand_landmarks:

        for hand_landmarks in resultado.hand_landmarks:

            maos_detectadas.append(hand_landmarks)

            for ponta in pontas_dedos:

                ponto = hand_landmarks[ponta]

                dedo_x = int(ponto.x * largura)
                dedo_y = int(ponto.y * altura)

                dedos_detectados.append(
                    (ponta, dedo_x, dedo_y)
                )


    # =========================
    # PIANO DA MESA
    # =========================

    mesa_largura = min(300, largura - 20)
    mesa_altura = 150

    mesa_x = (largura - mesa_largura) // 2
    mesa_y = altura - mesa_altura - 30

    largura_tecla_mesa = mesa_largura // 5


    # =========================
    # CAMADA TRANSLÚCIDA
    # =========================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (mesa_x, mesa_y),
        (mesa_x + mesa_largura, mesa_y + mesa_altura),
        (180, 180, 180),
        -1
    )


    # Desenhar as teclas na camada transparente
    for i in range(5):

        x1 = mesa_x + i * largura_tecla_mesa
        x2 = x1 + largura_tecla_mesa - 3

        y1 = mesa_y
        y2 = mesa_y + mesa_altura

        nota = notas[i]

        cor_tecla = (240, 240, 240)

        if efeito_tecla[nota] > 0:
            cor_tecla = (255, 20, 147)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            cor_tecla,
            -1
        )


    # Aplicar transparência
    cv2.addWeighted(
        overlay,
        0.35,
        frame,
        0.65,
        0,
        frame
    )


    # =========================
    # BORDA DO PIANO
    # =========================

    cv2.rectangle(
        frame,
        (mesa_x, mesa_y),
        (mesa_x + mesa_largura, mesa_y + mesa_altura),
        (30, 30, 30),
        3
    )


    # =========================
    # TOQUE
    # =========================

    for ponta, dedo_x, dedo_y in dedos_detectados:

        # O dedo precisa estar dentro da área do piano

        dentro_do_piano = (
            mesa_x <= dedo_x <= mesa_x + mesa_largura
            and mesa_y <= dedo_y <= mesa_y + mesa_altura
        )


        if dentro_do_piano:

            if y_anterior[ponta] is not None:

                # Y aumenta para baixo.
                # Portanto, valor positivo = dedo descendo.

                descida = dedo_y - y_anterior[ponta]


                if (
                    descida > 5
                    and not dedo_pressionando[ponta]
                ):

                    nota = nota_do_dedo[ponta]


                    winsound.PlaySound(
                        sons[nota],
                        winsound.SND_FILENAME | winsound.SND_ASYNC
                    )


                    # Impede repetição enquanto o dedo
                    # continuar pressionado

                    dedo_pressionando[ponta] = True


                    # Flash visual

                    efeito_tecla[nota] = 5


        else:

            # Se saiu do piano, libera o dedo

            dedo_pressionando[ponta] = False


    # =========================
    # LIBERAR NOVO TOQUE
    # =========================

    for ponta, dedo_x, dedo_y in dedos_detectados:

        if y_anterior[ponta] is not None:

            subida = y_anterior[ponta] - dedo_y


            # O dedo subiu -> pode tocar novamente

            if subida > 5:

                dedo_pressionando[ponta] = False


        y_anterior[ponta] = dedo_y


    # =========================
    # BORDAS E NOMES DAS TECLAS
    # =========================

    for i in range(5):

        x1 = mesa_x + i * largura_tecla_mesa
        x2 = x1 + largura_tecla_mesa - 3

        y1 = mesa_y
        y2 = mesa_y + mesa_altura


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (30, 30, 30),
            2
        )


        texto = notas[i]

        tamanho_texto = cv2.getTextSize(
            texto,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )[0]

        texto_x = x1 + (
            (x2 - x1 - tamanho_texto[0]) // 2
        )

        texto_y = y1 + 85


        cv2.putText(
            frame,
            texto,
            (texto_x, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA
        )


    # =========================
    # DIMINUIR FLASH DAS NOTAS
    # =========================

    for nota in efeito_tecla:

        if efeito_tecla[nota] > 0:

            efeito_tecla[nota] -= 1


    # =========================
    # DESENHAR A MÃO POR CIMA
    # =========================

    for hand_landmarks in maos_detectadas:

        for conexao in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:

            ponto_inicio = hand_landmarks[conexao.start]
            ponto_fim = hand_landmarks[conexao.end]

            x1 = int(ponto_inicio.x * largura)
            y1 = int(ponto_inicio.y * altura)

            x2 = int(ponto_fim.x * largura)
            y2 = int(ponto_fim.y * altura)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 20, 147),
                2
            )


        for landmark in hand_landmarks:

            x = int(landmark.x * largura)
            y = int(landmark.y * altura)

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 20, 147),
                -1
            )


    # =========================
    # PIANO PEQUENO - FEEDBACK
    # =========================

    piano_largura = 300
    piano_altura = 110

    margem_direita = 25
    margem_cima = 25

    piano_x = largura - piano_largura - margem_direita
    piano_y = margem_cima

    largura_tecla = piano_largura // 5


    for i in range(5):

        x1 = piano_x + i * largura_tecla
        x2 = x1 + largura_tecla - 3

        y1 = piano_y
        y2 = piano_y + piano_altura

        nota = notas[i]

        if efeito_tecla[nota] > 0:
            cor_tecla = (255, 20, 147)
        else:
            cor_tecla = (240, 240, 240)


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            cor_tecla,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (30, 30, 30),
            2
        )


        tamanho_texto = cv2.getTextSize(
            nota,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )[0]

        texto_x = x1 + (
            (x2 - x1 - tamanho_texto[0]) // 2
        )

        texto_y = y1 + 65


        cv2.putText(
            frame,
            nota,
            (texto_x, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA
        )


    # =========================
    # MOSTRAR CÂMERA
    # =========================

    cv2.imshow(
        "Pianinho Digital",
        frame
    )


    # ESC PARA SAIR

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================
# ENCERRAR
# =========================

camera.release()
landmarker.close()
cv2.destroyAllWindows()
