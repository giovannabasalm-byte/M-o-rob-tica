import cv2
import mediapipe as mp
import pygame
import time

# Inicializar o mixer do Pygame
pygame.mixer.init()

# Sons carregados com Pygame
sons = {
    "DÓ": pygame.mixer.Sound("pianinho/Sons/do.wav"),
    "RÉ": pygame.mixer.Sound("pianinho/Sons/re.wav"),
    "MI": pygame.mixer.Sound("pianinho/Sons/mi.wav"),
    "FÁ": pygame.mixer.Sound("pianinho/Sons/fa.wav"),
    "SOL": pygame.mixer.Sound("pianinho/Sons/sol.wav")
}

nota_do_dedo = {4: "DÓ", 8: "RÉ", 12: "MI", 16: "FÁ", 20: "SOL"}

# MediaPipe Hand Landmarker configurado para 2 mãos
MODEL_PATH = "models/hand_landmarker.task"
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2  # Deteta até 2 mãos
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Câmera
camera = cv2.VideoCapture(0)
frame_timestamp = 0

pontas_dedos = [4, 8, 12, 16, 20]
efeito_dedo = {ponta: 0 for ponta in pontas_dedos}
tecla_pressionada = {ponta: False for ponta in pontas_dedos}

# Altura padrão inicial para a linha amarela
altura_linha_global = 300 

# Loop principal
while True:
    sucesso, frame = camera.read()
    if not sucesso:
        print("Não foi possível aceder à câmara.")
        break

    frame = cv2.flip(frame, 1)
    altura, largura, _ = frame.shape

    # =========================
    # PREPARAR IMAGEM E DETECTAR AS MÃOS
    # =========================
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    novo_timestamp = int(time.monotonic() * 1000)

    if novo_timestamp <= frame_timestamp:
        novo_timestamp = frame_timestamp + 1
    frame_timestamp = novo_timestamp
    resultado = landmarker.detect_for_video(imagem_mp, frame_timestamp)

    dedos_detectados = []
    maos_detectadas = []
    
    mao_esquerda_vista = False
    alvo_linha_y = None

    dedos_ativos_neste_frame = {ponta: False for ponta in pontas_dedos}

    if resultado.hand_landmarks and resultado.handedness:
        for hand_landmarks, hand_classification in zip(resultado.hand_landmarks, resultado.handedness):
            maos_detectadas.append(hand_landmarks)
            
            # Como a imagem está espelhada (cv2.flip), o rótulo do MediaPipe fica invertido.
            # Portanto, o "Right" do MediaPipe corresponde à sua mão ESQUERDA na tela espelhada,
            # e o "Left" corresponde à sua mão DIREITA.
            rotulo_mao = hand_classification[0].category_name  # "Left" ou "Right"

            if rotulo_mao == "Right":
                # Mão Esquerda real (controla a linha amarela)
                mao_esquerda_vista = True
                alvo_linha_y = int(hand_landmarks[8].y * altura)
            
            elif rotulo_mao == "Left":
                # Mão Direita real (toca as notas)
                for ponta in pontas_dedos:
                    ponto = hand_landmarks[ponta]
                    dedo_x = int(ponto.x * largura)
                    dedo_y = int(ponto.y * altura)
                    dedos_detectados.append((ponta, dedo_x, dedo_y))
                    dedos_ativos_neste_frame[ponta] = True

    # Atualiza a altura da linha se a mão esquerda estiver visível
    if mao_esquerda_vista and alvo_linha_y is not None:
        altura_linha_global = alvo_linha_y

    # Processamento de clique para a mão direita com base na linha amarela atual
    for ponta, dedo_x, dedo_y in dedos_detectados:
        if dedo_y > altura_linha_global:
            if not tecla_pressionada[ponta]:
                nota = nota_do_dedo[ponta]
                sons[nota].play()
                tecla_pressionada[ponta] = True
                efeito_dedo[ponta] = 12
        else:
            tecla_pressionada[ponta] = False

    for ponta in pontas_dedos:
        if not dedos_ativos_neste_frame[ponta]:
            tecla_pressionada[ponta] = False

    # =========================
    # DESENHAR ELEMENTOS VISUAIS
    # =========================
    for hand_landmarks in maos_detectadas:
        for conexao in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
            ponto_inicio = hand_landmarks[conexao.start]
            ponto_fim = hand_landmarks[conexao.end]
            cv2.line(frame,
                     (int(ponto_inicio.x * largura), int(ponto_inicio.y * altura)),
                     (int(ponto_fim.x * largura), int(ponto_fim.y * altura)),
                     (255, 20, 147), 2)
            
        for landmark in hand_landmarks:
            cv2.circle(frame, (int(landmark.x * largura), int(landmark.y * altura)),
                       4, (200, 200, 200), -1)

    # Desenhar a linha amarela controlada pela mão esquerda
    cv2.line(frame, (50, altura_linha_global), (largura - 50, altura_linha_global), (0, 255, 255), 3)
    cv2.putText(frame, "LINHA DE TOQUE (MÃO ESQUERDA)", (60, altura_linha_global - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Desenhar as caixinhas nos dedos da mão direita
    for ponta, dedo_x, dedo_y in dedos_detectados:
        nota = nota_do_dedo[ponta]
        
        tamanho_caixa = 35
        x1 = dedo_x - tamanho_caixa // 2
        y1 = dedo_y - tamanho_caixa // 2
        x2 = dedo_x + tamanho_caixa // 2
        y2 = dedo_y + tamanho_caixa // 2

        cor_caixa = (0, 255, 0) if tecla_pressionada[ponta] or efeito_dedo[ponta] > 0 else (255, 20, 147)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa, 2)
        cv2.putText(frame, f"{nota}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_caixa, 2, cv2.LINE_AA)

    for ponta in efeito_dedo:
        if efeito_dedo[ponta] > 0:
            efeito_dedo[ponta] -= 1

    cv2.imshow("Pianinho Dual-Hand (Esquerda ajusta, Direita toca)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# ENCERRAR
# =========================
camera.release()
landmarker.close()
cv2.destroyAllWindows()