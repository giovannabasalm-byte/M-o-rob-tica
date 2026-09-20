import cv2
import mediapipe as mp
import pygame
import time
import math

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
    num_hands=2  
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Câmera
camera = cv2.VideoCapture(0)
frame_timestamp = 0

pontas_dedos = [4, 8, 12, 16, 20]
efeito_dedo = {ponta: 0 for ponta in pontas_dedos}
tecla_pressionada = {ponta: False for ponta in pontas_dedos}

# Coordenadas da Linha de Toque
linha_ponto_a = [100, 300]
linha_ponto_b = [540, 300]

# Estados de Controle e Modos
modo_ajuste_livre = False
linha_congelada = False
tempo_ultimo_clique_botao = 0
tempo_ultimo_clique_pinça = 0

centro_anterior_x = None
centro_anterior_y = None

def verifica_pinca(hand_landmarks, largura, altura):
    x1, y1 = int(hand_landmarks[4].x * largura), int(hand_landmarks[4].y * altura)
    x2, y2 = int(hand_landmarks[8].x * largura), int(hand_landmarks[8].y * altura)
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia < 35

def ponto_cruza_segmento(px, py, ax, ay, bx, by):
    ab_x = bx - ax
    ab_y = by - ay
    ap_x = px - ax
    ap_y = py - ay
    
    ab_len_sq = ab_x**2 + ab_y**2
    if ab_len_sq == 0:
        return False, 0
    
    t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
    if 0 <= t <= 1:
        proj_x = ax + t * ab_x
        proj_y = ay + t * ab_y
        distancia = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        return distancia < 18, t
    return False, 0

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
    
    mao_esquerda_dados = None
    mao_direita_dados = None

    dedos_ativos_neste_frame = {ponta: False for ponta in pontas_dedos}

    if resultado.hand_landmarks and resultado.handedness:
        for hand_landmarks, hand_classification in zip(resultado.hand_landmarks, resultado.handedness):
            maos_detectadas.append(hand_landmarks)
            rotulo_mao = hand_classification[0].category_name

            indicador_x = int(hand_landmarks[8].x * largura)
            indicador_y = int(hand_landmarks[8].y * altura)

            # Botão do canto superior direito
            botao_x1, botao_y1 = largura - 210, 20
            botao_x2, botao_y2 = largura - 20, 70
            if botao_x1 < indicador_x < botao_x2 and botao_y1 < indicador_y < botao_y2:
                if time.time() - tempo_ultimo_clique_botao > 1.0:
                    modo_ajuste_livre = not modo_ajuste_livre
                    tempo_ultimo_clique_botao = time.time()

            # Gesto de pinça para congelar/descongelar
            if verifica_pinca(hand_landmarks, largura, altura):
                if time.time() - tempo_ultimo_clique_pinça > 0.8:
                    linha_congelada = not linha_congelada
                    tempo_ultimo_clique_pinça = time.time()

            if rotulo_mao == "Right":
                mao_esquerda_dados = hand_landmarks
            elif rotulo_mao == "Left":
                mao_direita_dados = hand_landmarks

    # =========================
    # LÓGICA DE POSICIONAMENTO E ROTAÇÃO
    # =========================
    centro_atual_x = (linha_ponto_a[0] + linha_ponto_b[0]) // 2
    centro_atual_y = (linha_ponto_a[1] + linha_ponto_b[1]) // 2

    if modo_ajuste_livre:
        if not linha_congelada:
            if mao_esquerda_dados:
                linha_ponto_a[0] = int(mao_esquerda_dados[8].x * largura)
                linha_ponto_a[1] = int(mao_esquerda_dados[8].y * altura)
            if mao_direita_dados:
                linha_ponto_b[0] = int(mao_direita_dados[8].x * largura)
                linha_ponto_b[1] = int(mao_direita_dados[8].y * altura)
        else:
            if mao_esquerda_dados:
                novo_centro_x = int(mao_esquerda_dados[9].x * largura)
                novo_centro_y = int(mao_esquerda_dados[9].y * altura)
                
                if centro_anterior_x is not None and centro_anterior_y is not None:
                    dx = novo_centro_x - centro_anterior_x
                    dy = novo_centro_y - centro_anterior_y
                    linha_ponto_a[0] += dx
                    linha_ponto_a[1] += dy
                    linha_ponto_b[0] += dx
                    linha_ponto_b[1] += dy
                
                centro_anterior_x = novo_centro_x
                centro_anterior_y = novo_centro_y
            else:
                centro_anterior_x = None
                centro_anterior_y = None
    else:
        centro_anterior_x = None
        centro_anterior_y = None
        
        if not linha_congelada and mao_esquerda_dados:
            centro_mao_y = int(mao_esquerda_dados[9].y * altura)
            deslocamento_y = centro_mao_y - centro_atual_y
            linha_ponto_a[1] += deslocamento_y
            linha_ponto_b[1] += deslocamento_y

        if mao_direita_dados:
            for ponta in pontas_dedos:
                ponto = mao_direita_dados[ponta]
                dedo_x = int(ponto.x * largura)
                dedo_y = int(ponto.y * altura)
                dedos_detectados.append((ponta, dedo_x, dedo_y))
                dedos_ativos_neste_frame[ponta] = True

    # =========================
    # PROCESSAMENTO DE TOQUE NAS NOTAS
    # =========================
    if not modo_ajuste_livre:
        for ponta, dedo_x, dedo_y in dedos_detectados:
            tocou, _ = ponto_cruza_segmento(dedo_x, dedo_y, linha_ponto_a[0], linha_ponto_a[1], linha_ponto_b[0], linha_ponto_b[1])
            if tocou:
                if not tecla_pressionada[ponta]:
                    nota = nota_do_dedo[ponta]
                    sons[nota].play()
                    tecla_pressionada[ponta] = True
                    efeito_dedo[ponta] = 15
            else:
                tecla_pressionada[ponta] = False
    else:
        for ponta in pontas_dedos:
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

    # Botão de Modo
    cor_botao = (0, 255, 0) if modo_ajuste_livre else (0, 0, 255)
    texto_botao = "MODO: AJUSTE" if modo_ajuste_livre else "MODO: TOCAR"
    cv2.rectangle(frame, (largura - 210, 20), (largura - 20, 70), cor_botao, -1)
    cv2.putText(frame, texto_botao, (largura - 200, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Cor da Linha
    if linha_congelada:
        cor_linha = (0, 165, 255)
    elif modo_ajuste_livre:
        cor_linha = (255, 0, 255)
    else:
        cor_linha = (0, 255, 255)

    cv2.line(frame, (linha_ponto_a[0], linha_ponto_a[1]), (linha_ponto_b[0], linha_ponto_b[1]), cor_linha, 3)

    # =========================
    # NOVIDADE: CÍRCULO GUIA E CÁLCULO DE ÂNGULO EM GRAUS
    # =========================
    # Calcula o ângulo da linha em relação ao eixo horizontal
    dx = linha_ponto_b[0] - linha_ponto_a[0]
    dy = linha_ponto_b[1] - linha_ponto_a[1]
    angulo_rad = math.atan2(dy, dx)
    angulo_graus = int(math.degrees(angulo_rad))

    if modo_ajuste_livre:
        # Desenha círculos refletores/níveis de rotação nas pontas A e B
        cv2.circle(frame, (linha_ponto_a[0], linha_ponto_a[1]), 15, (0, 255, 0), 2)
        cv2.circle(frame, (linha_ponto_b[0], linha_ponto_b[1]), 15, (0, 255, 0), 2)
        
        # Desenha um círculo de rotação/nível no centro exato da linha exibindo os graus
        cv2.circle(frame, (centro_atual_x, centro_atual_y), 30, (255, 0, 255), 2)
        cv2.putText(frame, f"{angulo_graus} deg", (centro_atual_x - 25, centro_atual_y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    # Mensagens de Estado
    if linha_congelada:
        label_linha = f"CONGELADA | Angulo: {angulo_graus}° (PINCA p/ descongelar)"
    elif modo_ajuste_livre:
        label_linha = f"AJUSTE LIVRE | Angulo: {angulo_graus}° (PINCA p/ congelar)"
    else:
        label_linha = "LINHA DE TOQUE ATIVA"

    cv2.putText(frame, label_linha, (min(linha_ponto_a[0], linha_ponto_b[0]), min(linha_ponto_a[1], linha_ponto_b[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_linha, 1, cv2.LINE_AA)

    # Caixinhas nos dedos (modo tocar)
    if not modo_ajuste_livre:
        for ponta, dedo_x, dedo_y in dedos_detectados:
            nota = nota_do_dedo[ponta]
            
            if efeito_dedo[ponta] > 0:
                tamanho_caixa = 25
                cor_caixa = (0, 255, 0)
                espessura = -1
            else:
                tamanho_caixa = 35
                cor_caixa = (255, 20, 147)
                espessura = 2

            x1 = dedo_x - tamanho_caixa // 2
            y1 = dedo_y - tamanho_caixa // 2
            x2 = dedo_x + tamanho_caixa // 2
            y2 = dedo_y + tamanho_caixa // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa, espessura)
            cv2.putText(frame, f"{nota}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_caixa if espessura != -1 else (255, 255, 255), 2, cv2.LINE_AA)

    for ponta in efeito_dedo:
        if efeito_dedo[ponta] > 0:
            efeito_dedo[ponta] -= 1

    cv2.imshow("Pianinho com Indicador de Angulo e Rotação", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# ENCERRAR
# =========================
camera.release()
landmarker.close()
cv2.destroyAllWindows()