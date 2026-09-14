import cv2
import mediapipe as mp

# Caminho do modelo
MODEL_PATH = "models/hand_landmarker.task"

# Configurações do Hand Landmarker
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Cria o detector
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# Abre a câmera
camera = cv2.VideoCapture(0)

# Contador de frames
frame_timestamp = 0

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break

    # Converte BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cria imagem para o MediaPipe
    imagem_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detecta a mão
    resultado = landmarker.detect_for_video(
        imagem_mp,
        frame_timestamp
    )

    # Desenha pontos e conexões
    if resultado.hand_landmarks:
        for hand_landmarks in resultado.hand_landmarks:

            altura, largura, _ = frame.shape

            # Desenha as linhas
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

            # Desenha os pontos
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

    # Mostra a câmera
    cv2.imshow("Hand Tracking", frame)

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_timestamp += 1

# Fecha tudo
camera.release()
landmarker.close()
cv2.destroyAllWindows()