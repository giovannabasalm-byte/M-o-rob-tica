import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)
frame_timestamp = 0

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    resultado = landmarker.detect_for_video(
        imagem_mp,
        frame_timestamp
    )

    if resultado.hand_landmarks:
        for hand_landmarks in resultado.hand_landmarks:

            altura, largura, _ = frame.shape

            # Desenha as conexões da mão
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

            # Desenha os pontos e seus números
            for numero, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * largura)
                y = int(landmark.y * altura)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    str(numero),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA
                )

            # Pega o landmark 8: ponta do indicador
            ponto_8 = hand_landmarks[8]

            x_8 = int(ponto_8.x * largura)
            y_8 = int(ponto_8.y * altura)

            cv2.putText(
                frame,
                f"Indicador: X={x_8} Y={y_8}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )

             # Pega o landmark 12: ponta do médio
            ponto_12 = hand_landmarks[12]

            x_12 = int(ponto_12.x * largura)
            y_12 = int(ponto_12.y * altura)

    cv2.putText(
                 frame,
                 f"Médio: X={x_12} Y={y_12}",
                 (20, 120),
                 cv2.FONT_HERSHEY_SIMPLEX,
                 0.7,
                 (255, 0, 255),
                 2,
                 cv2.LINE_AA
             )           

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_timestamp += 1

camera.release()
landmarker.close()
cv2.destroyAllWindows()