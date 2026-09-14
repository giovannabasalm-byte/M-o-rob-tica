# inicio ligando com o fluxograma dá
"""
CÂMERA
   ↓
OpenCV pega a imagem
   ↓
MediaPipe analisa a imagem
   ↓
MediaPipe encontra as mãos
   ↓
MediaPipe marca 21 pontos em cada mão
   ↓
OpenCV desenha os pontos na tela
   ↓
AGORA: colocamos números nesses pontos
comando
   ↓
Arduino
   ↓
servo
   ↓
 mão robótica
""" 
# Importar as bibliotecas
"""
import cv2 # estamos importanto do OpenCV, pois ele trabalha com câmeras e com linguagens 
import mediapipe as mp # importar o mediapip
"""
# Modelo de mão

"""
MODEL_PATH = "models/hand_landmarker.task" # criamos um variavel chamada MODEL_PATH, ela guarda : models/hand_landmarker.task(arquivo task)
"""

#estrutura de pastas
"""
Mão robótica/
│
├── camera.py
├── hand_tracking.py
│
└── models/
    └── hand_landmarker.task
"""

# Configurando o MediaPipe
"""
options = mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(
    model_asset_path=MODEL_PATH
), 
# muitas configurações, pois dentro do parentese pedimos para ele possa seguir o modelo no caminho desejado!
E MODEL_PATH contém: models/hand_landmarker.task
Vou criar as configurações que o detector de mãos vai usar.
"""

# MOdo video
"""
running_mode=mp.tasks.vision.RunningMode.VIDEO,
Isso diz ao MediaPipe:
"Eu vou fornecer imagens de um vídeo, uma depois da outra
"""
#DUAS mãos
"""
num_hands=2 # detecta duas mãos!
"""

# O detector
"""
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
A variável: landmarker passa a representar nosso detector.
"""

#abrir a câmera
"""
camera = cv2.VideoCapture(0)
Estamos pedindo ao OpenCV: "Abra a câmera de número 0."
A variavel "camera" guarda o acessoa a camera
"""
# tipestamp
"""
frame_timestamp = 0 importante porque escolhi: RunningMode.VIDEO
O MediaPipe precisa saber a ordem temporal dos frames.

Começamos com:

frame_timestamp = 0

Depois:

frame_timestamp += 1

forma de fornecer uma sequência crescente de timestamps para os frames.
"""
# O loop
"""
while True: # eles significam para fazer algo repetinamente 
    sucesso, frame = camera.read() # estamos pedindo para o OpenCV: "Leia um frame da câmera."
    if not sucesso:
        print("Não foi possível acessar a câmera.")
        break
"""
#BGR → RGB
"""
O OpenCV normalmente trabalha com as cores na ordem:
BGR
Blue
Green
Red

Enquanto o MediaPipe espera:
RGB
Red
Green
Blue
essa linha não controla a cor rosa dos pontos.

Ela apenas transforma o formato de cores da imagem para o MediaPipe conseguir trabalhar corretamente.
"""

# tranformar a imagem para o MediaPipe
"""
imagem_mp = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=frame_rgb
)
transformando nossa imagem em um objeto que o MediaPipe consegue receber.

frame_rgb
   ↓
mp.Image
   ↓
imagem que o MediaPipe consegue analisar
"""
# Detectar mão
"""
resultado = landmarker.detect_for_video(
    imagem_mp,
    frame_timestamp
)

pede para analisar a imagem e o resultado fica guarado na variavl resultado
encontrou mão?
quantas mãos?
onde estão os landmarks?
"""
#Verificando se encontrou uma mão
"""
if resultado.hand_landmarks: caso apareça alguma resultado de landmarker

if resultado.hand_landmarks: Caso nenhum resultado aparecer!
Se uma ou duas mãos forem encontradas, teremos os landmarks.
"""
#Aqui entra a parte das DUAS mãos
""" 
for hand_landmarks in resultado.hand_landmarks:
    Para cada mão que o MediaPipe encontrou, faça o código abaixo
    resultado.hand_landmarks

   ↓

mão 1
mão 2
for: PEGA mão 1
→ desenha os pontos

PEGA mão 2
→ desenha os pontos
""" 