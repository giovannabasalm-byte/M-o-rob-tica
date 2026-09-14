# carrega o OpenCV
import cv2
# Abre a câmera, o 0 indica que queremos usar a primeira câmera conectada ao computador
camera = cv2.VideoCapture(0)
#Cria uma loop que continua funcionando enquanto a câmera estiver aberta
while True:
    sucesso, frame = camera.read() # pega o próximo frame da câmera ( Frame é a captura daquele instante)

    if not sucesso:
        print("Não foi possível acessar a câmera.") # caso não funcione o comando!
        break
# Mostra esses frames em uma janela chamada "Minha camera"
    cv2.imshow("Minha camera", frame) 

    if cv2.waitKey(1) & 0xFF == 27: # permite o programa a acessar a tecla ESC.
        break
#Quando saímos, liberamos a câmera e fechamos a janela.
camera.release()
cv2.destroyAllWindows()