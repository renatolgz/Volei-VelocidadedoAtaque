import numpy as np
import matplotlib.pyplot as plt
import cv2
import moviepy
from moviepy import *
import scipy 
from scipy.signal import find_peaks

video_path = r"C:\Users\60102062\Downloads\facul3.mov"
cap = cv2.VideoCapture(video_path)

#Matriz de homografia do chão da quadra - para calcular o ponto de impacto da bola no chão
matriz_pixels = np.array([[1765, 1763],[127, 1681],[3167, 1607],[1623, 1595]], dtype="float32")
matriz_ms = np.array([[0,0],[7,1],[0,9],[9,9]], dtype="float32")

H,status = cv2.findHomography(matriz_pixels,matriz_ms)

#Matriz de homografia da rede - para calcular a altura e a posição X do momento do ataque
#A posição Z (profundidade) tem que ser estimada
matriz_pixels_rede = np.array([[3011, 1083],[3001, 911],[1671, 1199],[1673, 1079]], dtype="float32")
matriz_ms_rede = np.array([[0,2.43],[0,3.23],[9,2.43],[9,3.23]], dtype="float32")

H_rede, status_rede = cv2.findHomography(matriz_pixels_rede, matriz_ms_rede)

ret, frame = cap.read()

#determina em quais frames ocorreram os impactos usando o áudio do vídeo
try:
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_fps = audio.fps
    video_fps = video.fps

    array_audio = audio.to_soundarray()

    amplitude_abs = np.mean(abs(array_audio), axis =1)
    dist_amostras = int(0.1*audio_fps) 

    indice_picos, _ = find_peaks(amplitude_abs, height =0.3, distance = dist_amostras)

    tempo_picos = indice_picos / audio_fps
    frame_picos = (tempo_picos * 60).astype(int)
    frames = frame_picos - 2

except Exception as e:
    print(e)
    print('deu bo')

limite_inferior_cor = np.array([15 ,0 ,0])
limite_superior_cor = np.array([110 ,255 ,255])

#Inicio do tratamento da imagem
contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
subtrator = cv2.createBackgroundSubtractorMOG2()

#variáveis para armazenar os valores úteis
frame_counter = 0
frames_de_interesse = {}
posição_u = []
posição_v = []

#Loop que itera em todos os frames do vídeo, aplicando as transformações de imagem
#A máscara de movimento é fundamental para rastrear a bola, por isso o loop tem que iterar todos os frames, não somente os dois de interesse
while cap.isOpened():

    ret, frame = cap.read()

    frame_counter +=1
    
    if not ret:
        break

    tempo = frame_counter / video_fps

    #Processamento de imagem
    #COR
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara_cor = cv2.inRange(frame_hsv, limite_inferior_cor, limite_superior_cor)
    #MOVIMENTO
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_cinza, (5,5),0)
    final = contraste.apply(frame_blur)
    mascara = subtrator.apply(final)
    
    mascara_final = cv2.bitwise_and(mascara, mascara_cor)
    
    kernel_p = np.ones((7,7), np.uint8) 
    mascara_aberta = cv2.morphologyEx(mascara_final, cv2.MORPH_OPEN, kernel_p)
    kernel_g = np.ones((11,11), np.uint8) 
    mascara_fechada = cv2.morphologyEx(mascara_aberta, cv2.MORPH_CLOSE, kernel_g)

    contornos, hierarquia = cv2.findContours(mascara_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Redimensionamento para permitir a visualização
    width = int(frame.shape[1]*0.30)
    height = int(frame.shape[0]*0.30)
    dim = (width,height)


    #Inicio da lógica para isolar a bola
    melhor_contorno = None
    maior_circularidade = 0.0
    
    #Loop que itera sobre todos os contornos (tudo que se mexe no vídeo)
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        #Evita erros por divisão por 0
        if perimetro == 0:
            continue

        x, y, w, h = cv2.boundingRect(contorno) 
        if h == 0: 
            continue 
        
        aspect_ratio = float(w) / h
        
        if aspect_ratio < 0 or aspect_ratio > 300: #Não estou usando essa lógica por enquanto
            continue

        if area>1000:  
            circularidade = (4 * np.pi * area) / (perimetro * perimetro)

            #Isola o contorno mais circular como sendo a bola
            if circularidade > maior_circularidade:
                maior_circularidade = circularidade
                melhor_contorno = contorno

    #Desenha o contorno na imagem original para avaliação do tratamento de imagem e isolamento da bola
    if melhor_contorno is not None:
        cv2.drawContours(frame, [melhor_contorno], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    contornos_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    #Salva as informações da bola nos frames de interesse
    if frame_counter in frames:
        M = cv2.moments(melhor_contorno)

        if M["m00"] != 0:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        else:
            u,v = 0,0

        posição_u.append(u)
        posição_v.append(v)

        frames_de_interesse[frame_counter] = contornos_resized.copy()
        print(f"Frame {frame_counter} capturado na memória.")

    #Cria a janela que permite visualizar o vídeo
    cv2.imshow('Ataque - tentativa de identificar a bola', contornos_resized)
    cv2.waitKey(3) & 0xFF == ord('q')

audio.close()
video.close()
cap.release()
cv2.destroyAllWindows()

#Lógica para exibir os frames de interesse e o contorno avaliado em cada frame (para validação)
lista_de_frames = list(frames_de_interesse.keys())

f1numero = lista_de_frames[0]
f2numero = lista_de_frames[1]

f1imagem = frames_de_interesse[f1numero]
f2imagem = frames_de_interesse[f2numero]

fig = plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(f1imagem)
plt.title(f"Frame {f1numero}")
plt.axis('off') 

plt.subplot(1, 2, 2)
plt.imshow(f2imagem)
plt.title(f"Frame {f2numero}")
plt.axis('off')

plt.show()

#Definição dos pontos de ataque e pouso da bola
ponto1 = np.array([[[posição_u[0], posição_v[0]]]], dtype=np.float32)
ponto1projetado_rede = cv2.perspectiveTransform(ponto1, H_rede)
ponto2 = np.array([[[posição_u[1], posição_v[1]]]], dtype=np.float32)
ponto2projetado = cv2.perspectiveTransform(ponto2, H)

x1 = ponto1projetado_rede[0][0][0]
y1 = ponto1projetado_rede[0][0][1]
z1 = 9.1 #arbitrário
x2 = ponto2projetado[0][0][0]
y2 = 0.0 #a bola está tocando o chão
z2 = ponto2projetado[0][0][1]

ataque = (x1,y1,z1)
impacto = (x2,y2,z2)

ataque_array = np.array(ataque)
impacto_array = np.array(impacto)

#lógica para o cálculo da velocidade média
d_metros = np.linalg.norm(impacto_array - ataque_array)

t_ataque = tempo_picos[0]
t_impacto = tempo_picos[1]

tempo_seg = t_impacto - t_ataque

if tempo_seg >0:
    velocidade_ms = d_metros / tempo_seg
    velocidade_kmh = velocidade_ms * 3.6

else:
    print('deu bo')


ataque_formatado = f"({ataque[0]:.1f}, {ataque[1]:.1f}, {ataque[2]:.1f})"
impacto_formatado = f"({impacto[0]:.1f}, {impacto[1]:.1f}, {impacto[2]:.1f})"

print(f"Ponto de ataque: {ataque_formatado} metros")
print(f"Ponto do pouso: {impacto_formatado} metros")
print(f"Distância 3D percorrida pela bola: {d_metros:.2f} metros")
print(f"Tempo de voo da bola: {tempo_seg:.3f} segundos")
print(f"Velocidade Média: {velocidade_ms:.2f} m/s")
print(f"Velocidade Média: {velocidade_kmh:.2f} km/h")




