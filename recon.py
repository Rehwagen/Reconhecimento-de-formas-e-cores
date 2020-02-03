import cv2
import numpy as np

vid = cv2.VideoCapture(0)

font= cv2.FONT_HERSHEY_COMPLEX

count_rc = 0
count_bc = 0
count_rs = 0
count_bs = 0

while True:
    _, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converte para o sistema HSV

    lower_red = np.array([0, 114, 132]) #Valores de hsv para regular o filtro vermelho
    upper_red = np.array([180, 255, 212])
    lower_blue = np.array([97, 141, 137]) #Valores de hsv para regular o filtro azul
    upper_blue = np.array([143, 255, 255])

    mask_r = cv2.inRange(hsv, lower_red, upper_red) #Filtra a imagem para que apenas a cor vermelha apareça
    mask_b = cv2.inRange(hsv, lower_blue, upper_blue) #Filtra a imagem para que apenas a cor azul apareça
    kernel = np.ones((2, 2), np.uint8) #Redução de ruído
    mask_r  = cv2.erode(mask_r, kernel)
    mask_b = cv2.erode(mask_b, kernel)

    contours_r, _ = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Captura as linhas de contorno do filtro vermelho
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Captura as linhas de contorno do filtro azul

    for cnt in contours_r:
        area = cv2.contourArea(cnt)
        aprox = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x = aprox.ravel() [0] #Captura as coordenadas x e y do objeto detectado
        y = aprox.ravel() [1]
        if area > 300:
            cv2.drawContours(frame, [aprox], 0, (0), 2)
            if len(aprox) == 4:
                cv2.putText(frame, "Quadrado vermelho", (x, y), font, 0.7, (0))
                count_rs += 1
            if len(aprox) > 10:
                cv2.putText(frame, "Circulo vermelho", (x, y), font, 0.7, (0))
                count_rc += 1

    for cnt in contours_b:
        area = cv2.contourArea(cnt)
        aprox = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x = aprox.ravel() [0] #Captura as coordenadas x e y do objeto detectado
        y = aprox.ravel() [1]
        if area > 300:
            cv2.drawContours(frame, [aprox], 0, (0), 2)
            if len(aprox) == 4:
                cv2.putText(frame, "Quadrado azul", (x, y), font, 0.7, (0))
                count_bs += 1
            if len(aprox) > 10:
                cv2.putText(frame, "Circulo azul", (x, y), font, 0.7, (0))
                count_bc += 1
            
    cv2.imshow("Normal", frame)
    cv2.imshow("Mask B", mask_b)
    cv2.imshow("Mask R", mask_r)
    print("Círculos vermelhos: {}   Círculos azuis: {}   Quadrados vermelhos: {}   Quadrados azuis: {}\n".format(count_rc, count_bc, count_rs, count_bs))
    count_rc = 0
    count_bc = 0
    count_rs = 0
    count_bs = 0

    k = cv2.waitKey(1)
    if k == 27:
        break
vid.release()
cv2.destroyAllWindows()