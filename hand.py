import cv2
import mediapipe as mp
import time

#dar code zir ma webcam ra farakhoni mikonim
cap = cv2.VideoCapture(0)

#in khat miad on class marbote ro seda mikone
#ba estefade az in class mitonim handtrick dashte bashim
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
#dar dastor zir marbote be libery time fps tasvier ro bedast miarim
prev_time = 0 
curr_time = 0

#ma dar khat zir miyaim yek halghe binahayat minevisim 
while True:
    #nesbat be halhge bala chon opencv image mikhone miyam mizaim ke be sorat binahayat image bekhne like movie
    sec, img = cap.read()
    #be sorat khodkar opencv baraye shenasaei dast az ranghaye BGR estefade mikone dar code zir ma migim be ja BGR ba RGB moqeiat dast peydakon
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
#dar code haye zir miaym migim noqat(mafasel dast)be ma neshon bede
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0 or id == 4:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
#dar kaht zir noqat khat 25 , 26 be ham vasl mikone
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
#miam fps va time migirm
    curr_time = time.time()
    #fomol fps
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
#miad fps ro tasvir mindaze
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 
                3, (255, 0, 0), 3)

#dar 2 dastor zir ma miyam migim webcam baz beshe va tasvir maro bekhone
    cv2.imshow("webcam", img)
    cv2.waitKey(1)