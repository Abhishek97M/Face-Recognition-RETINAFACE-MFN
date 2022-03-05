import cv2
import time

vc = cv2.VideoCapture("rtsp://admin:jetson123@192.168.20.128:554/Streaming/Channels/101")
#print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
vc.set(cv2.CAP_PROP_BUFFERSIZE, 0)
while True:
    
    success, frame = vc.read()
    frame = cv2.resize(frame , (640,480),interpolation = cv2.INTER_AREA)
    t1=time.time()
    #print(frame.shape)
    cv2.imshow("SHOW",frame)
    #print(time.time()-t1)
    #DETECTION ,TRACKING, RECOGNITION
    #time.sleep(0.050)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
