import io
import pyqrcode
from base64 import b64encode
import eel
from  utils.camera import VideoCamera
import base64
import cv2
from shutil import copyfile
from utils.arcface_recognition import registered_face_loader,face_comparison
import argparse
from utils import face_model
import numpy as np
import os
import subprocess
import mysql.connector
import subprocess
from datetime import date, datetime

FMT = '%H:%M:%S'

def checkfresh(cursor,datetod,eid):
        sql = "SELECT EnEx FROM "+datetod+" WHERE Employee_ID = %s"
        val = (str(eid),)
        cursor.execute(sql,val)
        fresh = cursor.fetchall()
        return fresh

def checkTableExists(dbcon, tablename):
    dbcur = dbcon.cursor()
    dbcur.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{0}'
        """.format(tablename.replace('\'', '\'\'')))
    if dbcur.fetchone()[0] == 1:
        dbcur.close()
        return True

    dbcur.close()
    return False

def checktimer(tm,fresh):
    tdelta = datetime.strptime(tm, FMT) - datetime.strptime(fresh[0][0].split('_')[-1], FMT)
    tdelta=str(tdelta)
    if ', ' in str(tdelta):
            tdelta = str(tdelta).split(', ')[1]
    hw,mw,sw=tdelta.split(':')
    if int(sw)>30:
        return True
    else:
        return False

def greetcon(datetod,tot):
        sql = "SELECT EnEx FROM "+datetod+" WHERE Employee_ID = %s"
        val = (str(tot.split('#')[1]),)
        cursor.execute(sql,val)
        myresult = cursor.fetchall()
        if myresult==[]:
                return True
        else:
                return False
def gen():
    while True:
        frame,fcs = x.get_frame()    
        yield frame, fcs

@eel.expose
def video_feed():
    for each,fcs in gen():
            res=runrec(each,fcs)
            if res:
                eel.updateRecName(res[0],res[1],res[2])
                ret, jpeg1 = cv2.imencode('.jpg', res[3])
                blob = base64.b64encode(jpeg1.tobytes())
                blob = blob.decode("utf-8")
                eel.updateImageSrc(blob)()    
            else:
                ret, jpeg1 = cv2.imencode('.jpg', each)
                blob = base64.b64encode(jpeg1.tobytes())
                blob = blob.decode("utf-8")
                eel.updateImageSrc(blob)()
      
def runrec(frame,fcs):
            if len(fcs) != 0:
                        all_cat,all_sim,image,tot_names = face_comparison(frame,registered_feature,cat,model,threshold = 0.45) 
                        result = np.asarray(image)
                        if len(tot_names) !=0:
                            for names in tot_names:
                                print(names)
                                tot=names.split('_')
                                tot=' '.join(tot)
                                print("Attendance Noted for : "+tot)
                                #greet = greetcon(datetod,tot)
                                #conn.commit()
                                ye=str(date.today().year)[2]+str(date.today().year)[3]
                                str('{:0>2}'.format(date.today().day))+'-'+str('{:0>2}'.format(date.today().month))+'-'+ye
                                timenow=str(datetime.now().time().strftime("%H%M%S"))    
                                #wkd = os.path.join(cwd , 'Detections','rec#'+str('{:0>2}'.format(str(date.today().day))+'{:0>2}'.format(str(date.today().month))+str(date.today().year))+'#'+timenow+'#'+str(names).split('#')[0]+'#'+str(names).split('#')[1]+'.jpg')   
                                #cv2.imwrite(wkd,frame)
                                print(copyfile("Face_data/"+str(names)+'.jpg', 'web/thispic.jpg'))
                                namenow="NAME : "+tot.split('#')[0]
                                eidnow="EID : "+names.split('#')[1]
                                showtime="Time : "+str(datetime.now().time().strftime("%H:%M:%S")) 
                                namenowcon=tot.split('#')[0]
                                eidnowcon=names.split('#')[1]
                                showtimecon=str(datetime.now().time().strftime("%H:%M:%S"))
                                return namenow,eidnow,showtime,frame,namenowcon,eidnowcon,showtimecon
                        return False
            return False

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='face embedding and comparison')
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id,(-1) for CPU')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        parser.add_argument('--image_mode', default=False, action="store_true",help='Image detection mode')
        parser.add_argument('--regist', default=False, action="store_true",help='to regist face or to compare face')
        args = parser.parse_args()
        x=VideoCamera()
        eel.init('web')
        today = date.today()
        datetod =  str(today.strftime("%d_%m_%Y"))
        #conn = mysql.connector.connect(user='root', password='FRAS@db123', host='127.0.0.1', database='attendance_ccs')
        #cursor = conn.cursor()
        #print(checkTableExists(conn,datetod))
        #if not checkTableExists(conn,datetod):
        #        TableName ="CREATE TABLE "+datetod+" (Employee_Name VARCHAR(255),Employee_ID VARCHAR(255),EnEx VARCHAR(255));"
        #        cursor.execute(TableName)
        #        conn.commit()
        #else:
        #        print("EXISTS")
        #model = face_model.FaceModel(args)
        #cwd = os.getcwd()
        #registered_feature,cat = registered_face_loader()
        #SBS=subprocess.Popen("python Parallex.py",shell=True)
        #print(datetod)

        eel.start('home.html', size=(1366, 768))


        


    
    
    
