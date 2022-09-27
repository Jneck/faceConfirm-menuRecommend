import ftplib

import os

def request_video(filename):
    ftp=ftplib.FTP()
    try:
        ftp.connect("192.168.0.47", 21)
    except:
        print("동영상 업데이트에 실패하였습니다")
        return None

    ftp.login("user01","pass01")

    ftp.cwd("/test01")

    fd = open("face_recognition/data/video/" + filename, 'wb')

    ftp.retrbinary("RETR " + filename, fd.write)

    fd.close()
    return "face_recognition/data/video/" + filename
