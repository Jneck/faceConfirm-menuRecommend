import ftplib

import os

def request_video(filename):
    ftp=ftplib.FTP()

    ftp.connect("192.168.0.175", 21)

    ftp.login("user01","pass01")

    ftp.cwd("/test01")

    fd = open("./" + filename, 'wb')

    ftp.retrbinary("RETR " + filename, fd.write)

    fd.close()
    return None
