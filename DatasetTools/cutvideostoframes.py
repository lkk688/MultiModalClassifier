import glob
import os
import subprocess
import sys

# Include slash or it will search in the wrong directory!!
s = subprocess.check_output("ffmpeg -version", shell = True) 
print(s.decode("utf-8")) 
file_list = glob.glob("../videos/*.mp4")
print('file_list len: {}'.format(len(file_list)))
for video_file in file_list:
    simpletitle = video_file.split('/')[-1].split('.')[-2]
    ffmpegcmdstr = 'ffmpeg -i ' + \
        f'../videos/{simpletitle}.mp4 ' + \
        '-vf fps=1 ../TmpImages/'+simpletitle+'-%d.jpg'
    print(ffmpegcmdstr)
    
    #!ffmpeg - i ./videos/{simpletitle}.mp4 - vf fps = 1 ./TmpImages/{simpletitle}-%d.jpg
    #result = subprocess.call(ffmpegcmdstr, shell=True)
    s = subprocess.check_output(ffmpegcmdstr, shell = True) 
    print(s.decode("utf-8")) 
