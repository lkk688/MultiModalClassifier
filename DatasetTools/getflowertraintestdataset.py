import urllib.request
import zipfile
import os

print('Beginning file download with urllib2...')

#test download image
# url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
# urllib.request.urlretrieve(url, '/Developer/MyRepo/ImageClassificationData/cat.jpg')

#Download the flower data from this link: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60a0_flower-photos/flower-photos.zip

url = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60a0_flower-photos/flower-photos.zip'
extract_dir = '/Developer/MyRepo/ImageClassificationData'
zip_path, _ = urllib.request.urlretrieve(url, os.path.join(extract_dir, 'flower-photos.zip'))
with zipfile.ZipFile(zip_path, "r") as f:
    f.extractall(extract_dir)

#test  train under /Developer/MyRepo/ImageClassificationData/flower_photos