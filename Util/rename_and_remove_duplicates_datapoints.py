import os

paths = ['./Celeb-real-avg',
         './Celeb-real-diff',
         './Celeb-real-rnd',
         './Celeb-synthesis-avg',
         './Celeb-synthesis-diff',
         './Celeb-synthesis-rnd']

for path in paths:
    files = os.listdir(path)
    for file in files:
        if 'mp4' in str(file):
            renamed_file = f'{file[:-8]}.png'
            os.rename(os.path.join(path, file), 
                      os.path.join(path, renamed_file))
            
        if '(' in str(file) and ')' in str(file):
            os.unlink(os.path.join(path, file))