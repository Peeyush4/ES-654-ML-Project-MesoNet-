import os, shutil, numpy as np
files = os.listdir("train_videos")
files = np.random.choice(files, int(0.3*len(files)), replace = False)
if not os.path.isdir("test_videos"):
    os.mkdir("test_videos")
for file in files:
        shutil.move("train_videos/"+file, "test_videos")