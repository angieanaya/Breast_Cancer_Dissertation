import os
import shutil

PATH = "CESM_&_MASKS"
image_ids = next(os.walk(PATH))[1]
for id in image_ids:
    path = os.path.join(PATH, id, 'masks')
    dir = os.listdir(path)
    if len(dir) == 0:
        removePath = os.path.join(PATH, id + "/")
        print("Directory is empty")
        shutil.rmtree(removePath)