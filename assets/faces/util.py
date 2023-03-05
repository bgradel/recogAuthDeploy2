import glob
import os
import shutil

def copyAndRenameFaces(folder, outputLocation):
    imagePattern = folder + '/*.jpg'
    imgList=glob.glob(imagePattern)
    print(folder)
    print(imgList)
    
    n = 0
    for img in imgList:
        output = outputLocation + '/' + str(n) + '.jpg'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        shutil.copy(img, output)
        n += 1
        
def estimateNumberOfChaffImages(numUserImages):
    return math.floor(numUserImages*4.5)
        
        
#copyAndRenameFaces("./user1/chaff", "./user1/chaff_renamed")