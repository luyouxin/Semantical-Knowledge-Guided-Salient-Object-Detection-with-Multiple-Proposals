import os, random, shutil


def moveFile1(fileDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = 0.3
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    print (sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    return


def moveFile2(tarDir):
    imageDir = os.listdir(tarDir)
    maskDir = os.listdir(mDir)

    for image in imageDir:
        for mask in maskDir:
            a = image[:-4]
            b = mask[:-4]
            if image[:-4] == mask[:-4]:
                shutil.move(mDir + mask, tDir + mask)
    return


if __name__ == '__main__':
    fileDir = "/home/dorothy/project/CGS/dataset/DUTS-TR/DUTS-TR-Image/"
    tarDir = '/home/dorothy/project/CGS/dataset/DUTS-VA/DUTS-VA-Image/'

    mDir = "/home/dorothy/project/CGS/dataset/DUTS-TR/DUTS-TR-Mask/"
    tDir = '/home/dorothy/project/CGS/dataset/DUTS-VA/DUTS-VA-Mask/'
    # moveFile1(fileDir)
    moveFile2(tarDir)
















