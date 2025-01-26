import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():  # `iteritems()` -> `items()`
            txn.put(k.encode(), v.encode() if isinstance(v, str) else v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def load_image_label_pairs(file_path):
    """
    Load image paths and labels from a file.

    Args:
        file_path: Path to the text file containing image-label pairs.
    
    Returns:
        imagePathList, labelList
    """
    imagePathList = []
    labelList = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_path, label = line.strip().split(',')
            imagePathList.append(image_path)
            labelList.append(label)
    
    return imagePathList, labelList


if __name__ == '__main__':
    # Example usage
    mapping_file = "./labels.txt"  # Replace with the correct path to your text file
    output_lmdb_path = "./lmdb_dataset"

    # Load image-label pairs
    imagePathList, labelList = load_image_label_pairs(mapping_file)

    # Create LMDB dataset
    createDataset(output_lmdb_path, imagePathList, labelList, checkValid=True)