import os
import cv2

def loadImages(path):
    # Load images from path
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), -1)
        if img is not None:
            images.append(img)
    return images
'''
def generateFakeBackgrounds():
    # Generate background images from scratch

def augmentData(images):
    # Apply image transformations to images
    # - Rotation
    # - Size
    # - Crop
    # - Colour: Saturation, brightness
    # - Noise
    
'''
def mergeImage(front, back, pos):
    x, y = pos

    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh, bw = back.shape[:2]
    fh, fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    # replace an area in result with overlay
    result = back.copy()
    print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result

def mergeImages(foregrounds, backgrounds):
    # Paste image over background in random place and calculate BB

    results = []
    for front in [foregrounds[10]]:
        for back in [backgrounds[0]]:
            positions = []

            # TODO: Generate random point
            x = int(back.shape[0] / 2)
            y = int(back.shape[1] / 2)

            positions += [(x, y)]
            for pos in positions:
                results += [mergeImage(front, back, pos)]

    return results
'''
def saveDataset(outputPath):
    # Save dataset in the give path
'''
def main(foregroundsPath, backgroundsPath, outputPath):
    foregrounds = loadImages(foregroundsPath)
    # foregrounds = augmentData(foregrounds)

    backgrounds = loadImages(backgroundsPath)
    # backgrounds = augmentData(backgrounds)
    # backgrounds += generateFakeBackgrounds()

    mergedImages = mergeImages(foregrounds, backgrounds)
    cv2.imshow('image', mergedImages[0])
    cv2.waitKey(0)

    # saveDataset(outputPath)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate and Augmented Dataset from Object Images and Backgrounds')
    parser.add_argument('--foregrounds', metavar='path', required=True,
                        help='the path to foregrounds')
    parser.add_argument('--backgrounds', metavar='path', required=True,
                        help='path to backgrounds')
    parser.add_argument('--output', metavar='path', required=True,
                        help='path to output dataset')
    args = parser.parse_args()
    
    main(args.foregrounds, args.backgrounds, args.output)