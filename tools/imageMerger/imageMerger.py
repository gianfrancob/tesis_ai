import os
import cv2
import itertools
import numpy as np

def loadImages(path):
    # Load images from path
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), -1)
        if img is not None:
            images.append(img)
    return images

def generateFakeBackgrounds(outputPath):
    # Generate background images from scratch
    for red in range(0, 257, 64):
        for blue in range(0, 257, 64):
            for green in range(0, 257, 64):
                # Create a blank 300x300 black image
                image = np.zeros((300, 300, 3), np.uint8)
                # Fill image with red color(set each pixel to red)
                red = 255 if red == 256 else red
                green = 255 if green == 256 else green
                blue = 255 if blue == 256 else blue
                image[:] = (blue, green, red)
                cv2.imwrite(f'{outputPath}/R{red}G{green}B{blue}.PNG', image)


'''
def augmentData(images):
    # Apply image transformations to images
    # - Rotation
    # - Size
    # - Crop
    # - Colour: Saturation, brightness
    # - Noise
    
'''

'''
This function paste a foregrond image over abackground image in a given position
Returns a tuple with the new image and the bounding box
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
    xmin, xmax = max(x, 0), min(x+fw, bw)
    ymin, ymax = max(y, 0), min(y+fh, bh)
    y_min = ymin-y
    y_max = ymax-y
    x_min = xmin-x
    x_max = xmax-x
    if (y_min >= 0 and y_max >= 0 and x_min >= 0 and x_max >= 0):
        if (y_max - y_min >= fh/4 and x_max - x_min >= fw/4 ):
            front_cropped = front[y_min:y_max, x_min:x_max]
            back_cropped = back[ymin:ymax, xmin:xmax]

            alpha_front = front_cropped[:,:,3:4] / 255
            alpha_back = back_cropped[:,:,3:4] / 255
            
            print("ymin:ymax ", ymin, ":", ymax)
            print("xmin:xmax ", xmin, ":", xmax)
            print("ymin-y:ymax-y ", ymin-y, ":", ymax-y)
            print("xmin-x:xmax-x ", xmin-x, ":", xmax-x)
            print("front_cropped Y: ", front_cropped.shape[0])
            print("front_cropped X: ", front_cropped.shape[1])
            print("alpha_front Y: ", alpha_front.shape[0])
            print("alpha_front X: ", alpha_front.shape[1])
            # replace an area in result with overlay
            result = back.copy()
            print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
            result[ymin:ymax, xmin:xmax, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
            result[ymin:ymax, xmin:xmax, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255


            cv2.imshow('image', result)
            cv2.waitKey(0)

            bb = (xmin, ymin, xmax, ymax)

            return result, bb

def mergeImages(foregrounds, backgrounds):
    # Paste image over background in random place and calculate BB
    results = []
    for front in foregrounds[0:1]:
        v_offset = int(front.shape[0]) # / 2
        h_offset = int(front.shape[1]) # / 2
        for back in backgrounds[0:1]:
            v_step = int((back.shape[0] + 2*v_offset) / 8)
            h_step = int((back.shape[1] + 2*h_offset) / 8)

            y = range(-v_offset, back.shape[0] + v_offset, v_step)
            x = range(-h_offset, back.shape[1] + h_offset, h_step)

            positions = list(itertools.product(x, y))
            for pos in positions:
                img = mergeImage(front, back, pos)
                if type(img) != type(None):
                    results += [img]

    return results

def saveDataset(data, className, outputPath, extension="JPG"):
    # Save dataset in the give path
    # It saves all tyhe images with it's corresponding label file
    for i in range(len(data)):
        name = outputPath + f'{className}/{className}_{i}'
        img, label = data[i]
        with open(f'{name}.txt', 'w') as labelFile:
            cv2.imwrite(f'{name}.{extension}', img)
            labelFile.write(f'{className}/{className} {label[0]} {label[1]} {label[2]} {label[3]}')


def main(foregroundsPath, backgroundsPath, className, outputPath):
    foregrounds = loadImages(foregroundsPath)
    # foregrounds = augmentData(foregrounds)

    backgrounds = loadImages(backgroundsPath)
    # backgrounds = augmentData(backgrounds)
    fakeBackgroundsPath = "./fakeBackgounds"
    # generateFakeBackgrounds(fakeBackgroundsPath)
    backgrounds += loadImages(fakeBackgroundsPath)

    mergedImages = mergeImages(foregrounds, backgrounds)

    saveDataset(mergedImages, className, outputPath)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate and Augmented Dataset from Object Images and Backgrounds')
    parser.add_argument('--foregrounds', metavar='path', required=True,
                        help='the path to foregrounds')
    parser.add_argument('--backgrounds', metavar='path', required=True,
                        help='path to backgrounds')
    parser.add_argument('--className', required=True,
                        help='name of the class')
    parser.add_argument('--output', metavar='path', required=True,
                        help='path to output dataset')
    args = parser.parse_args()
    
    main(args.foregrounds, args.backgrounds, args.className, args.output)