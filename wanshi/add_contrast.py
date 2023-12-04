import os
import cv2

def enhance_contrast(image_path, output_path, alpha, beta):
    # Open the image file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert image to grayscale

    # Enhance Image Contrast
    image_enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Save the enhanced image
    cv2.imwrite(output_path, image_enhanced)

# Set the directory you want to start from
rootDir = '/home/jeff/test_slices'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith('.jpg'):
            print('\t%s' % fname)
            input_path = os.path.join(dirName, fname)
            #mkdir if not exist
            if not os.path.exists(os.path.join(dirName, 'enhanced')):
                os.mkdir(os.path.join(dirName, 'enhanced'))
            output_path = os.path.join(dirName, 'enhanced', fname)
            print(output_path)
            enhance_contrast(input_path, output_path, 3, 0)
