import os
import cv2

def enhance_edges(image_path, output_path, low_threshold, high_threshold):
    # Open the image file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert image to grayscale

    # Apply Gaussian blur to reduce noise
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(image_blur, low_threshold, high_threshold)

    # Save the enhanced image
    cv2.imwrite(output_path, edges)

# Set the directory you want to start from
rootDir = '/home/jeff/test_slices'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith('.jpg'):
            print('\t%s' % fname)
            input_path = os.path.join(dirName, fname)
            if not os.path.exists(os.path.join(dirName, 'enhanced_edge')):
                os.mkdir(os.path.join(dirName, 'enhanced_edge'))
            output_path = os.path.join(dirName, 'enhanced_edge', fname)
            print(output_path)
            enhance_edges(input_path, output_path, 100, 200)  # Adjust thresholds as needed
