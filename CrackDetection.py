import glob
import os
import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
import matplotlib.pyplot as plot
from skimage import measure
from collections import namedtuple


def mse(imageA, imageB):
  # the 'Mean Squared Error' between the two images is the
  # sum of the squared difference between the two images;
  # NOTE: the two images must have the same dimension
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1])

  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err


def compare_images(imageA, imageB, title):
  # compute the mean squared error and structural similarity
  # index for the images
  m = mse(imageA, imageB)
  s = measure.compare_ssim(imageA, imageB)

  # # setup the figure
  fig = plt.figure(title)
  plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
  return m, s


# get project path
folder_path = os.getcwd()
for filename in glob.glob(os.path.join(folder_path + '/Input-Set/', '*')):
  with open(filename, 'r') as f:

    currentMse, currentSsim = -99999, 99999
    # start of image size
    image = Image.open(filename)
    width, height = image.size
    im = image.size
    print("image size is ", im)
    # end of image size

    # start of sharpness
    im = Image.open(filename).convert('L')  # to grayscale
    array = np.asarray(im, dtype=np.int32)
    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = np.average(gnorm)
    # print("sharpness is ", sharpness)

    scale = math.ceil(sharpness)
    if scale == 0:
      scale = 1
    # print("sharpness value as an integer", scale)
    # end of sharpness

    # start of brightness
    # Convert the image te RGB
    imag = im.convert('RGB')
    # coordinates of the pixel
    X, Y = 0, 0
    # Get RGB
    pixelRGB = imag.getpixel((X, Y))
    R, G, B = pixelRGB
    brightness = sum([R, G, B]) / 3  ##0 is dark (black) and 255 is bright (white)
    # print("brightness is ", brightness)
    # end of brightness



    for sharpnessItr in np.linspace(sharpness*50/100, sharpness*800/100, num=4):
      for brightnessItr in np.linspace(brightness*50/100, brightness*200/100, num=4):
        # read a cracked sample image
        img = cv2.imread(filename)

        # Contrast Enhancement
        pil_image = Image.open(filename)
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_enhanced_image = contrast_enhancer.enhance(2)
        enhanced_image = np.asarray(pil_enhanced_image)
        r, g, b = cv2.split(enhanced_image)
        enhanced_image = cv2.merge([b, g, r])

        # Filter
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (9, 9))
        print(filename)

        # Apply logarithmic transform
        # img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * int(brightnessItr / 2 )
        img_log = (np.log(blur + 10) / (np.log(1 + np.max(blur)))) * int(brightnessItr)

        # Specify the data type
        img_log = np.array(img_log, np.uint8)

        # Image smoothing: bilateral filter
        bilateral = cv2.bilateralFilter(img_log, int(sharpnessItr) , int(sharpnessItr) * 5, int(sharpnessItr) * 5)

        # Canny Edge Detection
        edges = cv2.Canny(bilateral, int(sharpnessItr) , int(sharpnessItr) * 2 )

        # Morphological Closing Operator
        kernel = np.ones((int(100 / brightnessItr), 4), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Create feature detecting method
        # sift = cv2.xfeatures2d.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create()
        # orb = cv2.ORB_create(nfeatures=1500)

        orb = cv2.ORB_create(nfeatures=10)

        # Make featured Image
        keypoints, descriptors = orb.detectAndCompute(closing, None)
        featuredImg = cv2.drawKeypoints(closing, keypoints, None)

        replacedOutputFileName = filename.replace('Input-Set', 'Output-Set')
        generatedOutputFileName = replacedOutputFileName.replace('.jpg', str(sharpnessItr) + '_' + str(brightnessItr) + '.jpg')
        print("sharpness is ", sharpness)
        print("brightness is ", brightness)
        print("sharpnessItr is ", str(sharpnessItr))
        print("brightnessItr is ", str(brightnessItr))





        # Comparison
        m, s = compare_images(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(featuredImg, cv2.COLOR_BGR2GRAY), "Original vs. Crack")

        print("m : ", m)
        if m > currentMse:
          currentMse = m
          imgMse = featuredImg
          imgMsePath = replacedOutputFileName.replace('.jpg', "currentMse" + '.jpg')

        print("s : ", s)
        if s <= currentSsim:
          currentSsim = s
          imgSsim = featuredImg
          imgSsimPath = replacedOutputFileName.replace('.jpg', "currentSsim" + '.jpg')

        # sigma = 0.7
        # median = np.median(featuredImg)
        # print("median is ", median)
        # lower = int(max(0, (1.0 - sigma) * median))
        # print("lower is ", lower)
        # upper = int(min(255, ((1.0 + sigma) * median)) - 175)
        # print("upper is ", upper)
        # auto_canny = cv2.Canny(featuredImg, lower, upper)
        # # Create an output image
        # img = cv2.cvtColor(featuredImg, cv2.COLOR_BGR2GRAY)
        # image_array = np.array(img)
        # thresh, image_black = cv2.threshold(auto_canny, lower, 255, cv2.THRESH_BINARY)
        # contours = cv2.findContours(image_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # final_image = cv2.drawContours(img, contours, -1, Scalar(75, 10, 20), 3)
        cv2.imwrite(generatedOutputFileName, featuredImg)

        # Use plot to show original and output image
        plot.subplot(121), plt.imshow(image)
        plot.title('Original'), plt.xticks([]), plt.yticks([])
        plot.subplot(122), plt.imshow(featuredImg, 'gray')
        plot.title('Crack'), plt.xticks([]), plt.yticks([])
        plot.show()

        # show slowly via ThreadSleep
        # time.sleep(2.5)

  print("currentMse : ", currentMse)
  print("currentSsim : ", currentSsim)
  cv2.imwrite(imgMsePath, imgMse)
  cv2.imwrite(imgSsimPath, imgSsim)