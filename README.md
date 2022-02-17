# Crack detection using OpenCV

### Abstract
Construction industury is behind the technologies improvement comparing to the other sectors. However, automatization in construction sector will got effected in a possitive way especially in terms of safety, time/cost efficiency and improvement of maintenance industry. 
Regarding this factor creating safer environment especially in bridge or high rise building construction which put employee in a risky stuation is our aim. 

Lastly, our aim is making maintenance more efficient. In germany, %70 of construction activites are belongs to the maintenance. It's a huge area that we should improve.


### Introduction
When we need to repair the building, we need some information and analysis. Traditionally, this information is collected based on observation in place. However, it is a huge risk for employee. Moreover, If there is not a problem controlling process will put employees in a risk without reason. Additionally, this process need investment, equipment and engineer to control and organize process.
Based on these factors, our ultimate goal is to develop a system that can be able to detect these cracks on the bridge or high rise building automatically.


### Methodology
Here the crack detection methodology can be classified into some following steps below:
1. Image capture
2. Feature extraction
3. Image processing with permutations
4. Image Segmention
5. Comparing to the Results

#### Image capture
In a risky situation we offer using drone.

<img src="Input-Set/Cracked_01.jpg" width="400" height="250"> <img src="Input-Set/Cracked_07.jpg" width="400" height="250">

#### Image Evaluation techniques
Evaluation of brightness, sharpness, contrast and image size 

#### Permutations
It is not possible to run only one code for all crack photos. This process try different values in filters based on evaluation results.

#### Image processing techniques
All the steps in the processing section are being explained below. 

##### Contrast Enhancer
Makes cracks more visible

##### Gray scaling and averaging
The images is transformed in a new one in grayscale and blur. These make the images easier to visualize the processed images in next steps. 

<img src="Processed-Set/blur-1.jpg" width="400" height="250"> <img src="Processed-Set/blur-7.jpg" width="400" height="250">
<pre>              Blurred Image                                           Blurred Image</pre>

##### Logarithmic transformation
Logarithmic transformation is used to replace all the pixels values of an image with its logarithmic values. This transformation is used for image enhancement as it expands dark pixels of the image as compared to higher pixel values. So if we apply this method in an image having higher pixel values then it will enhance the image more and actual information of the image will be lost. Now after applying the log transformation in to our sample blurred images, they look like below.

<img src="Processed-Set/img_log-1.jpg" width="400" height="250"> <img src="Processed-Set/img_log-7.jpg" width="400" height="250">
<pre>           Log Transformed Image                                   Log Transformed Image</pre>

##### Image smoothing: bilateral filter
The bilateral filter also uses a Gaussian filter in the space domain, but it also uses one more (multiplicative) Gaussian filter component which is a function of pixel intensity differences. This method preserves edges, since for pixels lying near edges, neighboring pixels placed on the other side of the edge, and therefore exhibiting large intensity variations when compared to the central pixel, will not be included for blurring. So the sample logarithmic transformed images become as following after applying the bilateral filtering.

<img src="Processed-Set/bilateral-1.jpg" width="400" height="250"> <img src="Processed-Set/bilateral-7.jpg" width="400" height="250">
<pre>           Bilateral Filtered Image                           Bilateral Filtered Image</pre>

#### Image Segmention Techniques
##### Canny edge detection
Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed. It uses a multi-stage algorithm to detect a wide range of edges in images. 
Canny algorithm consists of three main steps:

1. Find the intensity gradient of the image: In this step the scale of the gradient vector is calculated for each pixel.
2. Non-maximum suppression: The aim of this step is to “thin” the edge to obtain a one-pixel width edge.
3. Threshold hysteresis: Finally, a two-step threshold hysteresis is applied in order to decrease the fake edges.

Now we apply canny algorithm to detect the crack edges in our bilateral filtered as following.

<img src="Processed-Set/edges-1.jpg" width="400" height="250"> <img src="Processed-Set/edges-7.jpg" width="400" height="250">
<pre>           Canny Edges Image                                   Canny Edges Image</pre>

##### Morphological closing operator
Morphological transformations are some simple operations based on the image shape. It is normally performed on binary images. It needs two inputs, one is our original image, second one is called structuring element or kernel which decides the nature of operation. 

There are many different types of morphological filtering, but after analyzing the results, the best filter for this detection is the closing filter. Closing filter helps to fill minor gaps in the image making the main crack continuous and more detailed. It is useful in closing small holes inside the foreground objects, or small black points on the object. Closing filter is defined as a dilation followed by an erosion.

Here we go to apply the morphological closing operator onto our canny edges detected images.

<img src="Processed-Set/closing-1.jpg" width="400" height="250"> <img src="Processed-Set/closing-7.jpg" width="400" height="250">
<pre>           Morphological Closing Image                          Morphological Closing Image</pre>

#### Feature extraction
There are various types of algorithm like (SIFT, SURF, ORB) that could be used in feature detection/extraction. SIFT and SURF are patented so not free for commercial use, while ORB is free. SIFT and SURF detect more features than ORB, but ORB is faster. ORB stands for Oriented FAST and Rotated BRIEF. It combines features of FAST and BRIEF for feature extraction and description. It has very fast computational speed, efficient memory usage, and high matching accuracy. ORB can be used instead of SIFT and SURF for feature extraction. 

So after applying this ORB method into our morphological closing images we get the result as following.

<img src="Output-Set/CrackDetected-1.jpg" width="400" height="250"> <img src="Output-Set/CrackDetected-7.jpg" width="400" height="250">
<pre>           Feature Detected Image                              Feature Detected Image</pre>

### Comparison
Process has resulted with too many output in wide range. To pick the best compatible result, we compare the results with the original image based on SSIM (Structural Similarity Index) and MSE(Mean Square Error) value.

### Result and discussion
Here we tried with around twenty images of both crack and non-crack to test. Without some cases, the cracks become very visible accurately in our output image. We added permutations and comparison step to improve accuracy.

### References
1. M Ann, P Johnson, Best Practices Handbook on asphalt pavement maintenance.Minnesota Technology Transfer (T2) Center / LTAP Program (2000). http://www.cee.mtu.edu/~balkire/CE5403/AsphaltPaveMaint.pdf.

2. TD Donald Walker, Pavement Surface Evaluation and Rating (PASER) Manuals (Wisconsin Transportation Information Center, Wisconsin, 2002). http://www.apa-mi.org/docs/Asphalt-PASERManual.pdf.

3. RK Kay, Pavement Surface Condition - Rating Manual. Northwest Technologies Transfer Center. Northwest Technologies Transfer Center. Washington State Department of Transportation, (Washington, 1992).

4. L Li, L Sun, G Ning, S Tan, Automatic pavement crack recognition based on Bp neural network. PROMET-Traffic Transp. 26(1), 11–22 (2014).

5. NHTSA, National Motor Vehicle Crash Causation Survey Report to Congress, (2008). http://www-nrd.nhtsa.dot.gov/Pubs/811059.PDF, Accessed Jan 2017.

6. JM Palomares, J González, E Ros, in AERFAI 2005. Detección de bordes en imágenes con sombras mediante LIP–Canny (Simposio de Reconocimiento de Formas y Análisis de Imágenes, AERFAI’2005, At Granada, 2005).

7. JS Miller, RB Rogers, GR Rada, in Distress Identification Manual for the Long-Term Pavement Performance Project. Appendiz A - Pavement Distress Types and Causes (National Cooperative Highway Research Program, At NW Washington, 1993), pp. 1–31.
