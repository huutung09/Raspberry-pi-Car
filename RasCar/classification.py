import cv2
import numpy as np
#from matplotlib import pyplot as plt
from os import listdir
# local modules
from common import clock, mosaic
from skimage.feature import hog
from sklearn.svm import SVC
import pickle
import imutils
from math import sqrt
#Parameter
SIZE = 32
CLASS_NUMBER = 13
SIGNS = ["ERROR",
        "STOP",
        "TURN LEFT",
        "TURN RIGHT",
        "DO NOT TURN LEFT",
        "DO NOT TURN RIGHT",
        "START",
        "SPEED LIMIT",
        "OTHER"]
def load_traffic_dataset():
    dataset = []
    labels = []
    for sign_type in range(CLASS_NUMBER):
        sign_list = listdir("./dataset/{}".format(sign_type))
        for sign_file in sign_list:
            if '.png' in sign_file:
                path = "./dataset/{}/{}".format(sign_type,sign_file)
                print(path)
                img = cv2.imread(path,0)
                img = cv2.resize(img, (SIZE, SIZE))
                img = np.reshape(img, [SIZE, SIZE])
                dataset.append(img)
                labels.append(sign_type)
    return np.array(dataset), np.array(labels)


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SIZE*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self):
        filename = 'svc_model.pkl'
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

    def save(self):
        filename = 'svc_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovr')

    def train(self, samples, responses):
        self.model.fit(samples, responses)

    def predict(self, samples):
        return self.model.predict(samples)


def get_hog() : 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = 4   
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog


def training():
#    # print('Loading data from data.png ... ')
#    # Load data.
#    #data, labels = load_data('data.png')
#    data, labels = load_traffic_dataset()
#    print(data.shape)
#    print('Shuffle data ... ')
#    # Shuffle data
#    rand = np.random.RandomState(10)
#    shuffle = rand.permutation(len(data))
#    data, labels = data[shuffle], labels[shuffle]
#    
#    print('Deskew images ... ')
#    #Xử lý góc nghiêng
#    data_deskewed = list(map(deskew, data))
#    
#    print('Defining HoG parameters ...')
#    # HoG feature descriptor
#    hog = get_hog()
#
#    print('Calculating HoG descriptor for every image ... ')
#    hog_descriptors = []
#    for img in data_deskewed:
#        hog_descriptors.append(hog.compute(img))
#    hog_descriptors = np.squeeze(hog_descriptors)
#
#    # print('Spliting data into training (90%) and test set (10%)... ')
#    train_n=int(0.9*len(hog_descriptors))
#    data_train, data_test = np.split(data_deskewed, [train_n])
#    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
#    labels_train, labels_test = np.split(labels, [train_n])
#    
#    
#    print('Training SVM model ...')
#    model = SVM()
#    model.train(hog_descriptors_train, labels_train)

#    print('Saving SVM model ...')
#    model.save()
#    print(model)
#
#    return model

    svm_loaded = SVM()
    svm_loaded.load()
    return svm_loaded

def getLabel(model, data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    #print(np.array(img).shape)
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])

    return int(model.predict(hog_descriptors)[0])

### Preprocess image
#Enhance the visibility of features in low-light images or images with poor contrast.
def constrastLimit(image):
    #converts the image from BGR color space to YCrCb
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #splits the image into 3 channels: Y, Cr, and Cb
    channels = cv2.split(img_hist_equalized)
    temps = list(channels)
    # y channel improve the contrast of the image
    temps[0] = cv2.equalizeHist(channels[0])
    channels = tuple(temps)
    # channels[0] = cv2.equalizeHist(channels[0])
    #merges the channels back together and converts the image back to BGR color
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    #Reduce noise and smooth the image
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    #Converted to grayscale
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    #Enhance the edges
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    #Scales the result to 8-bit unsigned integer format
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
# binarization on an image    
def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE    )
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    #print(top,left,bottom,right)
    return image[top:bottom,left:right]

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return sign, coordinate

def localization(image, min_size_components, similitary_contour_with_circle, model):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

#    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))


#    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    text = ""
    sign_type = -1
    i = 0

    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 8 else 8
        text = SIGNS[sign_type]

    return coordinate, original_image, sign_type, text


def onPredictSign(vidcap, model):
    #Training phase
#    model = training()

#    vidcap = cv2.VideoCapture(0)
    # vidcap = cv2.VideoCapture(args.file_name)


    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, fps , (640,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    success = True
    similitary_contour_with_circle = 0.65   # parameter
    min_size_components = 250

#    while True:
    success,frame = vidcap.read()
#    if not success:
#        print("FINISHED")
#        break
    width = frame.shape[1]
    height = frame.shape[0]
    frame = cv2.resize(frame, (640,480))

#        print("Frame:{}".format(count))
    coordinate, image, sign_type, text = localization(frame,min_size_components, similitary_contour_with_circle, model)
#    if (sign_type > 0):
#            print(SIGNS[sign_type])
#            q.put(sign_type)
#        return sign_type
    return sign_type
            
#        print("Sign:{}".format(sign_type))
#        cv2.imshow('Result', image)
        #Write to video
        # out.write(image)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

    # print("Finish {} frames".format(count))
