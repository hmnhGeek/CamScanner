import cv2
import numpy as np
import rect
from matplotlib import pyplot as plt


class scanner(object):
    ''' To Scan an Image, say img.jpg
        -----------------------------

        1. s = scanner("img.jpg")
        2. Without this step, you can't move further. s.start_scanning()

        Now you can use all the functions of the script.
        To see functions type "help(scanner)".

        Note: 1. self.show_scanned() will show you the final scanned image. This
                is the image we finally need.

              2. To save any image, once it is opened, click 's'. It will close the
                  window and save the image automatically in working directory.
        
        '''
    def __init__(self, image_adr):
        self.image = image_adr
        self.scanned = ''
        self.all = []

    def start_scanning(self):
        ''' Main function, which scans the image. '''
        
        # add image here.
        # We can also use laptop's webcam if the resolution is good enough to capture
        # readable document content
        image = cv2.imread(self.image)

        # resize image so it can be processed
        # choose optimal dimensions such that important content is not lost
        image = cv2.resize(image, (1500, 880))

        # creating copy of original image
        orig = image.copy()

        # convert to grayscale and blur to smooth
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #blurred = cv2.medianBlur(gray, 5)

        # apply Canny Edge Detection
        edged = cv2.Canny(blurred, 0, 50)
        orig_edged = edged.copy()

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        (_,contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        #x,y,w,h = cv2.boundingRect(contours[0])
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

        # get approximate contour
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break


        # mapping target points to 800x800 quadrilateral
        approx = rect.rectify(target)
        pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

        M = cv2.getPerspectiveTransform(approx,pts2)
        dst = cv2.warpPerspective(orig,M,(800,800))

        cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


        # using thresholding on warped image to get scanned effect (If Required)
        ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        
        self.all = [orig, gray, blurred, orig_edged, image, th1, th2, th3, th4, dst]
        self.scanned = th1

    def showAll(self):
        l = self.all

        orig = l[0]
        gray = l[1]
        blurred = l[2]
        orig_edged = l[3]
        image = l[4]
        th1 = l[5]
        th2 = l[6]
        th3 = l[7]
        th4 = l[8]
        dst = l[9]
        
        self.__use_matplotlib(orig, gray, blurred, orig_edged, image, th1, th2, th3, th4, dst)

    def __use_matplotlib(self, orig, gray, blurred, orig_edged, image, th1, th2, th3, th4, dst):
        plt.subplot(2,5,1),plt.imshow(orig),plt.title('Original.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,2),plt.imshow(gray),plt.title('Original Gray.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,3),plt.imshow(blurred),plt.title('Original Blurred.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,4),plt.imshow(orig_edged),plt.title('Original Edged.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,5),plt.imshow(image),plt.title('Outline.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,6),plt.imshow(th1),plt.title('Thresh Binary.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,7),plt.imshow(th2),plt.title('Thresh mean.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,8),plt.imshow(th3),plt.title('Thresh gauss.jpg')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,9),plt.imshow(th4),plt.title("Otsu's.jpg")
        plt.xticks([]), plt.yticks([])
        plt.subplot(2,5,10),plt.imshow(dst),plt.title('dst.jpg')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def show_scanned(self):
        self.__show("Scanned Image", self.scanned)

    def __show(self, title, image):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)
        
        if(cv2.waitKey(0) == ord('s')):
            self.__save_as(title, '.png', image)
        cv2.destroyAllWindows()
    
    def __save_as(self, name, extension, image):
        cv2.imwrite(name+extension, image)

    def show_original(self):
        self.__show('Original Image', self.all[0])

    def show_gray_image(self):
        self.__show('Gray Image', self.all[1])

    def show_blurred_image(self):
        self.__show('Blurred Image', self.all[2])

    def show_orignal_edged(self):
        self.__show('Edged Image', self.all[3])


    def show_outline(self):
        self.__show('Outline Image', self.all[4])

    def show_threshold_binary(self):
        self.__show('Binary Threshold Image', self.all[5])

    def show_threshold_mean(self):
        self.__show('Mean Threshold Image', self.all[6])

    def show_threshold_gaussian(self):
        self.__show('Gaussian Threshold Image', self.all[7])

    def show_otsu(self):
        self.__show('Otsu\'s Image', self.all[8])

    def show_dst(self):
        self.__show('DST Image', self.all[9])


