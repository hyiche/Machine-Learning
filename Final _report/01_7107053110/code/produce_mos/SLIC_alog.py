import numpy as np


class SLIC:

    def __init__(self, grayImg, abImg, numCLUS):
        self.img = np.copy(grayImg)
        self.height, self.width = self.img.shape[:2]
        # self.boundImg = np.zeros([self.height, self.width])
        self.mosaImgGray = np.copy(grayImg)
        self.mosaImgAB = np.copy(abImg)
        self.numCLUS = numCLUS

        # step: S
        self.step = int(np.sqrt(self.height * self.width / self.numCLUS))
        # maximum color distance (since L:0~1)
        self.nc = 1
        # maximum spatial distance
        self.ns = self.step
        self.FLT_MAX = 1000000
        self.ITERATIONS = 15
        self.generateSuperPixel()
        self.displayContours(0)
        self.Mosaica_gray()
        self.Mosaica_ab()

    def _initData(self):
        # img.shape[:2] only show the height and width of image
        # cluster center = [L, x, y]
        self.centers = np.zeros([self.numCLUS, 3])
        # label matrix (which category does each pixel belong to)
        self.clusters = -1 * np.ones(self.img.shape[:2])
        # distance matrix
        self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
        S = self.step
        # num of centers horizontal
        num_h = int(self.width / S)

        # for each superpixel
        for i in range(0, self.numCLUS):
            # x-axis
            x = int(S / 2 + S * (i % num_h) - 1)
            # y-axis
            y = int(S / 2 + S * np.floor(i/num_h) - 1)
            # Centering on (x, y), let the pixels in the outward S/2 distance be label i
            self.clusters[x - int(S / 2): x + int(S / 2), y - int(S / 2): y + int(S / 2)] = i
            # Adjustment center point
            nc = self._findLocalMinimum(center=(x, y))
            # new center point (x, y)
            x, y = nc[0], nc[1]
            # record i-th cluster center,  cluster center = [L, x, y]
            self.centers[i, :] = [self.img[x, y], x, y]

    """SLIC main model"""
    def generateSuperPixel(self):
        self._initData()
        S = self.step
        # (256, 256, 2)
        indnp = np.mgrid[0:self.height, 0:self.width].swapaxes(0, 2).swapaxes(0, 1)
        for i in range(self.ITERATIONS):
            self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
            # get the cluster Num
            for k in range(0, self.numCLUS):
                # restrict the search area
                xlow, xhigh = int(self.centers[k, 1] - S + 1), int(self.centers[k, 1] + S + 1)
                ylow, yhigh = int(self.centers[k, 2] - S + 1), int(self.centers[k, 2] + S + 1)
                if xlow <= 0:
                    xlow = 0
                if xhigh > self.width:
                    xhigh = self.width
                if ylow <= 0:
                    ylow = 0
                if yhigh > self.height:
                    yhigh = self.height

                # crop the image and compute the distance
                cropimg = self.img[ylow: yhigh, xlow: xhigh]
                colorDiff = cropimg - self.centers[k, 0]
                colorDist = np.sqrt(np.square(colorDiff))

                # yy.shape = (yhigh - ylow, 1) , xx.shape = (1, xhigh - xlow)
                yy, xx = np.ogrid[ylow: yhigh, xlow: xhigh]
                pixDist = ((xx - self.centers[k, 1])**2 + (yy - self.centers[k, 2])**2)**0.5
                # print(pixDist)
                # D'
                dist = ((colorDist[:, :]/self.nc)**2 + (pixDist/self.ns)**2)**0.5

                distanceCrop = np.copy(self.distances[ylow: yhigh, xlow: xhigh])
                # find the index which the distance is smaller than previous one
                idx = dist < distanceCrop
                # replacement
                distanceCrop[idx] = dist[idx]
                self.distances[ylow: yhigh, xlow: xhigh] = distanceCrop
                self.clusters[ylow: yhigh, xlow: xhigh][idx] = k

            """Adjustment the center of clustering, let the average be the new center"""
            all_base = np.ones([256, 256, 1])
            for k in range(0, self.numCLUS):
                # bool matrix if the pixel is the k-category then the Corresponding position is True else False
                idx = (self.clusters == k)
                # sum all luminance value of k-category
                colornp = np.sum(self.img[idx], axis=0)
                # when idx is True the value is 1, else 0
                basenp = all_base[idx]
                # find the all coordinate of the k-category
                distnp = indnp[idx]

                base = np.sum(basenp, axis=0)
                y, x = np.sum(distnp, axis=0)

                if int(base) is not 0:
                    y, x = y/base, x/base
                    self.centers[k] = [colornp/base, int(x), int(y)]

    """Move cluster centers to the lowest gradient position in a 3x3 neighbor"""
    def _findLocalMinimum(self, center):
        min_grad = self.FLT_MAX
        loc_min = center
        # Centering on (x, y), find the lowest gradient position in a 3x3 neighbor
        # because the image is read by skimage, x and y axis is exchanged (like cv2, image[y,x] = [h, w])
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                c1 = self.img[j + 1, i]
                c2 = self.img[j, i + 1]
                c3 = self.img[j, i]
                if ((c1 - c3) ** 2) ** 0.5 + ((c2 - c3) ** 2) ** 0.5 < min_grad:
                    min_grad = abs(c1 - c3) + abs(c2 - c3)
                    loc_min = [i, j]
        return loc_min

    """make the boundary map to original image (which boundary line is black)"""
    def displayContours(self, color):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]
        # bool matrix (all False) which record the pixel whether be use to draw boundary or not
        isTaken = np.zeros(self.img.shape[:2], np.bool)
        # use for record boundary coordinate
        contours = []

        # for each pixel
        for i in range(self.width):
            for j in range(self.height):
                nr_p = 0
                # search for 8-connected
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x>=0 and x < self.width and y>=0 and y < self.height:
                        # if the pixel has not use yet and the label of pixel is different from its neighbor
                        if isTaken[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    # record the (j, i) as True to draw boundary
                    isTaken[j, i] = True
                    contours.append([j, i])

        for i in range(len(contours)):
            self.img[contours[i][0], contours[i][1]] = color
            # self.boundImg[contours[i][0], contours[i][1]] = 1 - color

    """Method:
    Grab the pixel of the same label (class), calculate the average, then replacing those pixels with average value"""
    def Mosaica_ab(self):
        # for each cluster calculate the average
        color_shape = self.mosaImgAB.shape[2]
        for i in range(0, self.numCLUS):
            # bool matrix when the pixel position is i-th class the value is True else False
            idx = self.clusters == i
            idx_total = np.sum(idx)
            for j in range(0, color_shape):
                # find the luminance of same clustering
                cacu_np = self.mosaImgAB[:, :, j] * idx
                average = np.sum(cacu_np) / idx_total
                # do replacement
                self.mosaImgAB[idx, j] = average

    """Method2:
    Use color value of self.center (its shape is (self,numCLUS, 3), which contains information (L, x, y) of each cluster 
    center) as the average value, then replacing those pixels  which have same label (class) with the color value"""
    def Mosaica_gray(self):
        # for each cluster give values from self.center to self.mosaImg
        for i in range(0, self.numCLUS):
            # bool matrix when the pixel position is i-th class the value is True else False
            idx = self.clusters == i
            # get luminance value from self.centers
            value = self.centers[i, 0]
            # do replacement
            self.mosaImgGray[idx] = value

