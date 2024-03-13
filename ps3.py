
import math

import cv2
import numpy as np
from numpy import linalg
import scipy
from scipy import ndimage, signal


class Mouse_Click_Correspondence(object):

    def __init__(self, path1='', path2='', img1='', img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1 = self.sx1
            sy1 = self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self, event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2 = self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self, path1, path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        # path1 = r'1a_notredame.jpg'
        # path2 = r'1b_notredame.jpg'

        # path1 = self.path1
        # path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)


def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """

    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]

    return corners


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    # image_blur = cv2.GaussianBlur(image,(5,5),0)
    # image_blur = cv2.medianBlur(image,5,0)
    image_blur = image
    # template_blur = cv2.GaussianBlur(template,(5,5),0)
    # cv2.imshow('image_blur', image_blur)
    # cv2.waitKey()
    # Use cv2.matchtemplate to find markers on image - using SSD - method = 0

    # rotate template and try to find matching marker
    # the trial that found the brightest max spot will be used
    match_map_best = []
    final_template_size = []
    corr_norm_max = 0
    for i in range(50, 301, 10):
        template_resized = cv2.resize(src=template, dsize=None, fx=i / 100, fy=i / 100)

        for j in range(0, 37):
            template_resized_rotated = scipy.ndimage.rotate(template_resized, angle=5 * j, axes=(1, 0), cval=255)

            match_map = cv2.matchTemplate(image_blur, templ=template_resized_rotated, method=3)
            if np.max(match_map) > corr_norm_max:
                corr_norm_max = np.max(match_map)
                match_map_best = match_map[:]
                final_template_size = template_resized_rotated.shape

    # loop through each intensity value, from smallest to largest squared difference,
    # get the position of that pixel
    # assign the location of that pixel to 1 out of 4 spots if the pixel is far enough from all the filled position
    # until all 4 spots is filled
    # flatten_map = match_map_best.flatten('C')
    # sorted_map = np.sort(flatten_map)[::-1]
    # pixel_data = np.array()
    pixel_data = []
    for row in range(match_map_best.shape[0]):
        for col in range(match_map_best.shape[1]):
            # np.vstack(pixel_data,[match_map_best[row,col],row,col])
            pixel_data.append([match_map_best[row, col], row, col])

    sorted_pixel_data = sorted(pixel_data)
    sorted_pixel_data.reverse()
    mark1_loc = mark2_loc = mark3_loc = mark4_loc = np.array([0, 0])
    mark1_found = mark2_found = mark3_found = mark4_found = False

    distance_threshold = match_map_best.shape[0] * 0.25
    while not (mark1_found == mark2_found == mark3_found == mark4_found == True):
        current_pixel = sorted_pixel_data[0]
        # print('current_pixel',current_pixel)
        current_pixel_loc = [current_pixel[1], current_pixel[2]]
        # print('current_pixel_loc',current_pixel_loc)
        if mark1_found == False:
            mark1_loc = current_pixel_loc
            mark1_found = True
        elif mark2_found == False and euclidean_distance(tuple(current_pixel_loc),
                                                         tuple(mark1_loc)) > distance_threshold:
            mark2_loc = current_pixel_loc
            mark2_found = True
        elif mark3_found == False and euclidean_distance(tuple(current_pixel_loc),
                                                         tuple(mark1_loc)) > distance_threshold \
                and euclidean_distance(tuple(current_pixel_loc), tuple(mark2_loc)) > distance_threshold:
            mark3_loc = current_pixel_loc
            mark3_found = True
        elif mark4_found == False \
                and euclidean_distance(tuple(current_pixel_loc), tuple(mark1_loc)) > distance_threshold \
                and euclidean_distance(tuple(current_pixel_loc), tuple(mark2_loc)) > distance_threshold \
                and euclidean_distance(tuple(current_pixel_loc), tuple(mark3_loc)) > distance_threshold:
            mark4_loc = current_pixel_loc
            mark4_found = True
        sorted_pixel_data = sorted_pixel_data[1:]
    padding = int(final_template_size[0] / 2)
    # export to marker_list
    marker_list = np.array([[mark1_loc[1] + padding, mark1_loc[0] + padding],
                            [mark2_loc[1] + padding, mark2_loc[0] + padding],
                            [mark3_loc[1] + padding, mark3_loc[0] + padding],
                            [mark4_loc[1] + padding, mark4_loc[0] + padding]])

    marker_distance = np.array([euclidean_distance((0, 0), (i[0], i[1])) for i in marker_list])

    # get top left and bottom left point - assuming it has the shortest euclidean distance
    top_left_index = np.where(marker_distance == min(marker_distance))
    top_left_pos = marker_list[top_left_index].tolist()[0]
    # print('top_left_index', top_left_index)

    marker_list_m1 = np.delete(marker_list, top_left_index, 0)
    # print('marker_list_m1',marker_list_m1)

    # get the bottom left marker - assuming it's the marker with lowest x among the remaining 3
    marker_list_m1_sorted = marker_list_m1[marker_list_m1[:, 0].argsort()]
    # print('marker_list_m1_sorted',marker_list_m1_sorted)
    bottom_left_pos = marker_list_m1_sorted[0]
    # print('bottom_left_pos',bottom_left_pos)
    marker_list_m2 = np.delete(marker_list_m1_sorted, 0, 0)
    # print('marker_list_m2',marker_list_m2)

    # get the top right marker - assume it has lower y coordinate between the 2 markers

    # marker_list_m2_sorted = np.sort(marker_list_m2, axis=1)
    marker_list_m2_sorted = marker_list_m2[marker_list_m2[:, 1].argsort()]
    # print('marker_list_m2 sorted',marker_list_m2_sorted)
    top_right_pos = marker_list_m2_sorted[0]
    bottom_right_pos = marker_list_m2_sorted[1]
    out_list = [tuple(top_left_pos), tuple(bottom_left_pos), tuple(top_right_pos), tuple(bottom_right_pos)]
    # cv2.imshow('match_map', match_map)
    # cv2.waitKey()
    # print('out_list', out_list)
    return out_list


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    print('markers', markers)

    cv2.line(image, markers[0], markers[1], color=(255, 0, 0))
    cv2.line(image, markers[1], markers[3], color=(255, 0, 0))
    cv2.line(image, markers[3], markers[2], color=(255, 0, 0))
    cv2.line(image, markers[2], markers[0], color=(255, 0, 0))
    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    h = imageB.shape[0]
    w = imageB.shape[1]
    H = linalg.inv(homography)
    index_y, index_x = np.indices((h, w), dtype=np.float64)
    P = np.array([index_x.reshape(-1), index_y.reshape(-1), np.ones_like(index_x).reshape(-1)])
    res = H.dot(P)
    map_1, map_2 = res[:-1] / res[-1]
    map_1 = map_1.reshape(h, w).astype(np.float32)
    map_2 = map_2.reshape(h, w).astype(np.float32)
    res_background = imageB.copy()
    res = cv2.remap(imageA, map_1, map_2, cv2.INTER_LINEAR, dst=res_background)

    # create a white image with shape of imageA
    img_white_A = imageA.copy()
    img_white_A[:, :, :] = 255
    # project white image A similarly to image A, to create a mask
    mask_background = imageB.copy()
    res_mask = cv2.remap(img_white_A, map_1, map_2, cv2.INTER_LINEAR, dst=mask_background)

    final = imageB.copy()
    for r in range(h):
        for c in range(w):
            if sum(res_mask[r, c]) != 0:
                final[r, c] = res[r, c]
                # print('here')
            else:
                final[r, c] = imageB[r, c]

    # raise NotImplementedError
    # res = imageB.copy()
    return final

def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """

    P = np.array([
        [-srcPoints[0][0], -srcPoints[0][1], -1, 0, 0, 0, srcPoints[0][0] * dstPoints[0][0],
         srcPoints[0][1] * dstPoints[0][0], dstPoints[0][0]],
        [0, 0, 0, -srcPoints[0][0], -srcPoints[0][1], -1, srcPoints[0][0] * dstPoints[0][1],
         srcPoints[0][1] * dstPoints[0][1], dstPoints[0][1]],
        [-srcPoints[1][0], -srcPoints[1][1], -1, 0, 0, 0, srcPoints[1][0] * dstPoints[1][0],
         srcPoints[1][1] * dstPoints[1][0], dstPoints[1][0]],
        [0, 0, 0, -srcPoints[1][0], -srcPoints[1][1], -1, srcPoints[1][0] * dstPoints[1][1],
         srcPoints[1][1] * dstPoints[1][1], dstPoints[1][1]],
        [-srcPoints[2][0], -srcPoints[2][1], -1, 0, 0, 0, srcPoints[2][0] * dstPoints[2][0],
         srcPoints[2][1] * dstPoints[2][0], dstPoints[2][0]],
        [0, 0, 0, -srcPoints[2][0], -srcPoints[2][1], -1, srcPoints[2][0] * dstPoints[2][1],
         srcPoints[2][1] * dstPoints[2][1], dstPoints[2][1]],
        [-srcPoints[3][0], -srcPoints[3][1], -1, 0, 0, 0, srcPoints[3][0] * dstPoints[3][0],
         srcPoints[3][1] * dstPoints[3][0], dstPoints[3][0]],
        [0, 0, 0, -srcPoints[3][0], -srcPoints[3][1], -1, srcPoints[3][0] * dstPoints[3][1],
         srcPoints[3][1] * dstPoints[3][1], dstPoints[3][1]],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])

    C = [[0], [0], [0], [0], [0], [0], [0], [0], [1]]

    H = np.matmul(np.linalg.inv(P), C)

    # print('H',H)
    homography = np.reshape(H, (3, 3))
    # print('homography',homography)
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    cap = cv2.VideoCapture(filename)
    amount_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(amount_of_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, frame = cap.read()
        # cv2.imshow('frame', frame)
        # cv2.waitKey()
        yield frame

    # Close video (release) and yield a 'None' value. (add 2 lines)
    cap.release()
    yield None


class Automatic_Corner_Detection(object):

    def __init__(self):
        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)

    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''
        SOBEL_X = np.flip(self.SOBEL_X)
        SOBEL_Y = np.flip(self.SOBEL_Y)
        Ix = signal.convolve2d(image_bw, SOBEL_X,mode = 'same',boundary = 'fill', fillvalue=0.)
        Iy = signal.convolve2d(image_bw, SOBEL_Y,mode = 'same',boundary = 'fill', fillvalue=0. )

        # Ix = signal.correlate2d(image_bw, self.SOBEL_X, mode='same',boundary = 'fill', fillvalue=0.)
        # Iy = signal.correlate2d(image_bw, self.SOBEL_Y, mode='same',boundary = 'fill', fillvalue=0.)
        return Ix, Iy

    def gkernel(self, ksize, sigma):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """

        Ix, Iy = self.gradients(image_bw)
        # Gk = cv2.getGaussianKernel2d(ksize = ksize, sigma = sigma)
        Gk = self.gkernel( ksize, sigma)
        print('Gk', Gk)
        # Gk_T = np.transpose(Gk)

        sx2 = signal.convolve2d(np.square(Ix), Gk,mode = 'same',boundary = 'fill', fillvalue=0.)
        sxsy = signal.convolve2d(Ix * Iy, Gk,mode = 'same',boundary = 'fill', fillvalue=0.)
        sy2 = signal.convolve2d(np.square(Iy), Gk,mode = 'same',boundary = 'fill', fillvalue=0.)

        # sx2 = signal.correlate2d(Ix ** 2, Gk,mode = 'same',boundary = 'fill', fillvalue=0.)
        # sxsy = signal.correlate2d(Iy * Ix, Gk,mode = 'same',boundary = 'fill', fillvalue=0.)
        # sy2 = signal.correlate2d(Iy ** 2, Gk,mode = 'same',boundary = 'fill', fillvalue=0.)

        return sx2, sy2, sxsy

    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """
        sx2, sy2, sxsy = self.second_moments(image_bw, ksize, sigma)
        det_M = sx2 * sy2 - sxsy ** 2
        trace_M = sx2 + sy2
        R = det_M - alpha * (trace_M ** 2)
        R_min = np.min(R)
        R_max = np.max(R)
        R_norm = (R - R_min) / (R_max - R_min)
        return R_norm

    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """
        median = np.median(R)
        median_cutoff = np.where( R <= median, 0, R)
        max_filter = ndimage.maximum_filter(median_cutoff, size = ksize, mode = 'constant', cval = 0.0)
        binary_map = np.where(R == max_filter,1,0)
        nms = R * binary_map
        nms_sorted_flatten = np.sort(nms.reshape(-1))[::-1]
        k_val = nms_sorted_flatten[k]
        loc_map = np.where(nms > k_val)
        x = loc_map[1]
        y = loc_map[0]
        return x, y

    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """
        R = self.harris_response_map(image_bw, sigma=5, alpha=0.05)
        x1, y1 = self.nms_maxpool( R, k, ksize=7)
        return x1, y1


class Image_Mosaic(object):

    def __int__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''

        h = im_src.shape[0]
        w = im_src.shape[1]
        H = np.linalg.inv(H)
        index_y, index_x = np.indices((h, w), dtype=np.float64)
        P = np.array([index_x.reshape(-1), index_y.reshape(-1), np.ones_like(index_x).reshape(-1)])
        res = H.dot(P)
        map_1, map_2 = res[:-1] / res[-1]
        map_1 = map_1.reshape(h, w).astype(np.float32)
        map_2 = map_2.reshape(h, w).astype(np.float32)
        # res_background = im_src.copy()
        im_src_black = im_src.copy()
        im_src_black[:, :, :] = 0
        res_background = np.hstack((im_src, im_src_black))

        warped_img = cv2.remap(im_dst, map_1, map_2, cv2.INTER_LINEAR, dst=res_background)
        # warped_img = project_imageA_onto_imageB(im_dst,im_src,H)
        # cv2.imshow('res_background',res_background)
        # cv2.imshow('warped_img',warped_img)
        # cv2.waitKey()
        return warped_img

    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''

        im_mos_out = img_src.copy()
        for r in range(im_mos_out.shape[0]):
            for c in range(im_mos_out.shape[1]):
                if sum(img_warped[r, c]) == 0:
                    im_mos_out[r, c] = img_src[r, c]
                else:
                    im_mos_out[r, c] = img_src[r, c] // 2 + img_warped[r, c] // 2
        # cv2.imshow('im_mos_out',im_mos_out)
        # cv2.waitKey()

        return im_mos_out