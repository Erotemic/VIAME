# -*- coding: utf-8 -*-
"""
Reimplementation of matlab algorithms for fishlength detection in python.
This is the first step of moving them into kwiver.

# Adjust the threshold value for left (thL) and right (thR) so code will
# select most fish without including non-fish objects (e.g. the net)
thL = 20
thR = 20

# Species classes
# This is fixed for each training set, so it will remain the same throughout an entire survey
# pollock, salmon unident., rockfish unident.
sp_numbs = [21740, 23202, 30040]


# number to increment between frames
by_n = 1

# Factor to reduce the size of the image for processing
factor = 2
"""
from __future__ import division, print_function
from collections import namedtuple
import cv2
import itertools as it
import numpy as np
import scipy.optimize
from imutils import (
    imscale, ensure_grayscale, overlay_heatmask, from_homog, to_homog)
from os.path import expanduser, basename, join

OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'angle'))


class FishDetector(object):
    """
    Uses background subtraction and 4-way connected compoments algorithm to
    detect potential fish objects. Objects are filtered by size, aspect ratio,
    and closeness to the image border to remove bad detections.

    References:
        https://stackoverflow.com/questions/37300698/gaussian-mixture-model
        http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html

    """
    def __init__(self, **kwargs):
        bg_algo = kwargs.get('bg_algo', 'custom')

        self.config = {
            # limits accepable targets to be within this region [padx, pady]
            'edge_limit': [12, 12],
            # Maximum aspect ratio for filtering out non-fish objects
            'max_aspect': 7.5,
            # Minimum aspect ratio for filtering out non-fish objects
            'min_aspect': 3.5,

            # minimum number of pixels to keep a section, after sections
            # are found by component function
            'bg_algo': bg_algo,
        }

        # Different default params depending on the background subtraction algo
        if self.config['bg_algo'] == 'custom':
            self.config.update({
                'factor': 4.0,
                'min_size': 50,
                'diff_thresh': 19,
            })
        else:
            self.config.update({
                'factor': 2.0,
                'min_size': 2000,
                'n_training_frames': 30,
                'gmm_thresh': 30,
            })

        self.config.update(kwargs)

        # Choose which algo to use for background subtraction
        if self.config['bg_algo'] == 'custom':
            self.background_model = CustomBackgroundSubtractor(
                diff_thresh=self.config['diff_thresh'],
            )
        elif self.config['bg_algo'] == 'gmm':
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=self.config['n_training_frames'],
                varThreshold=self.config['gmm_thresh'],
                detectShadows=False)
        # self.background_model = cv2.createBackgroundSubtractorKNN(
        #     history=self.config['n_training_frames'],
        #     dist2Threshold=50 ** 2,
        #     detectShadows=False
        # )

    def postprocess_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # opening is erosion followed by dilation
        mask = cv2.erode(src=mask, kernel=kernel, dst=mask)
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        # do another dilation
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        return mask

    def upscale_detections(self, detections, upfactor):
        """
        inplace upscaling of bounding boxes and points
        (masks are not upscaled)
        """
        for detection in detections:
            center = upfactor * detection['oriented_bbox'].center
            extent = upfactor * detection['oriented_bbox'].extent
            angle = detection['oriented_bbox'].angle
            detection['oriented_bbox'] = OrientedBBox(
                tuple(center), tuple(extent), angle)
            detection['hull'] = upfactor * detection['hull']
            detection['box_points'] = upfactor * detection['box_points']

    def apply(self, img):
        """
        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> self, img = demodata(target_step='detect', target_frame_num=7)
            >>> detections, masks = self.apply(img)
            >>> print('detections = {!r}'.format(detections))
            >>> self.draw_detections(img, detections, masks)
        """
        # Convert to grayscale
        img_ = ensure_grayscale(img)
        # Downsample image before running detection

        if self.config['bg_algo'] == 'gmm':
            if self.config['factor'] != 1.0:
                downfactor_ = 1 / self.config['factor']
                img_, downfactor = imscale(img, downfactor_)
                upfactor = 1 / np.array(downfactor)
            else:
                img_ = img_
        else:
            # custom takes care of downsample
            upfactor = np.array([self.config['factor']] * 2)

        # Run detection / update background model
        orig_mask = self.background_model.apply(img_)

        # Remove noise
        if self.config['bg_algo'] == 'gmm':
            post_mask = self.postprocess_mask(orig_mask.copy())
        else:
            post_mask = orig_mask

        # Find detections using CC algorithm
        detections = list(self.masked_detect(post_mask))

        if self.config['factor'] != 1.0:
            # Upscale back to input img coordinates (to agree with camera calib)
            self.upscale_detections(detections, upfactor)

        masks = {
            'orig': orig_mask,
            'post': post_mask,
        }
        return detections, masks

    def masked_detect(self, mask):
        """
        Find pixel locs of each cc and determine if its a valid detection
        """
        img_h, img_w = mask.shape

        # 4-way connected compoment algorithm
        n_ccs, cc_mask = cv2.connectedComponents(mask, connectivity=4)

        # Define thresholds to filter edges
        minx_lim, miny_lim = self.config['edge_limit']
        maxx_lim = img_w - minx_lim
        maxy_lim = img_h - miny_lim

        # Filter ccs to generate only "good" detections
        for cc_label in range(1, n_ccs):
            cc = (cc_mask == cc_label)
            # note, `np.where` returns coords in (r, c)
            cc_y, cc_x = np.where(cc)

            # Remove small regions
            n_pixels = len(cc_x)
            if n_pixels < self.config['min_size']:
                continue

            # Filter objects detected on the edge of the image region
            minx, maxx = cc_x.min(), cc_x.max()
            miny, maxy = cc_y.min(), cc_y.max()
            if any([minx < minx_lim, maxx > maxx_lim,
                    miny < miny_lim, maxy > maxy_lim]):
                continue

            # generate the valid detection
            points = np.vstack([cc_x, cc_y]).T

            # Find a minimum oriented bounding box around the points
            hull = cv2.convexHull(points)
            oriented_bbox = OrientedBBox(*cv2.minAreaRect(hull))
            w, h = oriented_bbox.extent

            # Filter objects without fishy aspect ratios
            ar = max([(w / h), (h / w)])
            if any([ar < self.config['min_aspect'],
                    ar > self.config['max_aspect']]):
                continue

            detection = {
                # 'points': points,
                'box_points': cv2.boxPoints(oriented_bbox),
                'oriented_bbox': oriented_bbox,
                'cc': cc,
                'hull': hull[:, 0, :],
            }
            yield detection


class CustomBackgroundSubtractor(object):
    """
    custom algorithm for subtracting background
    """

    def __init__(self, diff_thresh=19):
        self.diff_thresh = diff_thresh
        self.bgimg = None

    def apply(self, img):
        """

        Debug:
            import sys
            sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            from gmm_online_background_subtraction import *

            image_path_list1, _, _ = demodata_input(dataset=2)

            import plottool as pt
            pt.qtensure()
            self = CustomBackgroundSubtractor()
            for i in [0, 3, 6, 9]:
                img1 = cv2.imread(image_path_list1[0 + i])
                img2 = cv2.imread(image_path_list1[1 + i])
                img3 = cv2.imread(image_path_list1[2 + i])
                mask1, bg1 = self.apply(img1), self.bgimg.copy()
                mask2, bg2 = self.apply(img2), self.bgimg.copy()
                mask3, bg3 = self.apply(img3), self.bgimg.copy()

                pt.imshow(img1, fnum=i, pnum=(3, 3, 1))
                pt.imshow(img2, fnum=i, pnum=(3, 3, 2))
                pt.imshow(img3, fnum=i, pnum=(3, 3, 3))
                pt.imshow(mask1, fnum=i, pnum=(3, 3, 4))
                pt.imshow(mask2, fnum=i, pnum=(3, 3, 5))
                pt.imshow(mask3, fnum=i, pnum=(3, 3, 6))
                pt.imshow(bg1, fnum=i, pnum=(3, 3, 7))
                pt.imshow(bg2, fnum=i, pnum=(3, 3, 8))
                pt.imshow(bg3, fnum=i, pnum=(3, 3, 9))
        """
        img_ = ensure_grayscale(img)

        def downsample_average_blocks(img, factor):
            """
            Funky way of downsampling. Averages blocks of pixels.
            Equivalent to a strided 2D convolution with a uniform matrix
            Unfortunately scipy doesn't seem to have a strided implementation
            """
            dsize = tuple(np.divide(img_.shape, factor).astype(np.int)[0:2])
            temp_img = np.zeros(dsize)
            for r, c in it.product(range(factor), range(factor)):
                temp_img += img_[r::factor, c::factor]
            new_img = temp_img / (factor ** 2)
            return new_img

        factor = 4
        new_img = downsample_average_blocks(img, factor)

        # Subtract the previous background image and make a new one
        if self.bgimg is None:
            self.bgimg = new_img
            mask = np.zeros(img_.shape, dtype=np.uint8)
        else:
            fr_diff = new_img - self.bgimg
            mask = fr_diff > self.diff_thresh

            # This seems to put black pixels always in the background.
            fg_mask = (fr_diff > self.diff_thresh)
            fg_img = (fg_mask * new_img)  # this is background substracted image
            mask = (fg_img > 0).astype(np.uint8) * 255

            # update the background image
            # (not sure what the intuition is behind this particular update)
            # My guess: slowly make pixels more likely to be fg, unless they
            # were background recently?
            self.bgimg -= 1
            self.bgimg[fr_diff > 1] += 2
        return mask


class StereoCalibration(object):
    """
    Helper class for reading / accessing stereo camera calibration params
    """
    def __init__(cal):
        cal.data = None
        cal.unit = 'milimeters'

    def __str__(cal):
        return '{}({})'.format(cal.__class__.__name__, cal.data)

    def extrinsic_vecs(cal):
        rvec1 = cal.data['left']['extrinsic']['om']
        tvec1 = cal.data['right']['extrinsic']['om']

        rvec2 = cal.data['right']['extrinsic']['om']
        tvec2 = cal.data['right']['extrinsic']['T']
        return rvec1, tvec1, rvec2, tvec2

    def distortions(cal):
        kc1 = cal.data['right']['intrinsic']['kc']
        kc2 = cal.data['left']['intrinsic']['kc']
        return kc1, kc2

    def intrinsic_matrices(cal):
        K1 = cal._make_intrinsic_matrix(cal.data['left']['intrinsic'])
        K2 = cal._make_intrinsic_matrix(cal.data['right']['intrinsic'])
        return K1, K2

    def _make_intrinsic_matrix(cal, intrin):
        fc = intrin['fc']
        cc = intrin['cc']
        alpha_c = intrin['alpha_c']
        KK = np.array([
            [fc[0], alpha_c * fc[0], cc[0]],
            [    0,           fc[1], cc[1]],
            [    0,               0,     1],
        ])
        return KK

    @classmethod
    def from_matfile(StereoCalibration, cal_fpath):
        """
        Loads a matlab camera calibration file from disk

        References:
            http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
            http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html

        Doctest:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> _, _, cal_fpath = demodata_input(dataset=1)
            >>> cal = StereoCalibration.from_matfile(cal_fpath)
            >>> print('cal = {}'.format(cal))
            >>> _, _, cal_fpath = demodata_input(dataset=2)
            >>> cal = StereoCalibration.from_matfile(cal_fpath)
            >>> print('cal = {}'.format(cal))
        """
        import scipy.io
        cal_data = scipy.io.loadmat(cal_fpath)
        keys = ['om', 'T', 'fc_left', 'fc_right', 'cc_left', 'cc_right',
                'kc_left', 'kc_right', 'alpha_c_left', 'alpha_c_right']

        if isinstance(cal_data, dict) and 'Cal' in cal_data:
            vals = cal_data['Cal'][0][0]
            flat_dict = {k: v.ravel() for k, v in zip(keys, vals)}
        else:
            flat_dict = {key: cal_data[key].ravel() for key in keys}

        data = {
            'left': {
                'extrinsic': {
                    # Center wold on the left camera
                    'om': np.zeros(3),  # rotation vector
                    'T': np.zeros(3),  # translation vector
                },
                'intrinsic': {
                    'fc': flat_dict['fc_left'],  # focal point
                    'cc': flat_dict['cc_left'],  # principle point
                    'alpha_c': flat_dict['alpha_c_left'][0],  # skew
                    'kc': flat_dict['kc_left'],  # distortion
                }
            },

            'right': {
                'extrinsic': {
                    'om': flat_dict['om'],  # rotation vector
                    'T': flat_dict['T'],  # translation vector
                },
                'intrinsic': {
                    'fc': flat_dict['fc_right'],  # focal point
                    'cc': flat_dict['cc_right'],  # principle point
                    'alpha_c': flat_dict['alpha_c_right'][0],  # skew
                    'kc': flat_dict['kc_right'],  # distortion
                }
            },
        }
        cal = StereoCalibration()
        cal.data = data
        return cal


class FishStereoTriangulationAssignment(object):
    def __init__(self):
        self.config = {
            # Threshold for errors between before & after projected
            # points to make matches between left and right
            # 'max_err': [6, 14],
            'max_err': [300, 300],
            'small_len': 15,  # in centimeters
        }

    def triangulate(self, cal, det1, det2):
        """
        Assuming, det1 matches det2, we determine 3d-coordinates of each
        detection and measure the reprojection error.

        References:
            http://answers.opencv.org/question/117141
            https://gist.github.com/royshil/7087bc2560c581d443bc
            https://stackoverflow.com/a/29820184/887074
        """
        _debug = False
        if _debug:
            # Use 4 corners and center to ensure matrix math is good
            # (hard to debug when ndims == npts, so make npts >> ndims)
            pts1 = np.vstack([det1['box_points'], det1['oriented_bbox'].center])
            pts2 = np.vstack([det2['box_points'], det2['oriented_bbox'].center])
        else:
            # Use only the corners of the bbox
            pts1 = det1['box_points'][[0, 2]]
            pts2 = det2['box_points'][[0, 2]]

        # Move into opencv point format (num x 1 x dim)
        pts1_cv = pts1[:, None, :]
        pts2_cv = pts2[:, None, :]

        # Grab camera parameters
        K1, K2 = cal.intrinsic_matrices()
        kc1, kc2 = cal.distortions()
        rvec1, tvec1, rvec2, tvec2 = cal.extrinsic_vecs()

        # Make extrincic matrices
        R1 = cv2.Rodrigues(rvec1)[0]
        R2 = cv2.Rodrigues(rvec2)[0]
        T1 = tvec1[:, None]
        T2 = tvec2[:, None]
        RT1 = np.hstack([R1, T1])
        RT2 = np.hstack([R2, T2])

        # Undistort points
        # This puts points in "normalized camera coordinates" making them
        # independent of the intrinsic parameters. Moving to world coordinates
        # can now be done using only the RT transform.
        unpts1_cv = cv2.undistortPoints(pts1_cv, K1, kc1)
        unpts2_cv = cv2.undistortPoints(pts2_cv, K2, kc2)

        # note: trinagulatePoints docs say that it wants a 3x4 projection
        # matrix (ie K.dot(RT)), but we only need to use the RT extrinsic
        # matrix because the undistorted points already account for the K
        # intrinsic matrix.
        world_pts_homog = cv2.triangulatePoints(RT1, RT2, unpts1_cv, unpts2_cv)
        world_pts = from_homog(world_pts_homog)

        # Compute distance between 3D bounding box points
        if _debug:
            corner1, corner2 = world_pts.T[[0, 2]]
        else:
            corner1, corner2 = world_pts.T

        # Convert to centimeters
        fishlen = np.linalg.norm(corner1 - corner2) / 10

        # Reproject points
        world_pts_cv = world_pts.T[:, None, :]
        proj_pts1_cv = cv2.projectPoints(world_pts_cv, rvec1, tvec1, K1, kc1)[0]
        proj_pts2_cv = cv2.projectPoints(world_pts_cv, rvec2, tvec2, K2, kc2)[0]

        # Check error
        err1 = ((proj_pts1_cv - pts1_cv)[:, 0, :] ** 2).sum(axis=1)
        err2 = ((proj_pts2_cv - pts2_cv)[:, 0, :] ** 2).sum(axis=1)
        errors = np.hstack([err1, err2])

        # Get 3d points in each camera's reference frame
        # Note RT1 is the identity and RT are 3x4, so no need for `from_homog`
        # Return points in with shape (N,3)
        pts1_3d = RT1.dot(to_homog(world_pts)).T
        pts2_3d = RT2.dot(to_homog(world_pts)).T
        return pts1_3d, pts2_3d, errors, fishlen

    def minimum_weight_assignment(self, assign_errors):
        """
        Finds optimal assignment of left-camera to right-camera detections

        Example:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> self = FishStereoTriangulationAssignment()
            >>> assign_errors = np.array([
            >>>     [9, 2, 1, 9],
            >>>     [4, 1, 5, 5],
            >>>     [9, 9, 2, 4],
            >>> ])
            >>> assign1 = self.minimum_weight_assignment(assign_errors)
            >>> assign2 = self.minimum_weight_assignment(assign_errors.T)
        """
        n1, n2 = assign_errors.shape
        n = max(n1, n2)
        # Embed the [n1 x n2] matrix in a padded (with inf) [n x n] matrix
        cost_matrix = np.full((n, n), fill_value=np.inf)
        cost_matrix[0:n1, 0:n2] = assign_errors

        # Find an effective infinite value for infeasible assignments
        is_infeasible = np.isinf(cost_matrix)
        is_positive = cost_matrix > 0
        feasible_vals = cost_matrix[~(is_infeasible & is_positive)]
        large_val = (n + feasible_vals.sum()) * 2
        # replace infinite values with effective infinite values
        cost_matrix[is_infeasible] = large_val

        # Solve munkres problem for minimum weight assignment
        indexes = list(zip(*scipy.optimize.linear_sum_assignment(cost_matrix)))
        # Return only the feasible assignments
        assignment = [(i, j) for (i, j) in indexes
                      if cost_matrix[i, j] < large_val]
        return assignment

    def find_matches(self, cal, detections1, detections2):
        """
        Match detections from the left camera to detections in the right camera

        Example:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> detections1, detections2, cal = demodata('triangulate')
            >>> self = FishStereoTriangulationAssignment()
            >>> assignment, assign_data = self.find_matches(cal, detections1, detections2)
        """
        n_detect1, n_detect2 = len(detections1), len(detections2)
        assign_world_pts = {}
        assign_fishlen = {}

        # Initialize matrix of reprojection errors
        assign_errors = np.full((n_detect1, n_detect2), fill_value=np.inf)

        # Find the liklihood that each pair of detections matches by
        # triangulating and then measuring the reprojection error.
        for (i, det1), (j, det2) in it.product(enumerate(detections1),
                                               enumerate(detections2)):

            # Triangulate assuming det1 and det2 match, but return the
            # reprojection error so we can check if this assumption holds
            pts1_3d, pts2_3d, errors, fishlen = self.triangulate(cal, det1, det2)

            # Check chirality
            # Both Z-coordinates must be positive (i.e. in front the cameras)
            z_coords1 = pts1_3d.T[2]
            z_coords2 = pts2_3d.T[2]
            both_in_front = np.all(z_coords1 > 0) and np.all(z_coords2 > 0)
            if not both_in_front:
                # Ignore out-of-view correspondences
                continue

            # Check if reprojection error is too high
            max_error = self.config['max_err']
            small_len = self.config['small_len']  # hardcoded to 15cm in matlab version
            if len(max_error) == 2:
                error_thresh = max_error[0] if fishlen <= small_len else max_error[1]
            else:
                error_thresh = max_error[0]

            error = errors.mean()
            if error  >= error_thresh:
                # Ignore correspondences with high reprojection error
                continue

            # Mark the pair (i, j) as a potential candidate match
            assign_world_pts[(i, j)] = pts1_3d
            assign_errors[i, j] = error
            assign_fishlen[(i, j)] = fishlen

        # Find the matching with minimum reprojection error, such that each
        # detection in one camera can match at most one detection in the other.
        assignment = self.minimum_weight_assignment(assign_errors)

        # get associated data with each assignment
        assign_data = [
            {
                'fishlen': assign_fishlen[(i, j)],
                'error': assign_errors[(i, j)],
            }
            for i, j in assignment
        ]
        return assignment, assign_data


class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    def draw_detections(img, detections, masks):
        # Upscale masks to original image size
        dsize = tuple(img.shape[0:2][::-1])
        orig_mask = cv2.resize(masks['orig'], dsize)
        post_mask = cv2.resize(masks['post'], dsize)

        # Create a heatmap for detections
        draw_mask = np.zeros(post_mask.shape[0:2], dtype=np.float)
        draw_mask[orig_mask > 0] = .2
        draw_mask[post_mask > 0] = .75
        for n, detection in enumerate(detections, start=1):
            cc = cv2.resize(detection['cc'].astype(np.uint8), dsize)
            draw_mask[cc > 0] = 1.0
        draw_img = overlay_heatmask(img, draw_mask, alpha=.7)

        # Draw bounding boxes and contours
        for detection in detections:
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = np.round(detection['box_points']).astype(np.int)
            hull_points = np.round(detection['hull']).astype(np.int)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[hull_points], contourIdx=-1,
                color=(255, 0, 0), thickness=2)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[box_points], contourIdx=-1,
                color=(0, 255, 0), thickness=2)
        return draw_img

    @staticmethod
    def draw_stereo_detections(img1, detections1, masks1,
                               img2, detections2, masks2,
                               assignment=None, assign_data=None):
        BGR_RED = (0, 0, 255)
        line_color = BGR_RED
        text_color = BGR_RED

        draw1 = DrawHelper.draw_detections(img1, detections1, masks1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2)
        stacked = np.hstack([draw1, draw2])

        def putMultiLineText(img, text, org, **kwargs):
            """
            References:
                https://stackoverflow.com/questions/27647424/
            """
            getsize_kw = {
                k: kwargs[k]
                for k in ['fontFace', 'fontScale', 'thickness']
                if k in kwargs
            }
            x0, y0 = org
            ypad = kwargs.get('thickness', 2) + 4
            y = y0
            for i, line in enumerate(text.split('\n')):
                (w, h), text_sz = cv2.getTextSize(text, **getsize_kw)
                img = cv2.putText(img, line, (x0, y), **kwargs)
                y += (h + ypad)
            return img

        if assignment:
            for (i, j), info in zip(assignment, assign_data):

                center1 = np.array(detections1[i]['oriented_bbox'].center)
                center2 = np.array(detections2[j]['oriented_bbox'].center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_, color=line_color,
                                   lineType=cv2.LINE_AA,
                                   thickness=2)

                import textwrap
                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}cm
                    error = {error:.2f}
                    '''
                ).strip().format(**info)

                stacked = putMultiLineText(stacked, text, org=center1,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=text_color,
                                           thickness=2, lineType=cv2.LINE_AA)

        with_orig = False
        if with_orig:
            # Put in the original images
            bottom = np.hstack([img1, img2])
            stacked = np.vstack([stacked, bottom])

        return stacked


# -----------------
# Testing functions
# -----------------


class FrameStream(object):
    """
    Helper for iterating through a sequence of image frames
    """
    def __init__(stream, image_path_list, stride=1):
        stream.image_path_list = image_path_list
        stream.stride = stride
        stream.length = len(image_path_list)

    def __len__(stream):
        return stream.length

    def __getitem__(stream, index):
        img_fpath = stream.image_path_list[index]
        frame_id = basename(img_fpath).split('_')[0]
        img = cv2.imread(img_fpath)
        return frame_id, img

    def __iter__(stream):
        for i in range(0, len(stream), stream.stride):
            yield stream[i]


def demodata_input(dataset=1):
    import glob

    if dataset == 1:
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 2:
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        # cal_fpath = join(data_fpath, 'code/Calib_Results_stereo_1608.mat')
        cal_fpath = join(data_fpath, 'code/cal_201608.mat')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    else:
        assert False, 'bad dataset'

    image_path_list1 = sorted(glob.glob(join(img_path1, '*.jpg')))
    image_path_list2 = sorted(glob.glob(join(img_path2, '*.jpg')))
    assert len(image_path_list1) == len(image_path_list2)
    return image_path_list1, image_path_list2, cal_fpath


def demodata(dataset=1, target_step='detect', target_frame_num=7):
    """
    Helper for doctests. Gets test data at different points in the pipeline.
    """
    if 'target_step' not in vars():
        target_step = 'detect'
    if 'target_frame_num' not in vars():
        target_frame_num = 7
    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    cal = StereoCalibration.from_matfile(cal_fpath)

    detector1 = FishDetector()
    detector2 = FishDetector()
    for frame_num, (img_fpath1, img_fpath2) in enumerate(zip(image_path_list1,
                                                             image_path_list2)):

        frame_id1 = basename(img_fpath1).split('_')[0]
        frame_id2 = basename(img_fpath2).split('_')[0]
        assert frame_id1 == frame_id2
        frame_id = frame_id1
        img1 = cv2.imread(img_fpath1)
        img2 = cv2.imread(img_fpath2)

        if frame_num == target_frame_num:
            if target_step == 'detect':
                return detector1, img1

        detections1, masks1 = detector1.apply(img1)
        detections2, masks2 = detector2.apply(img2)

        n_detect1, n_detect2 = len(detections1), len(detections2)
        print('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            # import vtool as vt
            import ubelt as ub
            # stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)[0]
            stacked = np.hstack([masks1['draw'], masks2['draw']])
            dpath = ub.ensuredir('out')
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_num), stacked)
            cv2.imwrite(dpath + '/mask{}_{}_draw.png'.format(frame_id, frame_num), stacked)
            # return detections1, detections2
            break

    return detections1, detections2, cal


def demo():
    import ubelt as ub
    dataset = 1
    dataset = 2

    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)
    cal = StereoCalibration.from_matfile(cal_fpath)

    dpath = ub.ensuredir('out')

    stride = 2
    stream1 = FrameStream(image_path_list1, stride=stride)
    stream2 = FrameStream(image_path_list2, stride=stride)

    detector1 = FishDetector(diff_thresh=19, bg_algo='custom')
    detector2 = FishDetector(diff_thresh=15, bg_algo='custom')

    _iter = enumerate(zip(stream1, stream2))
    for frame_num, ((frame_id1, img1), (frame_id2, img2)) in _iter:
        assert frame_id1 == frame_id2
        frame_id = frame_id1
        print('frame_id = {!r}'.format(frame_id))
        print('frame_num = {!r}'.format(frame_num))

        detections1, masks1 = detector1.apply(img1)
        detections2, masks2 = detector2.apply(img2)

        # stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)[0]
        if len(detections1) > 0 or len(detections2) > 0:
            self = FishStereoTriangulationAssignment()
            assignment, assign_data = self.find_matches(cal, detections1, detections2)
        else:
            assignment, assign_data = None, None

        # if assignment:
        stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                    img2, detections2, masks2,
                                                    assignment, assign_data)
        cv2.imwrite(dpath + '/mask{}_{}_draw.png'.format(frame_id, frame_num),
                    stacked)
        # if frame_num == 7:
        #     break

if __name__ == '__main__':
    demo()
