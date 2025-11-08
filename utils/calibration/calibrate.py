#!/usr/bin/env python

'''
ADAPTED FROM OPENCV SAMPLES
https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py
SAMPLE CALL:
python calibrate.py --debug ./calibration_output -w 6 -h 8 -t chessboard --square_size=35 ./calibration_pictures/frame*.jpg


SEE INSTRUCTIONS FROM OPENCV:
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [-w <width>] [-h <height>] [-t <pattern type>] [--square_size=<square size>] [<image mask>]

usage example:
    calibrate.py -w 4 -h 6 -t chessboard --square_size=50 ../data/left*.jpg

default values:
    --debug:    ./output/
    -w: 4
    -h: 6
    -t: chessboard
    --square_size: 10
    --marker_size: 5
    --threads: 4

NOTE: Chessboard size is defined in inner corners. Charuco board size is defined in units, and has been removed from this sample.


For this calibration on our cameras use:
python calibrate.py -w 6 -h 4 -t chessboard --square_size=8 ./calibration_pictures/frame*.jpg
'''

import numpy as np
import cv2 as cv
import json
import os

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def main():
    import sys
    import getopt

    args, img_names = getopt.getopt(sys.argv[1:], 'w:h:t:', ['debug=','square_size=', 'threads=', ])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('-w', 4)
    args.setdefault('-h', 6)
    args.setdefault('-t', 'chessboard')
    args.setdefault('--square_size', 10)
    args.setdefault('--threads', 4)

    assert img_names, 'Did you provide a path for images?'

    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    grid_height = int(args.get('-h'))
    grid_width = int(args.get('-w'))
    pattern_type = str(args.get('-t'))
    square_size = float(args.get('--square_size'))
    
    # Determine how big the checkerboard is (we assume chessboard only for this sample)
    pattern_size = (grid_width, grid_height)
    if pattern_type == 'chessboard':
        # compute total amount of squares, and make np array for each point in 3D corresponding to each corner. num_corners = num_squares -1
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) # assign x,y grid coordinates 
        pattern_points *= square_size # convert to "metric" space using square size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found = False
        corners = 0
        if pattern_type == 'chessboard':

            # Find the chess board corners (will try to find prod(pattern_size) corners) --> make sure u have this amount available on ur checkerboard
            found, corners = cv.findChessboardCorners(img, pattern_size)

            # If found ALL corners, refine the corners positions
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1) # terimination criteria for cornerSubPix
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                frame_img_points = corners.reshape(-1, 2) # 2d image coordinate
                frame_obj_points = pattern_points # corresponding 3d world coordinate. assume the order is the same for image coordinates and pre computed 3d coordinate (top left to bottom right) 
        else:
            print("unknown pattern type", pattern_type)
            return None

        # Draw and display the detected corner for debugging
        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if pattern_type == 'chessboard':
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_board.png')
            cv.imwrite(outfile, vis)

        # no corners found, skip image
        if not found:
            print('pattern not found')
            return None

        # at this point we have found the corners in image space + attached corresponding 3d coordinates
        print('           %s... OK' % fn)
        return (frame_img_points, frame_obj_points)
    

    # for each image, find the corner locations in image coordinates (in parallel if threads > 1)
    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)



    # now we have all the corner locations in image space in chessboards + attached corresponding 3d coordinates
    chessboards = [x for x in chessboards if x is not None] # filter out failed images


    for idx, (corners, pattern_points) in enumerate(chessboards):
        if len(corners) < 4:
            print("Not enough obj/img points for %d, skipping image!" % idx)
        else:
            img_points.append(corners)
            obj_points.append(pattern_points)
    
    # think: 3D points (Fixed) --> camera model parameter (intrinsic + extinsic + distortion coefficents) --> 2D points --> compare to GT using L2 loss and then use gradient descent to optimize the camera model parameters
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None) # determine the camera intrisnics and extrinsics that best map the 2d image points from the 3d object points
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h)) # undistort camera matrix and give us the "region of interest" / areas where there are valid pixels after undistortion
    # rvecs and tvecs are the extrinsic parameters for each image (where the chessboard is located in world space relative to the camera)
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    print("newcameramtx:\n", newcameramtx)
    print("roi: ", roi)

    # save the camera calibration. Note: this doesnt account for distortion
    data = {
    "RMS": rms,
    "CameraMatrix": camera_matrix.tolist(),
    "DistortionCoefficients": dist_coefs.ravel().tolist(),
    "NewCameraMatrix": newcameramtx.tolist(), # undistorted camera matrix (you dont have to use this if u use cv.undistort with original camera matrix + dist coefs)
    "ROI": roi
    }

    # 
    input_dirname = img_names[0].split('/')[1].split('.')[0]
    if input_dirname == 'transform_output':
        file_path = "check_intrinsics.json"
    else:
        file_path = "intrinsics.json"

    with open(file_path, "w") as json_file:
        json.dump(data, json_file)

    
    # evaluate undistortion on each image and save the result
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        # img_found = os.path.join(debug_dir, name + '_board.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(fn)
        if img is None:
            continue

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)
    print('Done')


# note: The better our intrinsics --> the better the mapping to Kinect inrinsics later --> the better the overall 3D reconstruction from depth model
# note: more images = longer time since its runninig optimization over all images
if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

