from PIL import Image
import cv2
import numpy as np
import random

def show_img(img, name=""):
    cv2.imshow(name, img)
    cv2.waitKey()  # waits for a key press; cv2.waitKey(1) waits only 1 ms
    cv2.destroyAllWindows()
    
def coordinates(point):
    return [int(i) for i in tuple(point.ravel())]

def draw(img, corners, imgpts):
    # WARNING: OpenCV uses BGR color space
    corner = coordinates(corners[0].ravel())
    img = cv2.line(img, corner, coordinates(imgpts[0]), (0,0,255), 5)
    img = cv2.line(img, corner, coordinates(imgpts[1]), (0,255,0), 5)
    img = cv2.line(img, corner, coordinates(imgpts[2]), (255,0,0), 5)
    return img
    
#  not_important, R, t = solvePnP_with_RANSAC(marker_points, corners2, newcameramtx, 0, False, cv2.SOLVEPNP_IPPE)
def solvePnP_with_RANSAC(objpoints, imgpoints, camera_matrix, threshold):
    bestCount = 0
    for k in range(10000):
        sample_idx = np.random.choice(len(objpoints), 6, replace=False)
        success, rvec, tvec = cv2.solvePnP(objpoints[sample_idx], imgpoints[sample_idx], camera_matrix, 0, False, cv2.SOLVEPNP_IPPE)
        if not success:
            continue
                
        projected, _ = cv2.projectPoints(
            objpoints, rvec, tvec, camera_matrix, None
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - imgpoints, axis=1
        )
        inliers = np.flatnonzero(errors < threshold)

        if len(inliers) > bestCount:
            bestCount = len(inliers)
            print(inliers)
            if len(inliers)>=6:
                bestFit = cv2.solvePnP(objpoints[inliers], imgpoints[inliers], camera_matrix, 0, False, cv2.SOLVEPNP_IPPE)
            else:
                bestFit = (success, rvec, tvec)
    print("ransac:", bestCount)
    return bestFit
    
## termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)

objp *= 30

## Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i in range(1, 18):
    filename = f"img{i:02d}.jpg"
    print(filename)
    img = cv2.imread("data/"+filename)
    #  show_img(img, "original")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #  show_img(gray, "gray")
    

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,5), corners2, ret)
        
        #  show_img(img, filename)

        cv2.imwrite("out/"+filename, img)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

R = np.eye(3)
m1type=cv2.CV_32FC1
interpolation=cv2.INTER_LINEAR

img = cv2.imread('data/img01.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, newcameramtx, (w,h), m1type)

for i in range(1, 18):
    filename = f"img{i:02d}.jpg"

    img = cv2.imread('data/'+filename)
    h,  w = img.shape[:2]
    
    ## undistort
    undistorted_image = cv2.remap(img, mapx, mapy, interpolation)


    combined = cv2.hconcat([img, undistorted_image])
    resized = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
    #  show_img(resized, filename)

    
    
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    # If found, add object points, image points (after refining them)
    if ret == False:
        continue
        
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    
    marker_points = objp.copy()
    
    trash = []
    rvecs = []
    tvecs = []
    not_important, R, t = cv2.solvePnP(marker_points, corners2, newcameramtx, 0, False, cv2.SOLVEPNP_IPPE)
    rvecs.append(R)
    tvecs.append(t)
    trash.append(not_important)
    #  return rvecs, tvecs, trash
    #  print("rvecs:", rvecs)
    #  print("tvecs:", tvecs)
    
    
    objpts = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]) * 30.0

    imgpts = np.rint(cv2.projectPoints(objpts, rvecs[0], tvecs[0], newcameramtx, 0,)[0]).astype(int)
    imgpts = imgpts.reshape((-1, 2))

    #  print(imgpts)
    
    img_draw = undistorted_image.copy()
    
    img_draw = draw(img_draw, [imgpts[0]], imgpts[1:])
    
    pnp_no_noise = img_draw.copy()

    #  show_img(img_draw, filename + " PNP")
    
    
    
    
    # NOISY DATA
    
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    # If found, add object points, image points (after refining them)
    if ret == False:
        continue
        
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

    # Draw and display the corners
    
    img_draw = undistorted_image.copy()
    cv2.drawChessboardCorners(img_draw, (8,5), corners2, ret)
    corners_no_noise = img_draw.copy()

    #  show_img(img_draw, filename + " corners no noise")
        
    # making data noisy

    corners_noisy = corners.copy()
    pts = corners_noisy.reshape(-1, 2)
    n = len(pts)
    n_perturb = min(10, n)

    rng = np.random.default_rng()
    perturb_idx = rng.choice(n, size=n_perturb, replace=False)
    replace_idx = np.setdiff1d(np.arange(n), perturb_idx)

    # Slightly disturb selected points
    pts[perturb_idx] += rng.normal(
        0, 0.5, size=(n_perturb, 2)
    ).astype(pts.dtype)

    # Replace the rest with random image coordinates
    h, w = gray.shape[:2]
    pts[replace_idx, 0] = rng.uniform(0, w, size=len(replace_idx))
    pts[replace_idx, 1] = rng.uniform(0, h, size=len(replace_idx))

    
    corners = corners_noisy

        
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    
    # Draw and display the corners
    
    img_draw = undistorted_image.copy()
    cv2.drawChessboardCorners(img_draw, (8,5), corners2, ret)

    corners_with_noise = img_draw.copy()
    
    combined = cv2.hconcat([corners_no_noise, corners_with_noise])
    resized = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
    show_img(resized, filename + " corners no noise vs with noise")
    
    
    marker_points = objp.copy()
    
    trash = []
    rvecs = []
    tvecs = []
    
    
    #  not_important, R, t = cv2.solvePnP(marker_points, corners2, newcameramtx, 0, False, cv2.SOLVEPNP_IPPE)
    not_important, R, t = solvePnP_with_RANSAC(marker_points, corners2, newcameramtx, 10)
    
    
    rvecs.append(R)
    tvecs.append(t)
    trash.append(not_important)
    #  return rvecs, tvecs, trash
    #  print("rvecs:", rvecs)
    #  print("tvecs:", tvecs)
    
    
    objpts = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]) * 30.0

    imgpts = np.rint(cv2.projectPoints(objpts, rvecs[0], tvecs[0], newcameramtx, 0,)[0]).astype(int)
    imgpts = imgpts.reshape((-1, 2))

    #  print(imgpts)
    
    img_draw = undistorted_image.copy()
    
    img_draw = draw(img_draw, [imgpts[0]], imgpts[1:])
    
    pnp_with_noise = img_draw.copy()


    #  show_img(img_draw, filename+ " PnP with noise")
    combined = cv2.hconcat([pnp_no_noise, pnp_with_noise])
    resized = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
    show_img(resized, filename + "PNP with no noise vs PNP with my RUNSAC with noise")
    
    
    
    
    
    
    

    #OPENCV IMPLEENTATION
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    
    # Draw and display the corners
    
    img_draw = undistorted_image.copy()
    cv2.drawChessboardCorners(img_draw, (8,5), corners2, ret)    
    
    marker_points = objp.copy()
    
    trash = []
    rvecs = []
    tvecs = []
    
    
    #  not_important, R, t = cv2.solvePnP(marker_points, corners2, newcameramtx, 0, False, cv2.SOLVEPNP_IPPE)
    success, R, t, inliers = cv2.solvePnPRansac(marker_points, corners2, newcameramtx, distCoeffs = None, iterationsCount=10000)
    
    print(success)
    
    rvecs.append(R)
    tvecs.append(t)
    trash.append(not_important)
    #  return rvecs, tvecs, trash
    #  print("rvecs:", rvecs)
    #  print("tvecs:", tvecs)
    
    
    objpts = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]) * 30.0

    imgpts = np.rint(cv2.projectPoints(objpts, rvecs[0], tvecs[0], newcameramtx, 0,)[0]).astype(int)
    imgpts = imgpts.reshape((-1, 2))

    #  print(imgpts)
    
    img_draw = undistorted_image.copy()
    
    img_draw = draw(img_draw, [imgpts[0]], imgpts[1:])
    
    opencv_pnp_ransac = img_draw.copy()
    
    combined = cv2.hconcat([pnp_with_noise, opencv_pnp_ransac])
    resized = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))
    show_img(resized, filename + " my vs opencv")
    
