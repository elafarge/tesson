import csv

# 47.96mm

# 3p
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

TRESH_LASER = 230
VERTICAL_OFFSET = 10
SIDE_OFFSET = 10
LINE_SHIFT = 10
COSA = 1/1.41
REF_H = 101.76 # mm
REF_W = 102.87 # mm
MM_PX_V_RATE = 10 # Ratio millimeters to pixels for the final bitmap (vertical)
MM_PX_H_RATE = 10 # Ratio millimeters to pixels for the final bitmap (horizontal)
HEIGHT_THRESHOLD = 200 # pixel jump to start/stop detection of the studhorse

def transpose_pixmap(pixmap):
    ret = np.zeros((pixmap.shape[1], pixmap.shape[0], pixmap.shape[2]), np.uint8)
    for i, l in enumerate(pixmap):
        for j, elt in enumerate(l):
            ret[j][i] = elt
    return ret

def laser_cloud_from_picture_in_px_before_projection(picture_path):
    # Load the image
    image = cv2.imread(picture_path)

    # Tresholding of the image (TODO: make threshold parametrizable)
    ret, tresh_img = cv2.threshold(image, TRESH_LASER, 255, cv2.THRESH_TOZERO)

    # Peak detection
    i = 0
    peak = []
    tresh_img_trans = transpose_pixmap(tresh_img)
    while i < tresh_img_trans.shape[0]:
        col = tresh_img_trans[i]
        sm = 0
        coef = 0
        for j in range(0, len(col)-1):
            intensity = pixel_intensity(col[j])
            coef += intensity
            sm += intensity * j
        if coef == 0:
            peak.append(None)
        else:
            peak.append(sm/coef)
        i += 1

    return peak, tresh_img

def pixel_intensity(pixel):
    # TODO: make coefficients parametrizable
    return 1/2*pixel[1] + 1/2*pixel[2]

def make_profile(ref_cloud, sample_cloud):
    heights = []

    # Compute the differences and keep track of the max
    max_height = 0
    for j, p in enumerate(sample_cloud):
        if p is not None and ref_cloud[j] is not None and (ref_cloud[j] - p) > LINE_SHIFT:
            heights.append(ref_cloud[j] - p)
            if ref_cloud[j] - p > max_height:
                max_height = ref_cloud[j] - p
        else:
            heights.append(None)

    # Now that we have the profile let's put it in a bitmap (# TODO change type to optimize
    # memory)
    bmp_h = max_height + 2 * VERTICAL_OFFSET

    bitmap = np.zeros((bmp_h, len(heights) + 2*SIDE_OFFSET, 1), np.uint8)
    for j, h in enumerate(heights):
        if h is not None:
            cv2.circle(bitmap, (j + SIDE_OFFSET, int(bmp_h - h)), 1, 255)

    return bitmap, heights

def unproject(studhorse_path, ref_cloud, heights_px):
    # Let's compute the profile of the reference...
    studhorse_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(
            studhorse_path)

    studhorse_bmp, studhorse_profile = make_profile(ref_cloud, studhorse_cloud)

    h_px = max([h for h in studhorse_profile if h is not None])

    # ... and compute the ratio
    v_ratio = REF_H/h_px

    # The horizontal ratio is computed in a slightly different way
    x1 = None
    y1 = 0
    x2 = None
    for i, h in enumerate(studhorse_profile):
        if h is not None and abs(h - y1) > HEIGHT_THRESHOLD:
            if x1 is None:
                x1 = i
                y1 = h
            else:
                x2 = i
                break
        i += 1

    # If we didn't find a last point, it probably means our base line is perfectly
    # removed, but we have to trick slightly to get our horizontal ratio
    if x2 is None:
        x2 = [(i, h) for i, h in enumerate(studhorse_profile) if h is not None][-1][0]

    h_ratio = REF_W/(x2-x1)

    # To finally get ourselves a nice scaled profile in millimeters
    heights_mm = []
    for j, h in enumerate(heights_px):
        x_mm = j*h_ratio
        heights_mm.append((x_mm, h*v_ratio if h is not None else None))

    return heights_mm


def run_profiling(reference_path, studhorse_path, sample_path, out_path, csv_out_path):
    # Get clouds from it
    print("Generating cloud for reference image...")
    ref_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(reference_path)
    cv2.namedWindow('Reference', cv2.WINDOW_NORMAL)
    cv2.imshow('Reference', bitmap)

    print("Generating cloud for artifact image...")
    sample_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(sample_path)
    cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
    cv2.imshow('Sample', bitmap)

    # Get the profile as a bitmap
    print("Generating pixel profile...")
    profile, heights_px = make_profile(ref_cloud, sample_cloud)
    cv2.namedWindow('Pixel profile', cv2.WINDOW_NORMAL)
    cv2.imshow('Pixel profile', profile)

    # Get a profile in millimeters and another one in pixels
    print("Using the studhorse to get a profile in millimeters...")
    heights_mm = unproject(studhorse_path, ref_cloud, heights_px)

    # Let's save the profile
    x_data = []
    y_data = []
    with open(csv_out_path, 'w') as pro_file:
        profile_writer = csv.writer(pro_file, delimiter=' ', quotechar='|',
                quoting=csv.QUOTE_MINIMAL)
        for x, h in heights_mm:
            if h is not None:
                print([x, h])
                profile_writer.writerow([x, h])
                x_data.append(x)
                y_data.append(h)

    cv2.namedWindow('Pixel profile', cv2.WINDOW_NORMAL)
    cv2.imshow('Pixel profile', profile)

    # And plot it
    fig = plt.figure()
    profile_plot = fig.add_subplot(111)
    profile_plot.set_title("Profile Peeelot")
    profile_plot.set_xlabel("mm")
    profile_plot.set_ylabel("mm")
    profile_plot.plot(x_data, y_data, '-', label='Profile')
    plt.show()

    # Display it
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load the reference image TODO FIXME: make it possible to pass everything as
    # an argument (yes command line can be sexy)
    REFERENCE_PATH = './samples/reference.jpg'
    STUDOHORSE_PATH = './samples/studhorse.jpg'
    PROFILE_PATH = './samples/first.jpg'

    # TODO FIXME: catch CTRL-C interrupt signals

    # Launch the main function
    run_profiling(REFERENCE_PATH, STUDOHORSE_PATH, PROFILE_PATH, './out.png', 'profile.csv')
