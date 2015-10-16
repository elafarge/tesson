#stdlib
import csv
import sys
import signal
import argparse

# 3p
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

VERTICAL_OFFSET = 10
SIDE_OFFSET = 10
MM_PX_V_RATE = 10 # Ratio millimeters to pixels for the final bitmap (vertical)
MM_PX_H_RATE = 10 # Ratio millimeters to pixels for the final bitmap (horizontal)
POLY_DEGREE = 30 # degree of the approximation polynome


def transpose_pixmap(pixmap):
    ret = np.zeros((pixmap.shape[1], pixmap.shape[0], pixmap.shape[2]), np.uint8)
    for i, l in enumerate(pixmap):
        for j, elt in enumerate(l):
            ret[j][i] = elt
    return ret

def laser_cloud_from_picture_in_px_before_projection(picture_path, tresh_laser):
    # Load the image
    image = cv2.imread(picture_path)

    # Tresholding of the image
    ret, tresh_img = cv2.threshold(image, tresh_laser, 255, cv2.THRESH_TOZERO)

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

def make_profile(ref_cloud, sample_cloud, line_shift):
    heights = []

    # Compute the differences and keep track of the max
    max_height = 0
    for j, p in enumerate(sample_cloud):
        if p is not None and ref_cloud[j] is not None and (ref_cloud[j] - p) > line_shift:
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

def unproject(studhorse_path, ref_cloud, heights_px, tresh_laser, line_shift, ref_h, ref_w,
              height_threshold):
    # Let's compute the profile of the reference...
    studhorse_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(
            studhorse_path, tresh_laser)

    studhorse_bmp, studhorse_profile = make_profile(ref_cloud, studhorse_cloud, line_shift)

    h_px = max([h for h in studhorse_profile if h is not None])

    # ... and compute the ratio
    v_ratio = ref_h/h_px

    # The horizontal ratio is computed in a slightly different way
    x1 = None
    y1 = 0
    x2 = None
    for i, h in enumerate(studhorse_profile):
        if h is not None and abs(h - y1) > height_threshold:
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

    h_ratio = ref_w/(x2-x1)

    # To finally get ourselves a nice scaled profile in millimeters
    heights_mm = []
    for j, h in enumerate(heights_px):
        x_mm = j*h_ratio
        if h is not None:
            heights_mm.append((x_mm, h*v_ratio))

    return heights_mm, studhorse_bmp


def distance(array,i,j):
    val = (array[i][0]-array[j][0])*(array[i][0]-array[j][0]) + (array[i][1]-array[j][1])*(array[i][1]-array[j][1])
    return val


def polynomial_approximation(values_table, epsilon):

    i=0
    xdata = [np.array([-1])]#unknow size so we initialize a np.array with a value that will be deleted afterwards (cant initialize with nothing)
    ydata = [np.array([-1])]#unknow size so we initialize a np.array with a value that will be deleted afterwards (cant initialize with nothing)
    xinit = None
    xfinal = None
    x_range = []
    for j,px in enumerate(values_table):

        if(xinit is None):
            xinit = px[0]

        if(j>0 and distance(values_table,j,j-1)>epsilon):
            xfinal = values_table[j-1][0]
            x_range.append([xinit,xfinal])
            xinit = None

            xdata[i] = np.delete(xdata[i], 0, 0)#delete the initial -1 value
            ydata[i] = np.delete(ydata[i], 0, 0)#delete the initial -1 value
            xdata.append(np.array([-1]))#unknow size so we initialize a np.array with a value that will be deleted afterwards (cant initialize with nothing)
            ydata.append(np.array([-1]))#unknow size so we initialize a np.array with a value that will be deleted afterwards (cant initialize with nothing)
            i = i+1

        xdata[i] = np.append(xdata[i], [px[0]])
        ydata[i] = np.append(ydata[i], [px[1]])

    xdata[i] = np.delete(xdata[i], 0, 0)#delete the initial -1 value
    ydata[i] = np.delete(ydata[i], 0, 0)#delete the initial -1 value
    xfinal = xdata[i][len(xdata[i])-1]
    x_range.append([xinit,xfinal])

    #curves' approximation
    polynoms = []
    for j in range(0,len(xdata)):

        opt_coeff = np.polyfit(xdata[j], ydata[j], POLY_DEGREE)
        polynoms.append(opt_coeff)

    return polynoms,x_range

def run_profiling(reference_path, studhorse_path, sample_path, csv_out_path, tresh_laser,
        line_shift, ref_h, ref_w, height_threshold, epsilon):
    # Get clouds from it
    print("Generating cloud for reference image...")
    ref_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(reference_path,
            tresh_laser)
    cv2.namedWindow('Reference', cv2.WINDOW_NORMAL)
    cv2.imshow('Reference', bitmap)

    print("Generating cloud for artifact image...")
    sample_cloud, bitmap = laser_cloud_from_picture_in_px_before_projection(sample_path,
            tresh_laser)
    cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
    cv2.imshow('Sample', bitmap)

    # Get the profile as a bitmap
    print("Generating pixel profile...")
    profile, heights_px = make_profile(ref_cloud, sample_cloud, line_shift)
    cv2.namedWindow('Pixel profile', cv2.WINDOW_NORMAL)
    cv2.imshow('Pixel profile', profile)

    # Get a profile in millimeters and another one in pixels
    print("Using the studhorse to get a profile in millimeters...")
    heights_mm, bitmap = unproject(studhorse_path, ref_cloud, heights_px, tresh_laser,
                                   line_shift, ref_h, ref_w, height_threshold)
    cv2.namedWindow('Studhorse profile', cv2.WINDOW_NORMAL)
    cv2.imshow('Studhorse profile', bitmap)

    print("Generating profile approximation...")
    approximated_polynoms,x_range = polynomial_approximation(heights_mm, epsilon)

    # Let's save the profile
    x_data = []
    y_data = []
    with open(csv_out_path, 'w') as pro_file:
        profile_writer = csv.writer(pro_file, delimiter=' ', quotechar='|',
                quoting=csv.QUOTE_MINIMAL)
        for x, h in heights_mm:
            if h is not None:
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
    profile_plot.plot(x_data, y_data, 'o', markersize=2, label='Profile')
    profile_plot.set_aspect('equal', adjustable='box')

    #plot the approximated polynoms
    for j in range(0, len(x_range)):
         x = np.linspace(x_range[j][0],x_range[j][1])
         y = np.polyval(approximated_polynoms[j],x)
         approximated_plot = fig.add_subplot(111)
         approximated_plot.set_title("Approximated profile")
         approximated_plot.set_xlabel("mm")
         approximated_plot.set_ylabel("mm")
         approximated_plot.plot(x, y, '-', linewidth=2, label='Approximated profile')
         approximated_plot.set_aspect('equal', adjustable='box')

    plt.show()

    # Display it
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    REFERENCE_PATH_DEFAULT = './samples/reference.jpg'
    STUDHORSE_PATH_DEFAULT = './samples/studhorse.jpg'

    # Let's parse our arguments
    parser = argparse.ArgumentParser(description='Creates the 2d profile of an object using ' +
            "a laser diode and a camera. \n\n MAREVA - Projet Tesson - \n\n Pierre Benedetti \n Etienne Lafarge \n Vincent Villet")
    parser.add_argument('--source', type=str, required=True,
            help='Path to the picture to analyse')
    parser.add_argument('--dest', type=str, default='',
            help='Desired path for the output file (defaults to <source>.csv)')
    parser.add_argument('--reference', type=str, default=REFERENCE_PATH_DEFAULT,
            help='Path to the reference picture (laser line without an object).')
    parser.add_argument('--studhorse', type=str, default=STUDHORSE_PATH_DEFAULT,
            help='Path to the studhorse picture (for calibration).')
    parser.add_argument('--tresh-laser', type=int, default=230,
            help='Intensity threshold (0-255) to isolate the laser line. (default: 230)')
    parser.add_argument('--studhorse-height-tresh', type=int, default=40,
            help='Minimum height jump (in px) to detect the beginning/end of the ' +
                 'studhorse on the picture (defaults to 10px).')
    parser.add_argument('--studhorse-height', type=float, default=14.62, required=True,
            help='The height of the studhorse (in millimeters).')
    parser.add_argument('--studhorse-width', type=float, default=88.69, required=True,
            help='The width of the studhorse (in millimeters).')
    parser.add_argument('--line-drift-tolerance', type=int, default=10,
            help="A pixel will be considered as part of the reference line if it's" +
            'distance to the reference line is inferior to this value')
    parser.add_argument('--curve-clustering-distance', type=float, default=5,
            help="Minimum \"distance\" in mm² to seperate point clouds before approximating " +
            "them with polynomial curves. (default: 5mm²)")

    args = parser.parse_args()

    if args.dest == '':
        args.dest =  args.source + '.csv'
        pass

    # Catch interrupt signals
    def interrupt_handler(signal, frame):
        print("CTRL-C has been pressed, exiting...")
        sys.exit(0)

    # Launch the main function
    run_profiling(args.reference, args.studhorse, args.source, args.dest, args.tresh_laser,
                  args.line_drift_tolerance, args.studhorse_height, args.studhorse_width,
                  args.studhorse_height_tresh, args.curve_clustering_distance)


if __name__ == '__main__':
    main()
