'''Spacell Vaidation 

Validates cell type identification model against pathological annotations

Example Usage (for red annotations):

python spacell_validation.py -a annotation.png -w wsi_10x_downscale.jpeg -m transformation_matrix.txt -o output -k cluster_predictions.tsv -c 0 0 170 160 160 255 -t -f 10

'''


import argparse
from pathlib import Path
import glob
import io
import matplotlib.pyplot as plt
import numpy as np
import os
#import openslide
#from openslide import open_slide
import pandas as pd
from PIL import Image, ImageOps
import seaborn as sns
import cv2
from utils import scatter_plot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support


#Allow Pillow to open very big images
Image.MAX_IMAGE_PIXELS = None

###############################
# Image Processsing Functions #
###############################

def imshow(img_bgr):
    """Displays colour OpenCV image with Pillow
    """
    img_rgb = img_bgr[:, :, ::-1]  # Convert from BGR to RGB
    return Image.fromarray(img_rgb)


def scale_img(img, scale_factor):
    """Scales OpenCV images with a constant scale factor

        Parameters
        ----------
        img : OpenCV Image (BGR)

        scale_factor : float
            Constant factor to scale image by.

        Returns
        -------
        scaled_img : OpenCV Image (BGR)
    """
    new_width = int(round(img.shape[1] * scale_factor))
    new_height = int(round(img.shape[0] * scale_factor))
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return scaled_img


def thumbnail_img(img_path, resize_factor):
    """Resize Pillow images with a constant resize factor

        Parameters
        ----------
        img_path : image Path

        resize_factor : int
            Constant factor to resize image by.

        Returns
        -------
        scaled_img : Pillow thumbnail image
        """
    img = Image.open(img_path)
    size = int(round(img.shape[0] / resize_factor)), int(round(img.shape[1] / resize_factor))
    img.thumbnail(size)
    return img

def annotation_overlay_plot(annotation, best_scale, wsi, out_path):
    """Pastes annotation image onto whole slide image"""
    annotation_scaled = scale_img(annotation, best_scale)
    locs = np.nonzero(annotation_scaled)
    annotation_overlay = wsi.copy()
    annotation_overlay[locs[0] + best_loc[1], locs[1] + best_loc[0]] = annotation_scaled[locs[0], locs[1]]
    overlay_pil = imshow(annotation_overlay)
    overlay_pil.save(out_path, "JPEG")


def closed_contours_index(hierachy):
    """Determines the index of contours that contain contours within them"""
    return np.where(hierachy[0][:,2]!= -1)[0]

def solidity(contour):
    """Calcuates solidity which is the ratio of the contour area to its convex hull area
    """
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity

################################
# Image Registration Functions #
################################

def detect_max_scale_pil(annotation, wsi):
    """Detect maximum possible annotation scale for Pillow Images
    
    """
    annotation_width, annotation_height = annotation.size
    wsi_width, wsi_height = wsi.size
    max_width_scale = wsi_width/annotation_width
    max_height_scale = wsi_height/annotation_height
    return int(min(max_width_scale, max_height_scale))


def detect_max_scale(annotation, wsi):
    """Detects maximum possible annotation scale for OpenCV Images

    The maximum possible annotation scale is one where the annotation image is so big it doesn't fit within the whole slide image dimensions
    
        Parameters
        ----------
        annotation : OpenCV image
        wsi : OpenCV image

        Returns
        -------
        max_scale : int 
    """
    annotation_height, annotation_width, _ = annotation.shape
    wsi_height, wsi_width, _ = wsi.shape
    max_width_scale = wsi_width/annotation_width
    max_height_scale = wsi_height/annotation_height
    return min(max_width_scale, max_height_scale)


def detect_min_scale(annotation, wsi, annotation_proportion = 0.1):
    """Detects minimum annotation scale where the annotation makes up a certain proportion of the WSI area

        Parameters
        ----------
        annotation : OpenCV image
        wsi : OpenCV image

        Returns
        -------
        min_scale : float
    """
    annotation_height, annotation_width, _ = annotation.shape
    wsi_height, wsi_width, _ = wsi.shape
    wsi_area = wsi_height*wsi_width
    min_annotation_area = wsi_area*annotation_proportion
    k = annotation_width/annotation_height
    min_height = np.sqrt(min_annotation_area/k)
    min_scale = min_height/annotation_height
    return min_scale


def grid_search_registration(wsi_small_gray, annotation_gray, min_scale, max_scale, out_path, iteration=20):
    """Register the annotation image to the whole slide image

    Uses a sliding window approach to find the location and scale of the annotation image on the whole slide image
    that maximises the normalised correlation coefficient.
        
        Parameters
        ----------
        wsi_small_gray : OpenCV grayscale image
        annotation_gray : OpenCV grayscale image
        min_scale : float
        max_scale : int
        out_path : Path
        iteration : int
            Number of scale steps to try between min_scale and max_scale.
            More steps increases accuracy but also running time.

        Returns
        -------
        best_loc : tuple
        best_scale : float
        best_max_val : float
            Normalised correlation coefficient at this location and scale
    """
    best_loc = (0, 0)
    best_scale = 0
    best_max_val = 0
    max_vals = []
    scale_range = np.linspace(min_scale, max_scale, iteration)
    for scale_factor in scale_range:
        annotation_scaled_gray = scale_img(annotation_gray, scale_factor)
        result = cv2.matchTemplate(wsi_small_gray, annotation_scaled_gray, cv2.TM_CCOEFF_NORMED)
        sin_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        max_vals.append(max_val)
        if max_val > best_max_val:
            best_max_val = max_val
            best_scale = scale_factor
            best_loc = max_loc
    registration_correlation_coefficient_plot(scale_range, max_vals, out_path)
    return best_loc, best_scale, best_max_val


def registration_correlation_coefficient_plot(x, y, out_path):
    """Plots the annotation registration normalised correlation coefficent at different scale and saves it 
    """
    sns.lineplot(x=x, y=y)
    plt.xlabel('Annotation Scale Factor')
    plt.ylabel('Max. Normalised Correlation Coefficient')
    plt.savefig(out_path, dpi=180)
    plt.close()

#######################################
# Validation Quantification Functions #
#######################################

def parseAlignmentMatrix(alignment_file, resize_factor=1):
    alignment_matrix = np.identity(3)
    with open(alignment_file, "r") as filehandler:
        line = filehandler.readline()
        tokens = line.split()
        assert(len(tokens) == 9)
        alignment_matrix[0,0] = float(tokens[0])/resize_factor
        alignment_matrix[1,0] = float(tokens[1])/resize_factor
        alignment_matrix[2,0] = float(tokens[2])/resize_factor
        alignment_matrix[0,1] = float(tokens[3])/resize_factor
        alignment_matrix[1,1] = float(tokens[4])/resize_factor
        alignment_matrix[2,1] = float(tokens[5])/resize_factor
        alignment_matrix[0,2] = float(tokens[6])/resize_factor
        alignment_matrix[1,2] = float(tokens[7])/resize_factor
        alignment_matrix[2,2] = float(tokens[8])
    return alignment_matrix


def true_cluster(cluster_pred, mask):
    spot_x = int(round(cluster_pred['spot_x']))
    spot_y = int(round(cluster_pred['spot_y']))
    if mask[spot_y, spot_x] == 0:
        return 0
    else:
        return 1


def transform_spot(cluster_pred, x_offset, y_offset, x_scale, y_scale):
    spot_x = cluster_pred['spot_x']*x_scale + x_offset
    spot_y = cluster_pred['spot_y']*y_scale + y_offset
    return pd.Series({'spot_x': spot_x, 'spot_y': spot_y, 'pred_colour': cluster_pred['pred_colour']})


def calculate_performance(cluster_preds, out_path):
    acc = accuracy_score(cluster_preds.pred_label, cluster_preds.true_label)
    tn, fp, fn, tp = confusion_matrix(cluster_preds.true_label, cluster_preds.pred_label).ravel()
    fpr, tpr, _ = roc_curve(cluster_preds.true_label, cluster_preds.pred_label)
    roc_auc = auc(fpr, tpr)
    # if roc_auc < 0.5:
    #     fpr, tpr, _ = roc_curve(cluster_preds.true_label.invert(), cluster_preds_pred_label.invert())
    precision, recall, fscore, _ = precision_recall_fscore_support(cluster_preds.true_label,
                                                                   cluster_preds.pred_label,
                                                                   average='binary')
    plot_ROC(fpr, tpr, roc_auc, out_path)

    return [acc, tp, tn, fn, fp, precision, recall, fscore, roc_auc]


def plot_ROC(fpr, tpr, roc_auc, out_path):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=Path, default=Path(__file__).absolute().parent.parent / 'dataset' / 'validation',
                        help='Path to the data directory')
    parser.add_argument('-a', '--annotation_path', type=Path,
                        help='Path to the annotation image relative to the data path')
    parser.add_argument('-w', '--wsi_path', type=Path,
                        help='Path to the Whole Slide Image relative to the data path')
    parser.add_argument('-m', '--affine_matrix_path', type=str, default=None,
                        help='Path to the affine transformation matrix file relative to the data path')
    parser.add_argument('-o', '--out_path', type=Path, default=Path(__file__).absolute().parent.parent / 'dataset' / 'validation',
                        help='Path to the output relative to the data path')
    parser.add_argument('-k', '--cluster_path', type=Path,
                        help='Path to the clustering result directory')
    parser.add_argument('-c', '--annotation_colour_range', type=int, nargs='+', default = (),
                        help='annotation colour range - Blue_low Green_low Red_low Blue_high Green_high Red_high')
    parser.add_argument('-t', '--open_annotation', action='store_true',
                        help='open annotation area')
    parser.add_argument('-f', '--downscale_factor', type=float, default = 1,
                        help= 'downscale factor of the input wsi')
    parser.add_argument('-s', '--spot_size', type=int, default = 10,
                        help= 'spot size for plotting')
    parser.add_argument('-v', '--verbosity', action='store_true',
                        help= 'increase output verbosity')

    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################

    #Input Paths
    BASE_PATH = args.data_path
    ANNOTATION_PATH = BASE_PATH / args.annotation_path
    WSI_PATH = BASE_PATH / args.wsi_path
    CLUSTER_PREDICTIONS_PATH = BASE_PATH / args.cluster_path
    if args.affine_matrix_path:
        ATM_PATH = BASE_PATH / Path(args.affine_matrix_path)
    else:
        ATM_PATH = None

    #Output Paths    
    OUT_PATH = os.path.join(BASE_PATH , args.out_path)
    WSI_SMALL_PATH = BASE_PATH / Path(WSI_PATH.stem + "_10x" + WSI_PATH.suffix)
    REGISTRATION_PLOT_PATH = OUT_PATH/ Path('registration_correlation_coefficient_plot.pdf')
    OVERLAY_IMG_PATH = OUT_PATH / Path('registration_overlay_image_10x.jpeg')
    WSI_SMALL_CONTOUR_PATH = OUT_PATH / Path('contour_overlay_image_10x.jpeg')
    WSI_SMALL_CONTOUR_SPOT_PATH = OUT_PATH / Path('contour_spot_overlay_image_10x')
    ROC_CURVE_PATH = OUT_PATH / Path('roc_curve.pdf')
    #Verbose Only Outputs
    ANNOTATION_MASK_PATH = OUT_PATH / Path('annotation_mask.jpeg')
    WSI_MASK_PATH = OUT_PATH / Path('wsi_mask.jpeg')
    WSI_MASK_SPOTS_PATH = OUT_PATH / Path('wsi_mask_spots.jpeg')

    #User selectable parameters
    ANNOTATION_COLOUR_RANGE = tuple(args.annotation_colour_range)
    OPEN_ANNOTATION = args.open_annotation
    DOWNSCALE_FACTOR = args.downscale_factor    #Downscale factor that has been applied to input wsi
    SPOT_SIZE = args.spot_size
    VERBOSE = args.verbosity

    if VERBOSE:
        verboseprint = lambda *args: print(*args)
    else:
        verboseprint = lambda *args: None

    verboseprint('Output directory is: {0}'.format(OUT_PATH))

    ####################################
    # Extract and Register Annotations #
    ####################################
    
    # Generate 10x resized thumbnails for WSI
    RESIZE_FACTOR = 1
    #wsi_small = thumbnail_img(WSI_PATH, RESIZE_FACTOR)
    #wsi_small.save(WSI_SMALL_PATH, "JPEG")
    # Load annotation image and 10x resized WSI with OpenCV
    # cv2 reads string representation of paths
    annotation = cv2.imread(str(ANNOTATION_PATH))
    #To do: decide on whether or not to pre-downscale
    wsi_small = cv2.imread(str(WSI_PATH))
    # Convert images to grayscale
    annotation_gray = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
    wsi_small_gray = cv2.cvtColor(wsi_small, cv2.COLOR_BGR2GRAY)

    verboseprint('Registering Annotation to Whole Slide Image ... ')
    # Detect maximum and maximum possible annotation scale
    max_scale = detect_max_scale(annotation, wsi_small)
    min_scale = detect_min_scale(annotation, wsi_small)
    # Calculate best annotation scale based on pixel correlation coefficient after registration
    best_loc, best_scale, best_max_val = grid_search_registration(wsi_small_gray, annotation_gray,
                                                                  min_scale, max_scale,
                                                                  REGISTRATION_PLOT_PATH, iteration=20)
    # Predict pixel locations of annotation in 10x resized WSI
    verboseprint(best_loc, best_scale, best_max_val)
    best_scale_wsi = best_scale * RESIZE_FACTOR
    x_offset = int(round(best_loc[0] * RESIZE_FACTOR))
    y_offset = int(round(best_loc[1] * RESIZE_FACTOR))
    verboseprint(best_scale_wsi, x_offset, y_offset)
    # Overlay annotation onto 10x resized WSI
    annotation_overlay_plot(annotation, best_scale, wsi_small, OVERLAY_IMG_PATH)
    # Detect annotation contour
    # Generate Binary mask of annotation
    #Generates annotation mask from user-specified colour range
    annotation_mask = cv2.inRange(annotation, ANNOTATION_COLOUR_RANGE[0:3], ANNOTATION_COLOUR_RANGE[3:6])

    ############################
    # Generate Annotation Mask #
    ############################

    # Make contour Mask
    contour_line = cv2.bitwise_and(annotation, annotation, mask=annotation_mask.astype(np.uint8))
    contours, hierachy = cv2.findContours(annotation_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if OPEN_ANNOTATION:
        #Find longest contour
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        cnt_index = np.argmax(perimeters)
        cnt = contours[cnt_index]
        hull = cv2.convexHull(cnt)
        hull_mask = np.zeros(annotation_mask.shape)
        hull_mask = cv2.drawContours(hull_mask, [hull], 0, 255, 3)
        annotation_mask_hulled = cv2.bitwise_or(annotation_mask.astype(np.uint8), hull_mask.astype(np.uint8))
        #Set flood fill seed to be the centroid of the hull
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        #Flood fill 
        flood_mask = np.zeros((annotation_mask_hulled.shape[0] + 2, annotation_mask_hulled.shape[1] + 2)).astype(np.uint8)
        cv2.floodFill(annotation_mask_hulled, flood_mask, (centroid_x, centroid_y), 255)
        #remove hull
        annotation_mask_filled = cv2.subtract(annotation_mask_hulled.astype(np.uint8), hull_mask.astype(np.uint8))
        #Add any accidentally removed pixels from line
        annotation_mask = cv2.bitwise_or(annotation_mask_filled, (annotation_mask*255).astype(np.uint8))
    else:
        for c in contours:
            annotation_mask = cv2.drawContours(annotation_mask, [c], -1, 1, cv2.FILLED)

    if VERBOSE:
        #Show Masked area on annotated image
        masked_area = annotation_mask
        imshow(cv2.bitwise_and(annotation, annotation, mask = masked_area )).save(ANNOTATION_MASK_PATH)

    # Generate mask for 10x resized WSI
    # Scale contour Mask
    contour_line_scaled = scale_img(contour_line, best_scale_wsi)
    # overlay contour line onto 10x resized WSI
    contour_locs = np.nonzero(contour_line_scaled)
    wsi_annotated = wsi_small.copy()
    wsi_annotated[contour_locs[0] + y_offset, contour_locs[1] + x_offset] = contour_line_scaled[contour_locs[0], contour_locs[1]]
    wsi_annotated_pil = imshow(wsi_annotated)
    wsi_annotated_pil.save(WSI_SMALL_CONTOUR_PATH, "JPEG")

    # Make a mask of the annotation for the full-size WSI
    # Make Blank Mask
    wsi_full_height = int(round(wsi_small.shape[0] * DOWNSCALE_FACTOR))
    wsi_full_width = int(round(wsi_small.shape[1] * DOWNSCALE_FACTOR))
    wsi_mask = np.zeros((wsi_full_height, wsi_full_width), dtype=np.uint8)
    # Scale annotation mask
    annotation_mask_scaled = scale_img(annotation_mask.astype(np.uint8), best_scale_wsi * DOWNSCALE_FACTOR)
    wsi_x_offset = int(round(best_loc[0] * RESIZE_FACTOR * DOWNSCALE_FACTOR))
    wsi_y_offset = int(round(best_loc[1] * RESIZE_FACTOR * DOWNSCALE_FACTOR))
    mask_locs = np.nonzero(annotation_mask_scaled)
    wsi_mask[mask_locs[0] + wsi_y_offset, mask_locs[1] + wsi_x_offset] = annotation_mask_scaled[mask_locs[0], mask_locs[1]]
    verboseprint(wsi_mask.shape)
    # Overlay Spots and Annotation on 10x resized WSI
    if ATM_PATH:
        atm = parseAlignmentMatrix(ATM_PATH, DOWNSCALE_FACTOR)
        verboseprint(atm)
    else:
        atm = None

    if VERBOSE:
        plt.imshow(wsi_mask)
        plt.savefig(WSI_MASK_PATH, dpi=180)

    ################################
    # Visualise and Quantify Spots #
    ################################

    cluster_preds = pd.read_csv(CLUSTER_PREDICTIONS_PATH, header=0, sep=',')
    cluster_preds = cluster_preds.rename(columns={'label': 'pred_colour'})
    #Scale Affine Transform Matrix to account for DOWNSCALE FACTOR
    scatter_plot(cluster_preds.spot_x, cluster_preds.spot_y, colors=cluster_preds.pred_colour, alignment=atm, image=WSI_SMALL_CONTOUR_PATH,
                 output=WSI_SMALL_CONTOUR_SPOT_PATH)
    # transform spot coords to pixel coords
    spot_x_scale = atm[0, 0] * DOWNSCALE_FACTOR
    spot_x_offset = atm[0, 2] * DOWNSCALE_FACTOR
    spot_y_scale = atm[1, 1] * DOWNSCALE_FACTOR
    spot_y_offset = atm[1, 2] * DOWNSCALE_FACTOR
    verboseprint('x_scale: {0} | x_offset: {1} | y_scale: {2} | y_offset: {3}'.format(spot_x_scale, spot_x_offset, spot_y_scale, spot_y_offset))
    cluster_preds_scaled = cluster_preds.apply(transform_spot, args = (spot_x_offset, spot_y_offset, spot_x_scale, spot_y_scale), axis = 1)

    if VERBOSE:
        fig, a = plt.subplots(figsize = (4,4), dpi = 180)
        sc = a.scatter(cluster_preds_scaled['spot_x'], cluster_preds_scaled['spot_y'], c = cluster_preds_scaled.pred_colour, s = SPOT_SIZE)
        a.imshow(wsi_mask)
        fig.savefig(WSI_MASK_SPOTS_PATH, dpi = 180)

    color_code = cluster_preds_scaled['pred_colour'].unique().tolist()
    verboseprint(color_code)
    colour2int_dict = {color_code[1]: 0, color_code[0]: 1}

    verboseprint(colour2int_dict)
    int2colour_dict = {v: k for k, v in colour2int_dict.items()}
    # Convert the predicted colours to integers
    cluster_preds_scaled['pred_label'] = cluster_preds_scaled['pred_colour'].replace(colour2int_dict)
    # Generate the true labels
    cluster_preds_scaled['true_label'] = cluster_preds_scaled.apply(true_cluster, args= (wsi_mask, ), axis=1)
    # Get the colours for the true labels
    cluster_preds_scaled['true_colour'] = cluster_preds_scaled['true_label'].replace(int2colour_dict)

    if accuracy_score(cluster_preds_scaled.pred_label, cluster_preds_scaled.true_label) < 0.5:

        colour2int_dict = {color_code[0]: 0, color_code[1]: 1}

        verboseprint(colour2int_dict)
        int2colour_dict = {v: k for k, v in colour2int_dict.items()}
        # Convert the predicted colours to integers
        cluster_preds_scaled['pred_label'] = cluster_preds_scaled['pred_colour'].replace(colour2int_dict)
        # Generate the true labels
        cluster_preds_scaled['true_label'] = cluster_preds_scaled.apply(true_cluster, args= (wsi_mask, ), axis=1)
        # Get the colours for the true labels
        cluster_preds_scaled['true_colour'] = cluster_preds_scaled['true_label'].replace(int2colour_dict)

    # Model performance matrix
    acc, tp, tn, fn, fp, precision, recall, fscore, roc_auc = calculate_performance(cluster_preds_scaled, ROC_CURVE_PATH)

    verboseprint(cluster_preds_scaled)

    print("Accuracy: {0}, TP: {1}, TN: {2}, FN: {3}, FP: {4}, Precision: {5}, Recall: {6}, F-score: {7}, ROC AUC: {8}".format(acc, tp, tn, fn, fp, precision, recall, fscore, roc_auc))

















