import cv2
import random
import numpy as np



def visualize_segment(img, segment, plot_type="color_segment"):
    seg_img = np.zeros(img.shape, np.uint8)

    # plot each segment with a random color
    for i in range(np.max(segment)):
        # take out the coords for segment i
        y, x = np.where(segment == i)

        # random generate color
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        if plot_type == "color_segment":
            # set the color for segment i
            for xi, yi in zip(x, y):
                seg_img[yi, xi] = color
        elif plot_type == "bounding_box":
            # calculate the border
            top, bottom, left, right = min(y), max(y), min(x), max(x)
            cv2.rectangle(img, (left, bottom), (right, top), (0, 255, 0), 1)

    if plot_type == "color_segment":
        # combine the original image with the segmented color
        img = cv2.addWeighted(img, 0.3, seg_img, 0.7, 0)

    # show the result
    cv2.imshow("graph-based segmentation result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def graph_based_segmentation(img):
    # initialize segmentator
    segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=5000)

    # process segmentation
    segment = segmentator.processImage(img)

    return segment



if __name__ == "__main__":
    # img_path = "img/cat.png"
    img_path = "img/lakers.png"
    img = cv2.imread(img_path)

    # run graph-based segmentation
    segment = graph_based_segmentation(img)

    # visualize
    # visualize_segment(img, segment, "color_segment")
    visualize_segment(img, segment, "bounding_box")



