"""
reference: https://medium.com/lifes-a-struggle/%E5%8F%96%E5%BE%97-region-proposals-selective-search-%E5%90%AB%E7%A8%8B%E5%BC%8F%E7%A2%BC-be0aa5767901
"""
import cv2
import time



def visualize_region_proposal(img, rects):
    # number of region proposals to display and the increase/decrease number
    rects_show_num = 300
    increment = 50

    # show the proposals dynamically
    while True:
        # clone the original image to draw on it
        img_show = img.copy()

        # loop over every region proposal
        for i, rect in enumerate(rects):
            if i < rects_show_num:
                # draw the region proposal bounding box on the image
                x, y, w, h = rect
                cv2.rectangle(img_show, (x, y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)
            else:
                break

        # show
        cv2.imshow("region proposals", img_show)

        # read the pressed key
        key = cv2.waitKey(0) & 0xFF

        # if 'w' is pressed, increase rects_show_num
        if key == ord('w'):
            rects_show_num += increment
        
        # if 'e' is pressed, decrease rects_show_num
        elif key == ord('e'):
            rects_show_num -= increment
        
        # if 'q' is pressed, break from the loop
        elif key == ord('q'):
            break

        # if 's' is pressed, save the result
        # elif key == ord('s'):
        #     cv2.imwrite(f"result_show_{rects_show_num}.png", img_show)
    
    # close window
    cv2.destroyAllWindows()






def selective_search(img, method="fast"):
    # initialize selective search implementation and set the input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if method == "fast":
        # fast, but quality not so good
        ss.switchToSelectiveSearchFast()
    elif method == "quality":
        # good quality but relatively slow
        ss.switchToSelectiveSearchQuality()
    
    # run selective search on the input image, and record the time
    start = time.time()
    rects = ss.process()
    end = time.time()

    # print the time it took and the total number of the returned region proposals
    print(f"the selective search using mode {method} took {end - start:.2f} seconds")
    print(f"number of region proposals: {len(rects)}")

    return rects








if __name__ == "__main__":
    # img_path = "img/cat.png"
    img_path = "img/lakers.png"
    img = cv2.imread(img_path)

    # run selective search
    rects = selective_search(img, "quality")

    # visualize
    visualize_region_proposal(img, rects)

