import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans


def refine_segmentation_by_conncomps(img):
    totalLabels, label_ids, values, centroids = cv2.connectedComponentsWithStats(img, 4)
    areas = [values[i, cv2.CC_STAT_AREA] for i in range(1, totalLabels)]
    picked_label = 1 + np.argmax(areas)
    img_refined = (label_ids == picked_label) * 1

    return img_refined


def display_colors_point_cloud(colors_vec):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Extract RGB values from colors
    u = [color[0] for color in colors_vec]
    v = [color[1] for color in colors_vec]

    # Plot point cloud
    ax.scatter(u, v, marker='.', s=1)

    # Set axis labels
    ax.set_xlabel('U')
    ax.set_ylabel('V')

    plt.show()


def pick_color_in_image(img, colors_picked):
    colors_vec = img.reshape([-1, 2])
    # display_colors_point_cloud(colors_vec)

    km = KMeans(n_clusters=7).fit(colors_vec)
    pixels_clustered = km.labels_
    im_clustered = pixels_clustered.reshape((img.shape[0], img.shape[1]))

    # plt.figure()
    # plt.imshow(im_clustered)

    # finding the cluster closest to color_picked
    im_colors = []
    for color_point in colors_picked:
        colors_cluster = km.predict([color_point])
        im_color = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
        im_color[im_clustered == colors_cluster] = 1

        im_colors.append(im_color)
    return im_colors


def split_to_objects_masks(img_yuv, plot_results=False):
    img_repr_uv = img_yuv[:, :, 1:]

    green_uv = (139, 70)
    red_uv = (112, 198)
    blue_uv = (154, 110)
    pink_uv = (131, 160)

    picked_colors = [green_uv, red_uv, blue_uv, pink_uv]
    colored_images = pick_color_in_image(img_repr_uv, picked_colors)

    im_green, im_red, im_blue, im_pink = colored_images

    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(im_green)
    # ax[0, 1].imshow(im_red)
    # ax[1, 0].imshow(im_blue)
    # ax[1, 1].imshow(im_pink)

    im_green_refined = refine_segmentation_by_conncomps(im_green)
    im_red_refined = refine_segmentation_by_conncomps(im_red)
    im_blue_refined = refine_segmentation_by_conncomps(im_blue)
    im_pink_refined = refine_segmentation_by_conncomps(im_pink)

    if plot_results:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(im_green_refined)
        ax[0, 1].imshow(im_red_refined)
        ax[1, 0].imshow(im_blue_refined)
        ax[1, 1].imshow(im_pink_refined)
        plt.show()

    return im_green_refined, im_red_refined, im_blue_refined, im_pink_refined



if __name__ == '__main__':
    im_path = r"C:\Users\adirm\Desktop\projects\PS-segmentation\data\image1.jpg"

    img = cv2.imread(im_path)
    objects_masks = split_to_objects_masks(img, plot_results=True)
