
import cv2
import matplotlib.pyplot as plt
import LightShape
import numpy as np


def light_mask(img_yuv, mask, ax, lighting_level=50):
    img_lightned = img_yuv.copy()
    img_lightned[mask == 1, 0] += lighting_level
    saturation_pixels = img_lightned[:, :, 0] < img_yuv[:, :, 0]
    img_lightned[saturation_pixels == 1, 0] = 255

    img_lightned = cv2.cvtColor(img_lightned, cv2.COLOR_YUV2RGB)
    ax.cla()
    ax.imshow(img_lightned)
    plt.pause(0.0001)


def light_shape(img_yuv, shape, ax=None):
    shape_mask = shape.get_object_mask()
    shape.change_lighting(LightShape.light_on_sigmoid)

    if not ax:
        _, ax = plt.subplots()

    for i in range(10):
        shape.update()
        extra_luma = shape.get_extra_luma()
        light_mask(img_yuv, shape_mask, ax, lighting_level=extra_luma)

    shape.change_lighting(LightShape.light_off_sigmoid)
    for i in range(10):
        shape.update()
        extra_luma = shape.get_extra_luma()
        light_mask(img_yuv, shape_mask, ax, lighting_level=extra_luma)


def light_all_shapes(img_yuv, Shapes):
    img_lightned = img_yuv.copy()
    for S in Shapes:
        mask = S.get_object_mask()
        lighting_level = S.get_extra_luma()
        img_lightned[:, :, 0] += (lighting_level*mask).astype(np.uint8)

        saturation_pixels = img_lightned[:, :, 0] < img_yuv[:, :, 0]
        img_lightned[saturation_pixels == 1, 0] = 255

    img_lightned = cv2.cvtColor(img_lightned, cv2.COLOR_YUV2RGB)
    return img_lightned
