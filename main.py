import matplotlib.pyplot as plt
import numpy as np
import cv2

import ColorSegmentation
import vidUtils
import LightShape


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


def light_shape_example(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    shape_masks = ColorSegmentation.split_to_objects_masks(img_yuv)
    Triangle = LightShape.LightShape(shape_masks[0])
    Circle = LightShape.LightShape(shape_masks[1])
    Ex = LightShape.LightShape(shape_masks[2])
    Square = LightShape.LightShape(shape_masks[3])

    _, ax = plt.subplots()
    light_shape(img_yuv, Triangle, ax=ax)
    light_shape(img_yuv, Ex, ax=ax)
    light_shape(img_yuv, Circle, ax=ax)
    light_shape(img_yuv, Square, ax=ax)


def animate_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    shape_masks = ColorSegmentation.split_to_objects_masks(img_yuv)

    Triangle = LightShape.LightShape(shape_masks[0])
    Circle = LightShape.LightShape(shape_masks[1])
    Ex = LightShape.LightShape(shape_masks[2])
    Square = LightShape.LightShape(shape_masks[3])

    Shapes = (Triangle, Circle, Ex, Square)
    Turn_on_time = (0, 16, 7, 22)
    Turn_off_time = (10, 25, 20, 30)

    _, ax = plt.subplots()
    vid_length = 40
    for i in range(vid_length):
        if i in Turn_on_time:
            cur_shape = Turn_on_time.index(i)
            Shapes[cur_shape].change_lighting(LightShape.light_on_sigmoid)
        if i in Turn_off_time:
            cur_shape = Turn_off_time.index(i)
            Shapes[cur_shape].change_lighting(LightShape.light_off_sigmoid)

        for S in Shapes:
            S.update()

        img_lightned = light_all_shapes(img_yuv, Shapes)
        ax.cla()
        ax.imshow(img_lightned)
        plt.pause(0.0001)


def init_video_shapes(vid):
    img = vid[0, :, :, :]
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    shape_masks = ColorSegmentation.split_to_objects_masks(img_yuv)

    Shapes = {}
    Shapes['Triangle'] = LightShape.LightShape(shape_masks[0])
    Shapes['Circle'] = LightShape.LightShape(shape_masks[1])
    Shapes['Ex'] = LightShape.LightShape(shape_masks[2])
    Shapes['Square'] = LightShape.LightShape(shape_masks[3])

    return Shapes


def animate_video(vid):
    frames = []

    shapes = {}
    turn_on_times = {
        'Triangle': [0],
        'Circle': [100],
        'Ex': [70],
        'Square': [150]
    }
    turn_off_times = {
        'Triangle': [110],
        'Circle': [195],
        'Ex': [180],
        'Square': [245]
    }

    _, ax = plt.subplots()
    cycle_length = 250

    shapes = init_video_shapes(vid)

    for frame_ind, frame in enumerate(vid):
        print(f'{frame_ind=}')
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)

        shape_masks = ColorSegmentation.split_to_objects_masks(img_yuv)

        for ind, shape in enumerate(shapes.keys()):
            shapes[shape].update_object_mask(shape_masks[ind])

            if (frame_ind % cycle_length) in turn_on_times[shape]:
                shapes[shape].change_lighting(LightShape.light_on_linear)
            if (frame_ind % cycle_length) in turn_off_times[shape]:
                shapes[shape].change_lighting(LightShape.light_off_linear)

        for S in shapes.values():
            S.update()

        img_lighten = light_all_shapes(img_yuv, shapes.values())
        # ax.cla()
        # ax.imshow(img_lighten)
        # plt.pause(0.000001)

        frames.append(cv2.cvtColor(img_lighten, cv2.COLOR_RGB2BGR))

    return frames


if __name__ == '__main__':
    # example_im_path = r"C:\Users\adirm\Desktop\projects\PS-segmentation\data\image1.jpg"
    # img = cv2.cvtColor(cv2.imread(example_im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # light_shape_example(img)
    # animate_image(img)

    example_vid_path = r"C:\Users\adirm\Desktop\projects\PS-segmentation\data\vid2.mp4"
    output_path = r"C:\Users\adirm\Desktop\projects\PS-segmentation\results\vid2_ema_uncompressed.mp4"

    vid = vidUtils.read_video(example_vid_path)

    frames = animate_video(vid)
    vidUtils.save_video(frames, output_path)
    pass




