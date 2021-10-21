'''
The script does inference (semantic segmentation) on arbitrary images.
Just drop some JPG files into demo_dir and run the script.
Results will be written into the same folder.
For better results channel_means better be recalculated I suppose. But it is kinda tricky.
'''
from os import path as osp
from glob import glob
import numpy as np

from model import DeepLab
from utils import (save_load_means, subtract_channel_means, single_demo, read_image, label_to_color_image)
from PIL import Image
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':

    demo_dir = 'data/demos/deeplab/resnet_50_voc2012/'
    models_dir = 'data/models/deeplab/resnet_50_voc2012/'
    model_filename = 'resnet_50_0.7058.ckpt'

    channel_means = save_load_means(means_filename='channel_means.npz',image_filenames=None, recalculate=False)

    deeplab = DeepLab('resnet_50', training=False)
    deeplab.load(osp.join(models_dir, model_filename))
    files = glob(demo_dir+'*.jpg')
    predictions = []
    for image_filename in files:
        filename=osp.basename(image_filename).split('.')[0]
        image =  read_image(image_filename=image_filename)
        image_input = subtract_channel_means(image=image, channel_means=channel_means)
        output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]
        # single_demo(image, np.argmax(output, axis=-1), demo_dir, filename)
        colored_label = label_to_color_image(np.argmax(output, axis=-1))
        label_image = Image.fromarray(colored_label.astype(dtype=np.uint8))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.uint8)
        is_bg = np.tile(np.all(colored_label == (0, 0, 0), axis=-1), (3, 1, 1)).transpose(1, 2, 0)
        colored_mask_no_bg = np.where(is_bg, image, colored_label).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask_no_bg, 0.65, 0)

        predictions += [(image, colored_label, overlay)]

    plt.axis("off")  # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    fig = plt.figure(figsize=(32, 32))
    columns = len(predictions[0])
    rows = len(predictions)
    for r, data in enumerate(predictions):
        for c, img in enumerate(data):
            fig.add_subplot(rows, columns, r * columns + c + 1)
            plt.imshow(img)
    plt.savefig('pred.pdf')
    deeplab.close()
