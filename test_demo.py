from os import path as osp

import numpy as np

from model import DeepLab
from tqdm import trange
from utils import (Dataset, Iterator, save_load_means, subtract_channel_means,
                   label_to_color_image, count_label_prediction_matches,
                   mean_intersection_over_union)
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score
import os

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

if __name__ == '__main__':

    data_dir = 'data/datasets/VOCdevkit/VOC2012/'
    testset_filename = osp.join(data_dir, 'ImageSets/Segmentation/val.txt')
    images_dir = osp.join(data_dir, 'JPEGImages/')
    labels_dir = osp.join(data_dir, 'SegmentationClass/')
    demo_dir = 'data/demos/deeplab/resnet_50_voc2012/'
    models_dir = 'data/models/deeplab/resnet_50_voc2012/'
    model_filename = 'resnet_50_0.7058.ckpt'

    channel_means = save_load_means(means_filename='channel_means.npz', image_filenames=None)

    minibatch_size = 16

    test_dataset = Dataset(dataset_filename=testset_filename, images_dir=images_dir, labels_dir=labels_dir, image_extension='.jpg', label_extension='.png')
    test_iterator = Iterator(dataset=test_dataset, minibatch_size=minibatch_size, process_func=None, random_seed=None, scramble=False, num_jobs=1)

    deeplab = DeepLab('resnet_50', training=False)
    deeplab.load(osp.join(models_dir, model_filename))

    n_samples = 50
    fig = plt.figure(figsize=(16, 120))
    fig.tight_layout()
    col_names = ['Image', 'Prediction / mIoU', 'Prediction Overlay / Predicted Classes', 'Ignore Boundary']
    cl = len(col_names)
    for c, title in enumerate(col_names):
        ax = fig.add_subplot(n_samples + 1, cl, c + 1)
        ax.set_axis_off()
        ax.text(0.5, 0.5, col_names[c], fontsize=12, horizontalalignment='center', verticalalignment='center')

    for r in trange(n_samples):
        file, image, label = test_iterator.next_raw_data_with_name()
        image_input = subtract_channel_means(image=image, channel_means=channel_means)

        output = deeplab.test(inputs=[image_input], target_height=image.shape[0], target_width=image.shape[1])[0]

        # validation_single_demo(image, np.squeeze(label, axis=-1), np.argmax(output, axis=-1), demo_dir, str(i))
        colored_label = label_to_color_image(np.argmax(output, axis=-1))
        # label_image = Image.fromarray(colored_label.astype(dtype=np.uint8))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.uint8)
        is_bg = np.tile(np.all(colored_label == (0, 0, 0), axis=-1), (3, 1, 1)).transpose(1, 2, 0)
        colored_mask_no_bg = np.where(is_bg, image, colored_label).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask_no_bg, 0.65, 0)
        pred = np.expand_dims(np.argmax(output, axis=-1), axis=-1)
        pred_classes = np.unique(pred)
        no_classes = pred_classes.shape[0]
        pred_adj = np.where(label == 255, 255, pred)
        print(no_classes, label.shape, pred.shape)
        cv2.imwrite('label.png', label)
        cv2.imwrite('pred.png', pred_adj)

        num_pixels_union, num_pixels_intersection = count_label_prediction_matches(label.ravel(), pred.ravel(), 21, 255)
        mean_iou = round(mean_intersection_over_union(num_pixels_union, num_pixels_intersection), 2)

        iou = round(jaccard_score(label.ravel(), pred_adj.ravel(), average='weighted'), 2)
        ioumacro = round(jaccard_score(label.ravel(), pred_adj.ravel(), average='macro'), 2)
        ioumicro = round(jaccard_score(label.ravel(), pred_adj.ravel(), average='micro'), 2)
        print(mean_iou, iou, ioumicro, os.path.basename(file), np.unique(label.ravel(), return_counts=True), np.unique(pred_adj.ravel(), return_counts=True))
        # print(tf.metrics.mean_iou(tf.convert_to_tensor(label), tf.convert_to_tensor(pred_adj), no_classes))

        cols = [image, colored_label, overlay, np.where(label < 21, colored_label, 255).astype(np.uint8)]
        x_labels = [os.path.basename(file), 'mIoU ' + str(mean_iou), ', '.join([VOC_CLASSES[i] for i in pred_classes]), '']

        for c, img in enumerate(cols):
            ax = fig.add_subplot(n_samples + 1, cl, (r + 1) * cl + c + 1)
            ax.set_xlabel(x_labels[c], {'fontsize': 12})
            ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            plt.imshow(img)

    plt.savefig('pred.pdf')

    deeplab.close()
