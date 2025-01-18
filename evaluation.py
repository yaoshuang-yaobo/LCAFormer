import numpy as np
import cv2
import os
from sklearn.metrics import cohen_kappa_score


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)


def evaluation(hist, n_classes):  # mIoU
    TP = np.diag(hist)
    FP = hist.sum(0) - TP
    FN = hist.sum(1) - TP
    mIOU = TP / (TP + FP + FN)
    Precision = TP/ (TP+FP)
    Recall = TP/ (TP+FN)
    PA = TP.sum(0) / (TP.sum(0) + FP.sum(0))
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    mF1 = F1.sum(0) / n_classes
    Dice = 2*TP/(2*TP+FP+FN)

    n = np.sum(hist)
    sum_po = 0
    sum_pe = 0
    for i in range(len(hist[0])):
        sum_po += hist[i][i]
        row = np.sum(hist[i, :])
        col = np.sum(hist[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    Kappa = (po - pe) / (1 - pe)
    return mIOU, PA, mF1, Precision, Recall, Dice, Kappa


def SegColor2Label(img):
    """
    img: Shape [h, w, 3]
    mapMatrix: color-> label mapping matrix,

    return: labelMatrix: Shape [h, w]
    """
    VOC_COLORMAP = [[255, 255, 255], [0, 0, 0], ]
    # Landslides : [255, 255, 255] white
    # Background : [0, 0, 0] black
    mapMatrix = np.zeros(256 * 256 * 256, dtype=np.int32)
    for i, cm in enumerate(VOC_COLORMAP):
        mapMatrix[cm[2] * 65536 + cm[1] * 256 + cm[0]] = i

    indices = img[:, :, 0] * 65536 + img[:, :, 1] * 256 + img[:, :, 2]
    return mapMatrix[indices]


def Evaluation(test_label_dir,pred_dir,name_index_map,n_classes):

    hist = np.zeros((n_classes, n_classes))
    label_path_lists = os.listdir(test_label_dir)
    for i, label_path_list in enumerate(label_path_lists):
        label = cv2.imread(os.path.join(test_label_dir,label_path_list))
        label = SegColor2Label(label)
        pred = cv2.imread(os.path.join(pred_dir,label_path_list))
        pred = SegColor2Label(pred)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()),
                                                                                  os.path.join(test_label_dir,label_path_list),
                                                                                  os.path.join(pred_dir,label_path_list)))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), n_classes)
        if i > 0 and i % 10 == 0:
            mIOU, PA, F1, Precision, Recall, Dice, Kappa = evaluation(hist, n_classes)
            print('{}'.format(str(round(np.nanmean(Recall), 4))))
    mIoUs, PA , F1, Precision, Recall, Dice, Kappa = evaluation(hist, n_classes)  # mIoU values for all validation set images
    for ind_class in range(n_classes):
        print('===>' + name_index_map[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('===> PA: {}'.format(PA))
    print('===> F1: {}'.format(F1))
    print('===> Precision: {}'.format(Precision))
    print('===> Recall: {}'.format(Recall))
    print('===> Dice: {}'.format(Dice))
    print('===> Kappa: {}'.format(Kappa))
    return mIoUs, PA, F1, Precision, Recall, Dice, Kappa


if __name__ == "__main__":
    test_label_dir= 'data/Landslide4Sense/test/labels/'
    pred_dir = 'Results/LCAFormer/landslide4sense/'

    name_index_map = {0: 'landslide', 1: 'background'}
    Evaluation(test_label_dir, pred_dir, name_index_map, 2)
