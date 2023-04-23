import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2



def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


pr_filepath = "./widerface_evaluate/widerface_txt/data.pkl"
with open(pr_filepath, 'rb') as fh:
    pr_cureves = pickle.load(fh)




def plot_pr_curve(pr_curve, set = "Easy"):
    precision = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, precision)
    print(f"{set:6s} : {ap*100:0.4f} %")

    # plot the precision-recall curve
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve' + " - " + set)
    plt.grid()


plt.figure(figsize=(6,4))
sets = ["Easy", "Medium", "Hard"]
for i in range(3):
    #plt.subplot(1, 3, i+1)
    plot_pr_curve(pr_cureves[i], sets[i])
plt.legend(sets)
plt.show()


