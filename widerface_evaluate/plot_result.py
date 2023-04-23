import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

cfg = {}
cfg['loc_weight']= 2.0


def plot_pr_curves(pr_cureves):

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

pr_filepath = "./widerface_evaluate/widerface_txt/data-IoU-b20.pkl"
with open(pr_filepath, 'rb') as fh:
    pr_cureves = pickle.load(fh)
    plot_pr_curves(pr_cureves)



def plot_learning_curve(total_epochs, total_epochIter, lines):
    indx = 0
    loss_list = []
    for i in range(total_epochs):
        SUM = []
        for iter in range(total_epochIter):
            aline = lines[indx]
            words = aline.split('||')

            epoch = words[0].strip()
            epoch = int(epoch[6:epoch.find("/")])
            #print(epoch)

            losses = words[3].strip().split()
            loss_l, loss_c, loss_landm = float(losses[1]), float(losses[3]), float(losses[5])
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            if loss < np.inf:
                SUM.append(loss)
            #print(losses, loss)
            

            indx += 1
        loss_list.append(sum(SUM)/len(SUM))
        #print(epoch, loss_list[-1])

    #print(loss_list)
    plt.plot(loss_list)
    plt.xlabel('no. of epoches')
    plt.ylabel('loss')


train_filepath = "./widerface_evaluate/widerface_txt/train.o4597559"
with open(train_filepath, "r") as fh:
    lines = fh.readlines()
    print("Total lines:", len(lines))
    aline = lines[0]
    print(aline)

    words = aline.split('||')
    epoch = words[0].strip()
    total_epochs = int(epoch[epoch.find("/")+1:])

    epochIter = words[1].strip()
    total_epochIter = int(epochIter[epochIter.find("/")+1:])
    #print("total_epochs:", total_epochs, "total_epochIter:", total_epochIter)

    plot_learning_curve(total_epochs, total_epochIter, lines)
    plt.show()



