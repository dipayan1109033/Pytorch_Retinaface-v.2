import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

cfg = {}
cfg['loc_weight']= 2.0


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


def plot_Evaluation_performance(dirpath, group_indx = 2):
    sets = ["Easy", "Medium", "Hard"]
    #file_list = os.listdir(dirpath)
    file_list = ["mobilenet0.25_epoch_" + str(epoch) + ".pkl" for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]

    AP_list = []
    for i in range(len(file_list)):
        pr_filepath = dirpath + file_list[i]
        with open(pr_filepath, 'rb') as fh:
            pr_cureves = pickle.load(fh)
            pr_curve = pr_cureves[group_indx]

            precision = pr_curve[:, 0]
            recall = pr_curve[:, 1]
            ap = voc_ap(recall, precision)
            AP_list.append(round(100*ap,4))
    
    print(AP_list)
    x = np.arange(1, len(AP_list) + 1)
    labels = [label[15:-4] for label in file_list]
    plt.plot(x, AP_list, '-')
    plt.xticks(x, labels)
    plt.xlabel('Number of epochs ->')
    plt.ylabel('Average Precision (AP) ->')
    plt.title(f"Evaluation Performance [{sets[group_indx]} Set]")
    #plt.show()




def plot_pr_curve(pr_curve, set = "Easy", ld_label="base"):
    precision = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, precision)
    print(f"{set:6s} : {ap*100:0.4f} %")

    # plot the precision-recall curve
    plt.plot(recall, precision, lw=2, label = ld_label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"PR Curve - {set} Set")
    #plt.title(f"PR Curve - {set} Set [AP={ap*100:0.2f}%]")
    plt.legend()
    plt.grid()

def plot_pr_curves(pr_cureves, ld_label="base"):
    #plt.figure(figsize=(6,4))
    sets = ["Easy", "Medium", "Hard"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plot_pr_curve(pr_cureves[i], sets[i], ld_label)
        
    #plt.show()

pr_filepath = "./widerface_evaluate/widerface_txt/pickle_data/mobilenet_base_b100/mobilenet0.25_epoch_80.pkl"
with open(pr_filepath, 'rb') as fh:
    pr_cureves = pickle.load(fh)
    plot_pr_curves(pr_cureves, ld_label = "base")
    #plot_pr_curve(pr_cureves[1], set = "Medium")
    #plt.show()

pr_filepath = "./widerface_evaluate/widerface_txt/pickle_data/mobilenet_IoU_b100/mobilenet0.25_epoch_80.pkl"
with open(pr_filepath, 'rb') as fh:
    pr_cureves = pickle.load(fh)
    plot_pr_curves(pr_cureves, ld_label = "IoU")
    #plot_pr_curve(pr_cureves[1], set = "Medium")
    
    plt.show()


dirpath = "./widerface_evaluate/widerface_txt/pickle_data/mobilenet_base_b100/"
#plot_Evaluation_performance(dirpath, group_indx = 0)
#plot_Evaluation_performance(dirpath, group_indx = 1)
#plot_Evaluation_performance(dirpath, group_indx = 2)

dirpath = "./widerface_evaluate/widerface_txt/pickle_data/mobilenet_IoU_b100/"
#plot_Evaluation_performance(dirpath, group_indx = 0)

plt.legend(["base", "IoU"])
plt.grid()
#plt.show()


















def plot_training_curve(lines):
    total_lines = len(lines)
    epochs_losses = []
    
    line_indx = 0
    while line_indx < total_lines - 1:
        aline = lines[line_indx]
        words = aline.split('||')
        word_epoch = words[0].strip()
        initial_epochs = int(word_epoch[6:word_epoch.find("/")])
        current_epochs = initial_epochs

        lossSUM_list = []
        while current_epochs == initial_epochs:

            losses = words[3].strip().split()
            loss_l, loss_c, loss_landm = float(losses[1]), float(losses[3]), float(losses[5])
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            if loss < np.inf:
                lossSUM_list.append(loss)

            
            line_indx += 1
            if line_indx == total_lines - 1: break

            aline = lines[line_indx]
            words = aline.split('||')
            word_epoch = words[0].strip()
            #print(word_epoch)
            current_epochs = int(word_epoch[6:word_epoch.find("/")])
        
        avg_epoch_loss = sum(lossSUM_list)/len(lossSUM_list)
        epochs_losses.append(avg_epoch_loss)

    print("Total epochs:", len(epochs_losses))
    #print(epochs_losses)
    plt.plot(epochs_losses)
    plt.xlabel('Number of epoches')
    plt.ylabel('Multi-task loss')
    plt.title("Training loss of modified model")
       
def get_training_curve(train_filepath):
    with open(train_filepath, "r") as fh:
        lines = fh.readlines()
        print("Total lines:", len(lines))
        aline = lines[0]
        #print(aline[:-2])

        plot_training_curve(lines)



train_filepath = "./widerface_evaluate/widerface_txt/train_data/resnet50_base_b100/train.o4598961"
train_filepathIoU = "./widerface_evaluate/widerface_txt/train_data/resnet50_IoU_b100/trainIoU.o4598964"

def train_stats():
    #get_training_curve(train_filepath)
    get_training_curve(train_filepathIoU)
    #plt.legend(["base", "IoU"])
    plt.show()
#train_stats()


