import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data import cfg_mnet, cfg_re50
from scipy import stats


parser = argparse.ArgumentParser(description='Plotting and Saving plots')
parser.add_argument('--outputFolder', default="outputs/", help='Output folder directory to save plots')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--IoU_lossFunction', default='bce', help='IoU loss computation "bce" or "mse" [use only for "resnet50"]')
args = parser.parse_args()


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


def plot_pr_curve(pr_curve, group = "", ld_label=""):
    precision = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, precision)
    print(f"{ld_label:10s} {group:6s} : {ap*100:0.4f} %")

    # plot the precision-recall curve
    plt.plot(recall, precision, lw=2, label = ld_label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"PR Curve - {group} Set")
    plt.legend()
    plt.grid()


def plot_pr_curves(pr_cureves, ld_label=""):
    groups = ["Easy", "Medium", "Hard"]
    print()
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plot_pr_curve(pr_cureves[i], groups[i], ld_label)
    
        


def plot_singlePR_comparison(network, basePR_cureves, ourPR_cureves, outputFolder, groupIndex = 2):
    # plotting single PR curve from the validation data. [Easy-0, Medium-1, Hard-2]
    groups = ["Easy", "Medium", "Hard"]

    plot_pr_curve(basePR_cureves[groupIndex], group = groups[groupIndex], ld_label= f"base model-{network}")
    plot_pr_curve(ourPR_cureves[groupIndex], group = groups[groupIndex], ld_label= f"our model-{network}")
    plt.savefig(outputFolder + f'single-PR-curve-{network}.png')
    plt.show()


def plot_allPR_comparisons(network, basePR_cureves, ourPR_cureves, outputFolder):
    # plotting PR curves for each group [Easy, Medium, Hard] of validation data
    plt.figure(figsize=(12,4))
    plot_pr_curves(basePR_cureves, ld_label = f"base model-{network}")
    plot_pr_curves(ourPR_cureves, ld_label = f"our model-{network}")
    plt.savefig(outputFolder + f'All-PR-curves-{network}.png')
    plt.show()






def run_paired_t_test(basePR_cureves, ourPR_cureves):
    groups = ["Easy", "Medium", "Hard"]

    for i in range(3):
        basePrecisions = basePR_cureves[i][:,0]
        ourPrecisions = ourPR_cureves[i][:,0]

        t_value, p_value = stats.ttest_rel(ourPrecisions, basePrecisions)   # paired test
        print(f"For {groups[i]:6s} Set, p-value:", p_value)
        if p_value < 0.05:
            print("Statistically significant difference\n")
        else:
            print("Not statistically significant difference\n")










if __name__ == '__main__':

    outputFolder = args.outputFolder

    #################################### change these two variables for different experiment setup  #########################################################
    #network = "resnet50"        # using backbone netwrok either "resnet50" or "mobile0.25"
    #IoU_lossFunction = "bce"    # using different IoU loss computation "bce" or "mse" > binary cross entropy or mean squared error [only valid for "resnet50"]


    network = args.network        
    IoU_lossFunction = args.IoU_lossFunction

    

    dirname = os.path.dirname(outputFolder)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if network == "mobile0.25":
        cfg = cfg_mnet
        basePRs_filepath = "./weights/Pretrained-models/Pretrained_baseModel_MobileNet.pkl"
        ourPRs_filepath = "./weights/Pretrained-models/Pretrained_ourModel_MobileNet.pkl"
    elif network == "resnet50":
        cfg = cfg_re50
        basePRs_filepath = "./weights/Pretrained-models/Pretrained_baseModel_ResNet50.pkl"
        if IoU_lossFunction == "bce":
            ourPRs_filepath = "./weights/Pretrained-models/Pretrained_ourModel_ResNet50.pkl"
        elif IoU_lossFunction == "mse":
            ourPRs_filepath = "./weights/Pretrained-models/Pretrained_ourModel_ResNet50_mse.pkl"



    
    fh = open(basePRs_filepath, 'rb')
    basePR_cureves = pickle.load(fh)
    fh = open(ourPRs_filepath, 'rb')
    ourPR_cureves = pickle.load(fh)


    if IoU_lossFunction == "bce":
        # plotting single PR curve from the validation data. [Easy-0, Medium-1, Hard-2]
        plot_singlePR_comparison(network, basePR_cureves, ourPR_cureves, outputFolder, groupIndex = 2)

        # plotting PR curves for each group [Easy, Medium, Hard] of validation data
        plot_allPR_comparisons(network, basePR_cureves, ourPR_cureves, outputFolder)

    elif IoU_lossFunction == "mse":
        # plotting single PR curve from the validation data. [Easy-0, Medium-1, Hard-2]
        plot_singlePR_comparison(network + "-mse", basePR_cureves, ourPR_cureves, outputFolder, groupIndex = 2)

        # plotting PR curves for each group [Easy, Medium, Hard] of validation data
        plot_allPR_comparisons(network + "-mse", basePR_cureves, ourPR_cureves, outputFolder)

 

    run_paired_t_test(basePR_cureves, ourPR_cureves)





    


