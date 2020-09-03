import os
import torch 

def save_gt_sample(sample, filename):
    
    f = open (filename, 'w')
    for i in range(sample.size()[2]):
        moc_datum = sample[:, 0, i].cpu().numpy()
        f.write("{0:02d} ".format(i + 1) + str(moc_datum[0]) + " " + str(moc_datum[1]) + " " + str(moc_datum[2]) + "\n")

    f.close()