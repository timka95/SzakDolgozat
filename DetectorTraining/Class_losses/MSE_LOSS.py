
import torch.nn as nn
import cv2
import numpy as np

class MseLoss():

    def __init__(self):
        pass

    def mseloss(self, line_detected , line_GT, imagename):
        line_GT = line_GT.float()
        line_detected = line_detected.float()
        normalized_image = (line_detected-line_detected.min()) / (line_detected.max() - line_detected.min())
        norm_line_GT = (line_GT - line_GT.min()) / (line_GT.max() - line_GT.min())

        mse = nn.MSELoss()
        mse_loss = mse(normalized_image, norm_line_GT)

        #print("MSE VALUES")
        #print(normalized_image.min(), normalized_image.max(),  self.line_GT.min(),  self.line_GT.max())
        PathSave = "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/IMAGES/MSE/"
        normalized_image =  normalized_image.cpu().detach().numpy()
        norm_line_GT = norm_line_GT.cpu().detach().numpy()

        resized_image = np.squeeze(normalized_image)
        resized_gt = np.squeeze(norm_line_GT)

        cv2.imwrite(PathSave + imagename + '.png',  resized_image*255)
        cv2.imwrite(PathSave + imagename + '_GT.png',  resized_gt*255)

      

        return mse_loss



