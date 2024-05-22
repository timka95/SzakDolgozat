# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use
import torch
import torch.nn as nn

import torch.nn.functional as F

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import cv2







class MSELoss_new(nn.Module):
    """ Try to make the repeatability repeatable from one image to the other.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.name = 'difference'

    def forward(self, lines, hough_lines, lines_gt, hough_gt, batches_size=1):
        lines = (lines - lines.min()) / (lines.max() - lines.min())
        hough_lines = (hough_lines - hough_lines.min()) / (hough_lines.max() - hough_lines.min())

        # THIS IS THE SAME AS:
        mse = nn.MSELoss()
        mse_hough = mse(hough_lines, hough_gt)
        mse_lines = mse(lines, lines_gt)




        #######################################IMAGE################################

        # # Assuming your tensor is named 'hough_lines'
        # data_numpy = normalized_hough_line.detach().cpu().numpy()
        # # Save the NumPy array as an image
        # #plt.imsave("output_image_normalized_2.png", data_numpy[0, 0], cmap='viridis')
        #
        # # Create a figure with a subplot
        # fig, ax = plt.subplots(figsize=(8, 6))
        #
        # # Display the data as an image
        # cax = ax.imshow(data_numpy[0, 0], cmap='viridis', aspect='auto')
        #
        # # Add a colorbar
        # cbar = fig.colorbar(cax)
        #
        # # Save the image
        # plt.savefig("output_image_houghlines.png")
        #
        # # Assuming your tensor is named 'hough_lines'
        # data_numpy = hough_gt.detach().cpu().numpy()
        #
        # # Create a figure with a subplot
        # fig, ax = plt.subplots(figsize=(8, 6))
        #
        # # Display the data as an image
        # cax = ax.imshow(data_numpy[0, 0], cmap='viridis', aspect='auto')
        #
        # # Add a colorbar
        # cbar = fig.colorbar(cax)
        #
        # # Save the image
        # plt.savefig("output_image_houghgt.png")

        #######################################IMAGE END################################

        # kl_loss = nn.KLDivLoss(reduction="batchmean")
        # #hough_kl_mean = kl_loss(normalized_hough_line, hough_gt)
        # hough_kl_mean = (hough_gt*(hough_gt.log()-normalized_hough_line.log())).sum()

        # hough_croscor_mean = compute_loss(normalized_hough_line, hough_gt)
        #

        return mse_lines, mse_hough


    # def forward(self, lines,hough_lines,lines_gt,hough_gt,batches_size=1):
    #
    #     # def compute_loss(A, B):
    #     #     # Calculate mean and standard deviation along the specified axes
    #     #     meanA = torch.mean(A, dim=(2, 3), keepdim=True)
    #     #     meanB = torch.mean(B, dim=(2, 3), keepdim=True)
    #     #     nA = torch.std(A, dim=(2, 3), keepdim=True)
    #     #     nB = torch.std(B, dim=(2, 3), keepdim=True)
    #     #
    #     #     # Compute the cross-correlation loss
    #     #     loss = -torch.sum((A - meanA) / nA * (B - meanB) / nB)
    #     #
    #     #     return loss
    #
    #
    #     im_min = lines.view(batches_size, 1, -1).min(2)[0].view(batches_size,1,1,1)
    #     im_max = lines.view(batches_size, 1, -1).max(2)[0].view(batches_size,1,1,1)
    #     normalized_line = (lines - im_min) / (im_max - im_min + 1e-16)
    #
    #     # THIS IS THE SAME AS: mse_2 = nn.MSELoss(), mse_2_mean = mse_2(normalized_hough_line, hough_gt)
    #     mse_mean = ((normalized_line - lines_gt) ** 2).view(batches_size, 1, -1).mean(2).mean()
    #
    #     # hough_im_min = normalized_hough_line.view(batches_size, 1, -1).min(2)[0].view(batches_size, 1, 1, 1)
    #     # hough_im_max = normalized_hough_line.view(batches_size, 1, -1).max(2)[0].view(batches_size, 1, 1, 1)
    #     # normalized_hough_line = (normalized_hough_line - hough_im_min) / (hough_im_max - hough_im_min + 1e-16)
    #
    #     min_value = hough_lines.min()
    #     shifted_tensor = hough_lines - min_value
    #
    #     # Normalize the shifted values to be between 0 and 1
    #     max_value = shifted_tensor.max()
    #     normalized_tensor = shifted_tensor / max_value
    #
    #     # Adjust the normalized values to ensure their sum is 1
    #     sum_of_values = normalized_tensor.sum()
    #     normalized_hough_line = normalized_tensor/sum_of_values
    #
    #
    #
    #
    #
    #     sum_of_values_GT = hough_gt.sum()
    #     hough_gt = hough_gt / sum_of_values_GT
    #
    #
    #
    #     # min_value = torch.min(hough_gt)
    #     # max_value = torch.max(hough_gt)
    #     #
    #     #
    #     # min_value = torch.min(normalized_hough_line)
    #     # max_value = torch.max(normalized_hough_line)
    #
    #
    #
    #
    #     mse_2 = nn.MSELoss()
    #     hough_mse_mean = mse_2(normalized_hough_line, hough_gt)
    #
    #     # Assuming your array is named normalized_kl_slice
    #
    #
    #
    #     #hough_kl_mean = sum(rel_entr(hough_gt, hough_lines))
    #
    #     #######################################IMAGE################################
    #
    #     # # Assuming your tensor is named 'hough_lines'
    #     # data_numpy = normalized_hough_line.detach().cpu().numpy()
    #     # # Save the NumPy array as an image
    #     # #plt.imsave("output_image_normalized_2.png", data_numpy[0, 0], cmap='viridis')
    #     #
    #     # # Create a figure with a subplot
    #     # fig, ax = plt.subplots(figsize=(8, 6))
    #     #
    #     # # Display the data as an image
    #     # cax = ax.imshow(data_numpy[0, 0], cmap='viridis', aspect='auto')
    #     #
    #     # # Add a colorbar
    #     # cbar = fig.colorbar(cax)
    #     #
    #     # # Save the image
    #     # plt.savefig("output_image_houghlines.png")
    #     #
    #     # # Assuming your tensor is named 'hough_lines'
    #     # data_numpy = hough_gt.detach().cpu().numpy()
    #     #
    #     # # Create a figure with a subplot
    #     # fig, ax = plt.subplots(figsize=(8, 6))
    #     #
    #     # # Display the data as an image
    #     # cax = ax.imshow(data_numpy[0, 0], cmap='viridis', aspect='auto')
    #     #
    #     # # Add a colorbar
    #     # cbar = fig.colorbar(cax)
    #     #
    #     # # Save the image
    #     # plt.savefig("output_image_houghgt.png")
    #
    #     #######################################IMAGE END################################
    #
    #
    #
    #     # kl_loss = nn.KLDivLoss(reduction="batchmean")
    #     # #hough_kl_mean = kl_loss(normalized_hough_line, hough_gt)
    #     # hough_kl_mean = (hough_gt*(hough_gt.log()-normalized_hough_line.log())).sum()
    #
    #     # hough_croscor_mean = compute_loss(normalized_hough_line, hough_gt)
    #     #
    #
    #
    #     return mse_mean, hough_mse_mean
        # plt.imshow(normalized_line[0,0,:,:].cpu().detach().numpy(), cmap='gray')


        # normalized_pred1 = (lines - lines.min()) / (lines.max() - lines.min() + 1e-16)
        # l = (normalized_pred1 - lines_gt) ** 2
        # # l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        # l = l.mean()


        # sum_scc = 0
        # for k in range(10):
        #     l1 = (normalized_line[k] - lines_gt[k]) ** 2
        #     l1 = l1.mean()
        #     sum_scc = sum_scc + l1
        #
        # mm = sum_scc / 10
