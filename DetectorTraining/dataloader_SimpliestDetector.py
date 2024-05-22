
import numpy as np
import scipy.io
import os
import shutil
from PIL import Image, ImageOps
import PIL
import torch
import ast
import cv2
import csv


class DataLoaderMatrix():
    def __init__(self, batchsizenotall = None):

        # self.mat_file_path = "/project/ntimea/NewData/MATFILES/DetectorBatch.csv"
        # self.GT_images_folder_path = '/project/zenab/zzz/GT_images/' #Path to the Images Ground Truth
        # self.images_folder_path = '/project/zenab/zzz/images/' #Path To the images

        self.mat_file_path = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/BATCH_Cutted_Data_img_SOLD_seq05.csv'
        self.GT_images_folder_path = '/project/ntimea/NewData/IMAGES/ImageProcessing/Cutted_Data_img_SOLD_seq05/GT_512' #Path to the Images Ground Truth
        self.images_folder_path = '/project/ntimea/NewData/IMAGES/ImageProcessing/Cutted_Data_img_SOLD_seq05/Images_512' #Path To the images

        csvdata = self.read_csv(self.mat_file_path)
        self.csvdata = self.arrangecsvdata(csvdata)

        if batchsizenotall != None:
            self.batchsize = batchsizenotall
        else:
            self.batchsize = len(self.csvdata)

    def batchnumber(self):
        return self.batchsize
    
    def read_csv(self, file_path):
        # Read the CSV file
        with open(file_path, 'r') as csvfile:
            # Create a CSV reader object
            csvreader = csv.reader(csvfile)
            # Get the header
            header = next(csvreader)
            # Create an empty list to store the csvdata
            csvdata = []
            # Iterate over each row in the csv file
            for row in csvreader:
                # Create an empty dictionary
                row_dict = {}
                # Iterate over each cell in the row and the corresponding header cell
                for cell, header_cell in zip(row, header):
                    # Add the cell value to the dictionary with the header cell as the key
                    row_dict[header_cell] = cell
                # Add the dictionary to the csvdata list
                csvdata.append([row_dict])
            # Return the csvdata list
            return csvdata
        
    def arrangecsvdata(self,csvdata):
        for i in range(len(csvdata)):
            csvdata[i] = csvdata[i][0]
 #           csvdata[i]["2D_512"] = ast.literal_eval(csvdata[i]["2D_512"])
            # csvdata[i]["2D_376"] = ast.literal_eval(csvdata[i]["2D_376"])
            # csvdata[i]["2D_orig"] = ast.literal_eval(csvdata[i]["2D_orig"])
            # csvdata[i]["3D"] = ast.literal_eval(csvdata[i]["3D"])
            # csvdata[i]["cuttedhere"] = ast.literal_eval(csvdata[i]["cuttedhere"])
        return csvdata
    
    def find_image(self, folder_path, image_name):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith(image_name) and file.endswith('.png'):
                    return os.path.join(root, file)
        return None

    def imagetotorch(self, image_path):
        img = PIL.Image.open(image_path)
        img = ImageOps.grayscale(img)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np)
        img_out = img_tensor.unsqueeze(0)
        img = img_out.to(torch.float32)
        return img
    
    def detectorarray(self,Batchnumber):

        ThisBatch = self.csvdata[Batchnumber]
        imagesdicty = {}
        batchdictionary = {}
        images = []
        imagenames = []

        ImageName_arr = ThisBatch["ID"]
        ImageName_arr = ast.literal_eval(ImageName_arr)

        for Image in range(len(ImageName_arr)):

            ImageName = ImageName_arr[Image]

            # id = ImageName
            # number = id.split('_')[1]
            # id  = number.replace('[', '').replace(']', '')[:-1]

            # ImageName = id + number

            image_path = self.find_image(self.images_folder_path, ImageName)
            image = self.imagetotorch(image_path)

            GT_image_path = self.find_image(self.GT_images_folder_path, ImageName)
            GT_image = self.imagetotorch(GT_image_path)
            GT_image = (GT_image - GT_image.min()) / (GT_image.max() - GT_image.min())


            imagesdicty = {}
            imagesdicty['image'] = image
            imagesdicty['GT'] = GT_image
            images.append(imagesdicty)
            imagenames.append(ImageName)


        batchdictionary["images"] = images
        batchdictionary["imagenames"] = imagenames
        
        return batchdictionary
    




