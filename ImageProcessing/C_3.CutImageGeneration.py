from scipy.io import savemat
import scipy.io
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import ast
import cv2

#INPUT
#data = scipy.io.loadmat('/Users/timeanemet/Desktop/CNN/matfiles/subset_data.mat')
FilePath = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Cutted_SOLD.csv'
imagePath = "/project/ntimea/NewData/IMAGES/data_rect/"
#OUTPUT
#filename = '/project/ntimea/NewData/MATFILES/NewAugment/MateTest.csv'
OutPath = "/project/ntimea/NewData/IMAGES/ImageProcessing/Cutted_SOLD(M)/"

def read_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Get the header
        header = next(csvreader)
        # Create an empty list to store the array_3D
        array_3D = []
        # Iterate over each row in the csv file
        countrow = 0
        for row in csvreader:
            countrow = countrow+1
            # Create an empty dictionary
            row_dict = {}
            # Iterate over each cell in the row and the corresponding header cell
            for cell, header_cell in zip(row, header):
                # Add the cell value to the dictionary with the header cell as the key
                row_dict[header_cell] = cell
            # Add the dictionary to the array_3D list
            array_3D.append([row_dict])
        # Return the array_3D list
        return array_3D