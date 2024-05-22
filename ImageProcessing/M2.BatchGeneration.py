


import scipy.io
import csv
import ast
import numpy as np
import cv2
from PIL import Image
import os


FileName = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/Cutted_Data_img_SOLD_seq05.csv'
FilePath = '/project/ntimea/NewData/MATFILES/NewAugment/NewData/BATCH_Cutted_Data_img_SOLD_seq05.csv'



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
    
#fieldnames = ['ID', '2D_512', '3D', 'ID_orig']
def arrangearray(csvdata):
    for i in range(len(csvdata)):
        csvdata[i] = csvdata[i][0]
        csvdata[i]["ID"] = csvdata[i]["ID"]
        #csvdata[i]["2D_512"] = ast.literal_eval(csvdata[i]["2D_512"])
    return csvdata

def write_data_to_csv(data_array, filename, fieldnames):
   with open(filename, 'w', newline='') as csvfile:
       fieldnames = fieldnames
       writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       writer.writeheader()
       for data in data_array:
           writer.writerow(data)

csvdata = read_csv(FileName)
csvdata = arrangearray(csvdata)

alldata = []
added = 0
dicty = {}

for i in range(len(csvdata)):
    currentid = csvdata[i]["ID"]

    id = csvdata[i]["ID"]
    number = id.split('_')[1]
    id  = number.replace('[', '').replace(']', '')[:-1]
    number  = csvdata[i]["ID"][-2:] 

    currentid = id + number
    #current2d = csvdata[i]["2D_512"]

    if added == 0:
        dicty = {}
        dicty["ID"] = []
        #dicty["2D_512"] = []

    dicty["ID"].append(currentid)
    #dicty["2D_512"].append(current2d)
    added = added + 1

    if added == 15:
        alldata.append(dicty)
        added = 0

fieldnames = ['ID']

write_data_to_csv(alldata, FilePath, fieldnames)
    

    
    
