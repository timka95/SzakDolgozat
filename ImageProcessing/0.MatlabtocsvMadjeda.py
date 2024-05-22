import scipy.io
import csv
import ast
import numpy as np


#INPUT
# MatFile = "/Volumes/TIMKA/NewData/MATFILES/Data_img.mat"
MatFile = "/projnas/project/madjeda/MAIN_TRAINING_DATA_PERSPECTIVE/DataBase/DataBase_360/all_ID_SOLD_seq05/Data_img.mat"

#OUTPU
#FileName = "/Volumes/TIMKA/NewData/MATFILES/Data_img_ORIG.csv"
FileName = "/project/ntimea/NewData/MATFILES/Data_img_SOLD_seq05.csv"




data = scipy.io.loadmat(MatFile)
osszesitett_data = data['Data']


dicty = {}
array = []


for i in range(osszesitett_data.shape[0]):
   #extracted_strings = [item[0] for sublist in osszesitett_data[i, 0] for item in sublist]
   print(i, "---", osszesitett_data.shape[0])
   dicty = {}
   dicty["ID"] = osszesitett_data[i, 0]
   extracted_strings = [item[0] for sublist in osszesitett_data[i, 1] for item in sublist]
   dicty["id_2D"] = extracted_strings
   extracted_strings = [item[0] for sublist in osszesitett_data[i, 2] for item in sublist]
   dicty["id_3D"] = extracted_strings


   data2d = osszesitett_data[i, 3]
   data2did = osszesitett_data[i, 2]


   extracted_arrays = [item[0].tolist() for sublist in osszesitett_data[i, 3] for item in sublist]
   dicty["2D"] = extracted_arrays
   extracted_arrays = [item[0].tolist() for sublist in osszesitett_data[i, 4] for item in sublist]
   dicty["3D"] = extracted_arrays
   PosetoWorld = osszesitett_data[i, 5].tolist()
   dicty["PosetoWorld"] = PosetoWorld
   Camera = osszesitett_data[i, 6].tolist()
   dicty["PosetoCamera"] = Camera
   Calib = osszesitett_data[i, 7].tolist()
   dicty["CalibMatrix"] = Calib


   array.append(dicty)




def write_data_to_csv(data_array, filename):
   with open(filename, 'w', newline='') as csvfile:
       fieldnames = ['ID', 'id_2D', 'id_3D', '2D', '3D', 'PosetoWorld', 'PosetoCamera', 'CalibMatrix']
       writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       writer.writeheader()
       for data in data_array:
           writer.writerow(data)




# Replace this line with the actual name of your CSV file




write_data_to_csv(array, FileName)

