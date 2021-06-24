import json
import csv
import numpy as np
import math
import os
import cv2

directory = "C:\\Users\\hcaliskan\\Recursive-CNNs\\highres_double-20210617T092601Z-001\\highres_double\\"
# \\GitHub\\DRBox_keras-1\\training_kimlik_new_v1'

ratios = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      input_path = os.path.join(directory, filename)
      img = cv2.imread(input_path, cv2.IMREAD_COLOR)
      ratio_y = 300/img.shape[0]
      ratio_x = 300/img.shape[1]
      img = cv2.resize(img,(300,300))
      cv2.imwrite((directory+'resized_'+filename), img)
      ratios.append([ratio_x,ratio_y])
    else:
      pass




with open('C:\\Users\\hcaliskan\\Recursive-CNNs\\highres_double-20210617T092601Z-001\\highres_double\\highres_double\\annotations.json') as f:
  data = json.load(f)

# print(data)
with open("C:\\Users\\hcaliskan\\Recursive-CNNs\\gt.csv", 'w', newline='') as myfile:

  for img_name in data:
      if img_name == '___sa_version___':
          pass
      else:
        for n in range(0,len(ratios)):
          mylist= [img_name+',',
          (np.asarray([data[img_name]['instances'][7]['x']*ratios[n][0],
          data[img_name]['instances'][7]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][6]['x']*ratios[n][0],
          data[img_name]['instances'][6]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][5]['x']*ratios[n][0],
          data[img_name]['instances'][5]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][4]['x']*ratios[n][0],
          data[img_name]['instances'][4]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][3]['x']*ratios[n][0],
          data[img_name]['instances'][3]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][2]['x']*ratios[n][0],
          data[img_name]['instances'][2]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][1]['x']*ratios[n][0],
          data[img_name]['instances'][1]['y']*ratios[n][1]]),
          np.asarray([data[img_name]['instances'][0]['x']*ratios[n][0],
          data[img_name]['instances'][0]['y']*ratios[n][1]]))]

          wr = csv.writer(myfile, delimiter='|', quoting=csv.QUOTE_NONE,lineterminator = '|\n')
          wr.writerow(mylist)

