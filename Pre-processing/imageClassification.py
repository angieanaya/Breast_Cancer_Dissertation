'''
@File   :   imageClassification.py
@Date   :   30/09/2022
@Author :   María de los Ángeles Contreras Anaya
@Version:   2.0
@Desc:   Program that separates CESM images into their respective folder according to the classification (benign, malignant or normal).

Disclaimer: The folder structure should be create, this program only saves them in their corresponding paths.
'''

import pandas as pd
import shutil

findings_doc_path = "Detail/Findings.xlsx"

# read data from Excel file
excel_data = pd.read_excel(findings_doc_path, sheet_name="all") 
df = excel_data[["Image_name", "Type", "Pathology Classification/ Follow up"]]
for i in range(len(df)):
    type = df.iloc[i]['Type']
    # save every CESM image in the correct folder
    if(type == "CESM"):
        image_name = df.iloc[i]['Image_name'].strip()
        classification = df.iloc[i]['Pathology Classification/ Follow up']
        image = type + "/" + image_name + ".jpg"
        # copy image to the correct path based on pathology classification
        shutil.copy2(image, classification + "/" + image_name + ".jpg")