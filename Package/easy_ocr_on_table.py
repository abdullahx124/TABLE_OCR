import numpy as np
import cv2
from tqdm.auto import tqdm
import easyocr

reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
reader.recognize_by_word = True  # Enable recognition confidence at the word level

def apply_ocr(cell_coordinates,crop_val):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(crop_val.crop(cell["cell"]))
            # apply OCR
            try:
                result = reader.readtext(cell_image)
                # process result
                text = " ".join([x[1] for x in result])
                row_text.append(text)
            except Exception as e:
                print(f"OCR failed for cell: {cell}, Error: {str(e)}")

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

def easy_ocr_main(cell_coordinates, cropped_table_list):

    data = []
    for val, crop_val in zip(cell_coordinates, cropped_table_list):
        data.append(apply_ocr(val, crop_val))

    # for row_data in data:
    #     for key, value in row_data.items():
    #         print(value)

    return data
