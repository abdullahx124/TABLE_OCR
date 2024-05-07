import numpy as np
import cv2
from tqdm.auto import tqdm
import easyocr
from PIL import Image
import pytesseract
from post_processing import add_serial_number_with_header_ACPL


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


def apply_both_ocr(cell_coordinates,crop_val):
    data = dict()
    max_num_columns = 0

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []

        for cell in row["cells"]:
            cell_image = np.array(crop_val.crop(cell["cell"]))
            cell_image_pil = Image.fromarray(cell_image)

            # Apply OCR using pytesseract
            text_tesseract = pytesseract.image_to_string(cell_image_pil, config='--psm 6')

            # Apply OCR using easyocr
            result_easyocr = reader.readtext(cell_image)
            text_easyocr = " ".join([x[1] for x in result_easyocr])

            # Compare the results and choose the one with more characters
            if len(text_tesseract) > len(text_easyocr):
                text = text_tesseract.strip()
            else:
                text = text_easyocr.strip()

            row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # Pad rows to ensure uniformity in the number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + [""] * (max_num_columns - len(row_data))
        data[row] = row_data

    return data

def easy_ocr_main(cell_coordinates, cropped_table_list):

    data = []
    for val, crop_val in zip(cell_coordinates, cropped_table_list):
        data.append(apply_ocr(val, crop_val))


    return data

def both_ocr_main(cell_coordinates, cropped_table_list, condition = "default"):
    if condition == "ACPL":
        data = []
        for val, crop_val in zip(cell_coordinates, cropped_table_list):
            data.append(apply_both_ocr(val, crop_val))

        for i in range(len(data)):
            data_1 = []
            for row, row_data in data[i].items():
                data_1.append(row_data)
                print(row_data)
            data_with_serial_and_header_ACPL = add_serial_number_with_header_ACPL(data_1)

    return data
