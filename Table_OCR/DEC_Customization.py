from table_detection import table_detection_main
from table_structure import table_structure_main
from pdf_processing import PDFConverter
import os
from preprocessing import crop_and_save_images


import re
import numpy as np

from PIL import Image
import pytesseract
from tqdm.auto import tqdm

import numpy as np
import csv
from PIL import Image
import pytesseract
import easyocr
from tqdm.auto import tqdm

# Initialize easyocr reader
reader = easyocr.Reader(['en'], gpu=True)

def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []

        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            cell_image_pil = Image.fromarray(cell_image)

            # Apply OCR using pytesseract
            text_tesseract = pytesseract.image_to_string(cell_image_pil, config='--psm 6')

            # Apply OCR using easyocr
            result_easyocr = reader.readtext(cell_image)
            text_easyocr = " ".join([x[1] for x in result_easyocr])

            # Compare the results and choose the one with more characters
            if len(text_tesseract) > len(text_easyocr):
                text = text_easyocr.strip()
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

def preprocess_text(text):
    # Remove specific escape sequences and replace them with space
    text = re.sub(r'[\\|]', '', str(text))
    text = re.sub(r'[\\/]', '', str(text))
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', str(text))
    text = re.sub(r'\\r|\\n|\r\n', ' ', str(text))
    return text


def add_serial_number_with_header(data):
    header_row = data[0]
    # Adding "SR" header to the first row
    header_row.insert(0, "S.No.")

    data_with_serial = [header_row]  # Initialize with the modified header row

    # Iterating through rows starting from the second row
    for index, row_data in enumerate(data[1:], start=1):
        row_data.insert(0, f"{index}")
        data_with_serial.append(row_data)

    return data_with_serial
def DEC_function(pdf_name, condition):
    pdf_path = pdf_name

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = f"./CSV_Output/{pdf_name}"
    os.makedirs(output_folder, exist_ok=True)
    images_list = []
    images_list.append(crop_and_save_images(pdf_path, dpi=300))
    cropped_table_list = table_detection_main(images_list, condition)
    cell_coordinates, cropped_table_list = table_structure_main(cropped_table_list, condition)
    idx = 0
    for x, y in zip(cell_coordinates, cropped_table_list):
        data = apply_ocr(x, y)
        data_1 = []
        for row, row_data in data.items():
            data_1.append(row_data)
        data_1 = data_1[1:]
        data_with_serial_and_header = add_serial_number_with_header(data_1)
        # print(data_with_serial_and_header)

        import csv
        filename = os.path.join(output_folder, f'output_{idx}.csv')
        with open(filename, 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row_text in data_with_serial_and_header:
                wr.writerow(row_text)

        idx = idx + 1