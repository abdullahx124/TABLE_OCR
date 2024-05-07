import os
from PIL import Image
from table_detection import table_detection_main
from table_structure import table_structure_main



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


def Image_function(image_name, condition):
    image_path = image_name

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = f"./CSV_Output/{image_name}"
    os.makedirs(output_folder, exist_ok=True)
    # print(pdf_name)
    images_list = []
    images_list.append(Image.open(image_path).convert("RGB"))
    cropped_table_list = table_detection_main(images_list, condition)
    cell_coordinates, cropped_table_list = table_structure_main(cropped_table_list, condition)
    idx = 0
    for x, y in zip(cell_coordinates, cropped_table_list):
        data = apply_ocr(x, y)
        data_1 = []
        for row, row_data in data.items():
            data_1.append(row_data)

        import csv
        filename = os.path.join(output_folder, f'output_{idx}.csv')
        with open(filename, 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row_text in data_1:
                wr.writerow(row_text)

        idx = idx + 1
