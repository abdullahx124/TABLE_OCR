from table_detection import table_detection_main
from table_structure import table_structure_main
from pdf_processing import PDFConverter
import os
from preprocessing import darken_text_and_lines, edge_detection_using_canny
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
from PIL import Image


def apply_ocr(cell_coordinates, cropped_table):
    reader = easyocr.Reader(['en'], gpu=True)  # this needs to run only once to load the model into memory

    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
      row_text = []
      for cell in row["cells"]:
        # crop cell out of image
        cell_image = np.array(cropped_table.crop(cell["cell"]))
        # apply OCR
        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          # print([x[1] for x in list(result)])
          text = " ".join([x[1] for x in result])
          row_text.append(text)

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


def add_serial_number_with_header(data):
    header_row = data[0]
    # Adding "SR" header to the first row
    header_row.insert(0, "SR")

    data_with_serial = [header_row]  # Initialize with the modified header row

    # Iterating through rows starting from the second row
    for index, row_data in enumerate(data[1:], start=1):
        row_data.insert(0, f"{index}")
        data_with_serial.append(row_data)

    return data_with_serial

def ACPL_function(pdf_name, condition):
    pdf_path = pdf_name
    print(pdf_path)
    pdf_converter = PDFConverter(pdf_path, dpi=300)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(pdf_name)
    output_folder = f"./CSV_Output/{pdf_name}"
    os.makedirs(output_folder, exist_ok=True)
    # print(pdf_name)
    images_list = pdf_converter.pdf_to_images()
    # print(images_list)
    cropped_table_list = table_detection_main(images_list,condition)
    # print(cropped_table_list)
    processed_list = []

    for i in cropped_table_list:
        process_1 = darken_text_and_lines(i, 1.0)
        # print(process_1)
        process_2 = edge_detection_using_canny(process_1)
        # print(process_2)
        process_2 = Image.fromarray(process_2)
        processed_list.append(process_2)
    # print(processed_list)
    cell_coordinates, cropped_table_list = table_structure_main(processed_list,condition)
    # print(cell_coordinates)
    idx = 0
    for x, y in zip(cell_coordinates, cropped_table_list):
        data = apply_ocr(x, y)
        data_1 = []
        for row, row_data in data.items():
            data_1.append(row_data)

        data_with_serial_and_header = add_serial_number_with_header(data_1)
        # print(data_with_serial_and_header)

        import csv
        filename = os.path.join(output_folder, f'output_{idx}.csv')
        with open(filename, 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row_text in data_with_serial_and_header:
                wr.writerow(row_text)

        idx = idx + 1
