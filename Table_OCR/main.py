from ACPL_Customized import ACPL_function
from CPB_Customization import CPB_function
from Collective_Customized import Collective_function
from DEC_Customization import DEC_function
from NICL_Customization import NICL_function
from Service_Customization import Service_function
from Valves_Customization import Valves_function
from Piping_Cusomization import Piping_function
from Image_Customization import Image_function
from pdf_processing import PDFConverter
import os
from table_detection import table_detection_main
from table_structure import table_structure_main
from easy_ocr_on_table import easy_ocr_main
from csv_file import csv_file_main
import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a PDF file and extract tables using OCR.')
    parser.add_argument('--file_name', type=str, help='Name of the file to process')
    parser.add_argument('--customization', type=str, help='Customization if any')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Create the PDF path using the provided PDF name
    file_name = f'{args.file_name}'
    customization = args.customization
    file_type = os.path.splitext(os.path.basename(file_name))[1]
    print(file_name,customization)
    if file_type == ".pdf" and customization == None:
        print("This function is running.")
        pdf_converter = PDFConverter(file_name, dpi=300)
        pdf_name = os.path.splitext(os.path.basename(file_name))[0]

        images_list = pdf_converter.pdf_to_images()
        cropped_table_list = table_detection_main(images_list, "default")
        cell_coordinates, cropped_table_list = table_structure_main(cropped_table_list, "default")
        data = easy_ocr_main(cell_coordinates, cropped_table_list)
        csv_file_main(pdf_name, data)
    elif file_type == ".PNG" or file_type == ".JPG" or file_type == ".JPEG"  and customization == None :
        pdf_path = file_name
        special = "Image"
        Image_function(pdf_path, condition=special)
    elif file_type == ".pdf" and customization == "ACPL":
        pdf_path = file_name
        special = customization
        ACPL_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "Collective":
        pdf_path = file_name
        special = customization
        Collective_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "CPB":
        pdf_path = file_name
        special = customization
        CPB_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "DEC":
        pdf_path = file_name
        special = customization
        DEC_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "NICL":
        pdf_path = file_name
        special = customization
        NICL_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "Piping":
        pdf_path = file_name
        special = customization
        Piping_function(pdf_path,special)
    elif file_type == ".pdf" and  customization == "Service":
        pdf_path = file_name
        special = customization
        Service_function(pdf_path, special)
    elif file_type == ".pdf" and customization == "Valves":
        pdf_path = file_name
        special = customization
        Valves_function(pdf_path,special)


