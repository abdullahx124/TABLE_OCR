from pdf_processing import PDFConverter
import os
from table_detection import table_detection_main
from table_structure import table_structure_main
from easy_ocr_on_table import easy_ocr_main
from csv_file import csv_file_main
import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a PDF file and extract tables using OCR.')
    parser.add_argument('pdf_name', type=str, help='Name of the PDF file to process')

    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()

    # Create the PDF path using the provided PDF name
    pdf_path = f'{args.pdf_name}'

    # Create an instance of PDFConverter
    pdf_converter = PDFConverter(pdf_path, dpi=300)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(pdf_name)

    # Use the pdf_to_images method to get a list of images
    images_list = pdf_converter.pdf_to_images()
    cropped_table_list = table_detection_main(images_list)
    cell_coordinates, cropped_table_list = table_structure_main(cropped_table_list)
    data = easy_ocr_main(cell_coordinates, cropped_table_list)
    csv_file_main(pdf_name, data)


#
# if __name__ == '__main__':
#
#     # Create an instance of PDFConverter
#     pdf_path = './data/November/38848.pdf'
#     pdf_converter = PDFConverter(pdf_path, dpi=300)
#     pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     print(pdf_name)
#     # Use the pdf_to_images method to get a list of images
#     images_list = pdf_converter.pdf_to_images()
#     cropped_table_list = table_detection_main(images_list)
#     cell_coordinates, cropped_table_list = table_structure_main(cropped_table_list)
#     data = easy_ocr_main(cell_coordinates, cropped_table_list)
#     csv_file_main(pdf_name, data)





