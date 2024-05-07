import cv2
import numpy as np

def darken_text_and_lines(image, factor):
    # Convert image to grayscale
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to extract text and lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)

    # Invert the thresholded image
    thresh = 255 - thresh

    # Increase the darkness intensity
    darkened_thresh = cv2.multiply(thresh, factor)

    # Convert the thresholded image back to BGR for bitwise operations
    thresh_bgr = cv2.cvtColor(darkened_thresh.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Bitwise AND operation to darken the text and lines
    darkened_image = cv2.bitwise_and(image, thresh_bgr)

    return darkened_image

def edge_detection_using_canny(darkened_image):

    image = np.array(darkened_image)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find lines in the image using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=700, minLineLength=100, maxLineGap=20)

    # Draw lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return image


import fitz  # PyMuPDF
from PIL import Image
import os


def crop_and_save_images(pdf_path, dpi=300):
    cropped_images = []

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page in the PDF
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Convert the page to an image with the specified DPI
        image = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

        # Create a Pillow image from the raw pixel data
        pillow_image = Image.frombytes("RGB", (image.width, image.height), image.samples)

        # Crop the images according to page number
        if page_number == 0:
            # Crop 145 pixels from the bottom border of the first page
            pillow_image = pillow_image.crop((0, 0, pillow_image.width, pillow_image.height - 145))
        else:
            # Crop 80 pixels from the top border and 145 from the bottom border of other pages
            pillow_image = pillow_image.crop((0, 70, pillow_image.width, pillow_image.height - 145))

        # Append the cropped image to the list
        cropped_images.append(pillow_image)

    # Close the PDF document
    pdf_document.close()

    # Merge the cropped images vertically
    merged_image = merge_images(cropped_images)

    # Save the merged image to the specified folder
    # merged_image.save(os.path.join(image_folder, "merged_image.jpg"))

    # print(f"PDF converted to images, cropped, and merged. Merged image saved in the '{image_folder}' folder.")
    return merged_image


def merge_images(images):
    # Calculate the dimensions of the merged image
    merged_width = max(image.width for image in images)
    merged_height = sum(image.height for image in images)

    # Create a new blank image with the calculated dimensions
    merged_image = Image.new("RGB", (merged_width, merged_height))

    # Paste each image onto the merged image
    y_offset = 0
    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height

    return merged_image