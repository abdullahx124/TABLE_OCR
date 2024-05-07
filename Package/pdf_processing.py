import fitz
from PIL import Image
import io
import os

class PDFConverter:
    def __init__(self, pdf_path, dpi=300):
        self.pdf_path = pdf_path
        self.dpi = dpi

    def pdf_to_images(self):
        images = []

        # Open the PDF file
        pdf_document = fitz.open(self.pdf_path)

        for page_number in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_number]

            # Get the pixmap
            pixmap = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))

            # Convert the pixmap to an image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

            # Append the image to the list
            images.append(img)

        # Close the PDF file
        pdf_document.close()

        return images

# Example usage
if __name__ == "__main__":
    pdf_path = './data/November/piping eqp.pdf'

    pdf_converter = PDFConverter(pdf_path, dpi=300)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(pdf_name)

    images_list = pdf_converter.pdf_to_images()
