from model_loading_file import table_structure_loader
from torchvision import transforms
import torch
from table_detection import table_detection_main

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

structure_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pixel_values_function(image):
    pixel_values = structure_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def get_structure(pixel_values, image, structure_model, structure_id2label):
    with torch.no_grad():
        outputs = structure_model(pixel_values)
        objects = outputs_to_objects(outputs, image.size, structure_id2label)
        return objects


def get_cell_coordinates_by_row(table_data, condition = "default"):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        if condition == "default" or condition == "NICL" or condition == "Valves" or condition == "Piping" or condition == "Image":
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        elif condition == "ACPL" or condition == "Collective" or condition == "CPB" or condition == "DEC" or condition == "Service":
            for column in columns[1:]:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates



def table_structure_main(cropped_table_list, condition = "default"):

    structure_model, structure_id2label = table_structure_loader()

    pixel_values_list_two = []
    for image in cropped_table_list:
        values = pixel_values_function(image)
        pixel_values_list_two.append(values)

    cells = []

    # Populate object_list using get_detection
    for pixel_values, image in zip(pixel_values_list_two, cropped_table_list):
        objects = get_structure(pixel_values, image, structure_model, structure_id2label)
        cells.append(objects)

    cell_coordinates = []
    for cell in cells:
        res = get_cell_coordinates_by_row(cell,condition)
        cell_coordinates.append(res)

    return cell_coordinates, cropped_table_list