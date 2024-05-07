from torchvision import transforms
import torch
from model_loading import table_detection_loader


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

detection_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def pixel_values_function(image):
    pixel_values = detection_transform(image).unsqueeze(0)
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


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

cropped_table_list = []
def cropped_table_calling(image, objects):
    tokens = []
    detection_class_thresholds = {
        "table": 0.7,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 10
    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)

    if tables_crops:
        cropped_table = tables_crops[0]['image'].convert("RGB")
        cropped_table_list.append(cropped_table)
    else:
        print("No tables_crops found for the current image.")

# Assuming model, id2label, pixel_values_list, images_list, and objects_to_crops are defined

import torch


def get_detection(pixel_values, image, table_detection_model, id2label):
    with torch.no_grad():
        outputs = table_detection_model(pixel_values)
        objects = outputs_to_objects(outputs, image.size, id2label)
        return objects





def table_detection_main(images_list):

    table_detection_model, id2label = table_detection_loader()

    pixel_values_list = []
    for image in images_list:
        values = pixel_values_function(image)
        pixel_values_list.append(values)

    object_list = []

    # Populate object_list using get_detection
    for pixel_values, image in zip(pixel_values_list, images_list):
        objects = get_detection(pixel_values, image, table_detection_model, id2label)
        object_list.append(objects)



    # Call cropped_table_calling for each image and its corresponding objects
    for image, objects in zip(images_list, object_list):
        cropped_table_calling(image, objects)

    return cropped_table_list
