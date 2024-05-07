from transformers import AutoModelForObjectDetection, AutoConfig, TableTransformerForObjectDetection
import torch

class Table_Detection_ModelLoader:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path


    def load_table_detection_model(self):
        # Load the configuration
        config = AutoConfig.from_pretrained(self.config_path)

        # Load the model with the downloaded configuration
        model = AutoModelForObjectDetection.from_pretrained(self.model_path, config=config)

        # Move the model to the available device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print("Device:", device)

        # Update id2label to include "no object"
        id2label = model.config.id2label
        id2label[len(model.config.id2label)] = "no object"

        return model, id2label



class Table_Sttructure_ModelLoader:
    def __init__(self,local_model_path, config_path_structure):
        self.local_model_path = local_model_path
        self.config_path_structure = config_path_structure

    def load_structure_model(self):
        # Load the configuration
        config_structure = AutoConfig.from_pretrained(self.config_path_structure)

        # Move the model to the available device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model from the local directory
        structure_model = TableTransformerForObjectDetection.from_pretrained(
            self.local_model_path, config=config_structure
        )
        structure_model.to(device)

        # Update structure_id2label to include "no object"
        structure_id2label = structure_model.config.id2label
        structure_id2label[len(structure_model.config.id2label)] = "no object"

        return structure_model, structure_id2label


# Example usage:
def table_detection_loader():
    model_loader = Table_Detection_ModelLoader(
        model_path="../models/table_detection.bin",
        config_path="../models/table_detection_config.json",
    )

    table_detection_model, id2label = model_loader.load_table_detection_model()
    return table_detection_model, id2label


def table_structure_loader():
    model_loader = Table_Sttructure_ModelLoader(
        local_model_path="../models/table_structure.safetensors",
        config_path_structure="../models/structure_config.json",
    )

    structure_model, structure_id2label = model_loader.load_structure_model()
    return structure_model, structure_id2label