import io
from google.cloud import vision

vision_client = vision.ImageAnnotatorClient()

file_name=""


def analyse(file_name):
    with io.open(file_name,"rb") as image_file:
        image = vision_client.annotate_image({'image': {'source': {'image_uri': image_file}},
                                              'features': [{'type': vision.enums.Feature.Type.LABEL_DETECTION}], })

    labels = image.detect_labels()
    Label = []
    for label in labels:
        Label.append(label.description)
    
    return(Label)
