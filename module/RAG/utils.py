import base64
from io import BytesIO
import os



def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image, mime_type

def make_interleave_content(question, image_path):
    content = []
    text_elem = {
                "type": "text",
                "text": question
            }
    content.append(text_elem)
    if isinstance(image_path, list):
        for i, path in enumerate(image_path):
            base64_image, image_type = encode_image(path) 
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            content.append(image_elem)
    elif image_path is not None:
        base64_image, image_type = encode_image(image_path)
        image_elem = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(image_elem)
    else:
        pass
    return content


def make_interleave_content_old(question, image_path):
    content = []
    text_elem = {
                "type": "text",
                "text": question
            }
    content.append(text_elem)
    if isinstance(image_path, list):
        for i, path in enumerate(image_path):
            image_elem = {
                "type": "image_url",
                "image_url": {
                    "url": str(path)
                }
            }
            content.append(image_elem)
    elif image_path is not None:
   
        image_elem = {
            "type": "image_url",
            "image_url": {
                "url": str(image_path)
            }
        }
        content.append(image_elem)
    else:
        pass
    return content
