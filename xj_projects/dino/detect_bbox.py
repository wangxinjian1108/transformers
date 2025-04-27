import argparse
import requests
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def detect_objects(image_path, text_prompts, model_id="IDEA-Research/grounding-dino-tiny", 
                   box_threshold=0.4, text_threshold=0.3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Detect objects in an image using Grounding DINO model.
    
    Args:
        image_path (str): Path to local image or URL
        text_prompts (list): List of text prompts to detect, e.g. ["a cat", "a remote control"]
        model_id (str): Model ID from Hugging Face
        box_threshold (float): Confidence threshold for bounding boxes
        text_threshold (float): Confidence threshold for text
        device (str): Device to run model on (cuda or cpu)
        
    Returns:
        tuple: (list of detection results with boxes, scores, and labels, PIL Image object)
    """
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # Load image
    if image_path.startswith(("http://", "https://")):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    
    # Format text prompts
    text_labels = [text_prompts]
    
    # Process inputs
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    # Format results
    detections = []
    result = results[0]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        detections.append({
            "box": box,
            "score": round(score.item(), 3),
            "label": label
        })
    
    return detections, image

def visualize_detections(image, detections, output_path=None, show=True):
    """
    Visualize detections on the image.
    
    Args:
        image (PIL.Image): The original image
        detections (list): List of detection dictionaries with 'box', 'score', 'label'
        output_path (str, optional): Path to save the output image
        show (bool): Whether to display the image
        
    Returns:
        PIL.Image: The image with visualized detections
    """
    # Create a copy of the image for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Generate colors for different labels
    unique_labels = list(set(d["label"] for d in detections))
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels) + 1))
    color_map = {label: tuple(int(c * 255) for c in colors[i][:3]) for i, label in enumerate(unique_labels)}
    
    # Draw each detection
    for detection in detections:
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        color = color_map.get(label, (255, 0, 0))
        
        # Convert box [x1, y1, x2, y2] format
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        text = f"{label}: {score:.2f}"
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    
    # Save the image if specified
    if output_path:
        draw_image.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Show the image if requested
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(draw_image))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return draw_image

def main():
    parser = argparse.ArgumentParser(description="Object detection with Grounding DINO")
    parser.add_argument("--image", required=True, help="Path to image file or URL")
    parser.add_argument("--prompts", required=True, nargs="+", help="Text prompts to detect")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="Model ID")
    parser.add_argument("--box-threshold", type=float, default=0.4, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Text confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output", help="Path to save the output image with detections")
    parser.add_argument("--no-display", action="store_true", help="Do not display the result")
    
    args = parser.parse_args()
    
    detections, image = detect_objects(
        args.image, 
        args.prompts, 
        model_id=args.model,
        box_threshold=args.box_threshold, 
        text_threshold=args.text_threshold,
        device=args.device
    )
    
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"Detected {det['label']} with confidence {det['score']} at location {det['box']}")
    
    # Visualize detections
    visualize_detections(
        image, 
        detections, 
        output_path=args.output,
        show=not args.no_display
    )
    
    return detections

if __name__ == "__main__":
    main() 