import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from PIL import Image, ImageDraw, ImageFont
import requests
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_model(model_id="IDEA-Research/grounding-dino-tiny", device=None):
    """Load the model and processor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device

def detect_objects(image, processor, model, text_prompts, device,
                   box_threshold=0.4, text_threshold=0.3):
    """
    Detect objects in an image using Grounding DINO model.
    
    Args:
        image (PIL.Image): The image
        processor: The model processor
        model: The detection model
        text_prompts (list): List of text prompts to detect
        device (str): Device to run model on
        box_threshold (float): Confidence threshold for bounding boxes
        text_threshold (float): Confidence threshold for text
        
    Returns:
        list: List of detection results with boxes, scores, and labels
    """
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
    
    return detections

def visualize_detections(image, detections):
    """
    Visualize detections on the image.
    
    Args:
        image (PIL.Image): The original image
        detections (list): List of detection dictionaries
        
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
    
    return draw_image

def process_directory(img_dir, output_dir, text_prompts, model_id="IDEA-Research/grounding-dino-tiny", 
                     box_threshold=0.4, text_threshold=0.3, device=None):
    """Process all images in a directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    processor, model, device = load_model(model_id, device)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        image_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    
    image_files.sort()  # Sort to ensure consistent order
    
    if not image_files:
        print(f"No image files found in {img_dir}")
        return []
    
    print(f"Processing {len(image_files)} images...")
    result_files = []
    
    # Process each image
    for img_path in tqdm(image_files):
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue
            
        # Get file basename
        basename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"detected_{basename}")
        
        # Detect objects
        detections = detect_objects(
            image, processor, model, text_prompts, device,
            box_threshold=box_threshold, text_threshold=text_threshold
        )
        
        # Visualize detections
        vis_image = visualize_detections(image, detections)
        
        # Save result
        vis_image.save(output_path)
        result_files.append(output_path)
        
    return result_files

def create_video(image_files, output_video_path, fps=5):
    """Create a video from a list of image files."""
    if not image_files:
        print("No images to create video from.")
        return
    
    # Get first image to determine dimensions
    first_img = cv2.imread(image_files[0])
    h, w, layers = first_img.shape
    size = (w, h)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    print(f"Creating video from {len(image_files)} images...")
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        if img is not None and img.shape[0] == h and img.shape[1] == w:
            video.write(img)
    
    # Release video writer
    video.release()
    print(f"Video saved to {output_video_path}")

def create_video_ffmpeg(image_dir, output_video_path, fps=30):
    """
    Create a video from a directory of images using ffmpeg.
    
    Args:
        image_dir: Directory containing the processed images
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    try:
        # Get list of image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print("No images found in the directory")
            return False
        
        # Create a temporary directory for sequential frames
        temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy and rename frames sequentially for ffmpeg
        print("Preparing frames for video...")
        for i, img_file in enumerate(tqdm(image_files, desc="Preparing frames")):
            img_path = os.path.join(image_dir, img_file)
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(frame_path, img)
        
        # Ensure output path has .mp4 extension
        video_path = os.path.splitext(output_video_path)[0] + '.mp4'
        
        # Use ffmpeg to create the video
        print("Creating video with ffmpeg...")
        ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
        os.system(ffmpeg_cmd)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"Successfully created video: {video_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        # Clean up temporary directory if it exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def main():
    parser = argparse.ArgumentParser(description="Object detection on image directory with Grounding DINO")
    parser.add_argument("--img-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", default="detection_results", help="Directory to save results")
    parser.add_argument("--video-path", default="detection_video.mp4", help="Path to save output video")
    parser.add_argument("--prompts", required=True, nargs="+", help="Text prompts to detect")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="Model ID")
    parser.add_argument("--box-threshold", type=float, default=0.4, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Text confidence threshold")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--device", default="cpu", help="Device to use (cuda/cpu)")
    parser.add_argument("--use-ffmpeg", type=bool, default=True, help="Use ffmpeg for video creation (more compatible)")
    
    args = parser.parse_args()
    
    # Process directory
    result_files = process_directory(
        args.img_dir,
        args.output_dir,
        args.prompts,
        model_id=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )

    if args.use_ffmpeg:
        create_video_ffmpeg(args.output_dir, args.video_path, fps=args.fps)
    elif result_files:
        create_video(result_files, args.video_path, fps=args.fps)
        print("Note: If the video cannot be played, try using --use-ffmpeg for better compatibility")
    
if __name__ == "__main__":
    main() 