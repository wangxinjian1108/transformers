import argparse
import os
import glob
import sys
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
    """加载模型和处理器"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载模型 {model_id} 到 {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device

def detect_objects(image, processor, model, text_prompts, device,
                   box_threshold=0.4, text_threshold=0.3):
    """
    使用Grounding DINO模型进行目标检测
    
    Args:
        image (PIL.Image): 输入图像
        processor: 模型处理器
        model: 检测模型
        text_prompts (list): 需要检测的文本提示列表
        device (str): 运行模型的设备
        box_threshold (float): 边界框置信度阈值
        text_threshold (float): 文本置信度阈值
        
    Returns:
        list: 包含检测结果的列表（边界框、置信度、标签）
    """
    # 格式化文本提示
    text_labels = " . ".join(text_prompts) + " . "
    
    # 处理输入
    inputs = processor(images=image, text=[text_labels], return_tensors="pt").to(device)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理结果
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    # 格式化结果
    detections = []
    object2nb = {}
    result = results[0]
    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        box = [round(x, 2) for x in box.tolist()]
        detections.append({
            "box": box,
            "score": round(score.item(), 3),
            "label": label
        })
        if label not in object2nb:
            object2nb[label] = 0
        object2nb[label] += 1
    
    return detections, object2nb

def visualize_detections(image, detections):
    """可视化检测结果"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    unique_labels = list(set(d["label"] for d in detections))
    colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels) + 1))
    color_map = {label: tuple(int(c * 255) for c in colors[i][:3]) for i, label in enumerate(unique_labels)}
    
    for detection in detections:
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        color = color_map.get(label, (255, 0, 0))
        
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        text = f"{label}: {score:.2f}"
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill=color)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    
    return draw_image

def process_directory(img_dir, output_dir, text_prompts, vehicle_name, camera_name, class2id_map, object_nb_map,
                      processor, model, device, box_threshold=0.4, text_threshold=0.3):
    """
    处理目录中的所有图像
    
    object_nb_map: 每个类别所需要的最小检测数量
    
    """
    # 创建输出目录
    dirs = {
        'origin': os.path.join(output_dir, 'origin_img'),
        'debug': os.path.join(output_dir, 'debug_result'),
        'label': os.path.join(output_dir, 'labeling_result')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(img_dir, ext)))
        image_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    image_files.sort()
    
    if not image_files:
        print(f"在 {img_dir} 中未找到图像文件")
        return []
    
    print(f"正在处理 {len(image_files)} 张图像...")
    
    for img_path in tqdm(image_files):
        basename = os.path.basename(img_path)
        basename = f'{vehicle_name}_{camera_name}_{basename}'
        
        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"打开 {img_path} 失败: {e}")
            continue
            
        # 目标检测
        detections, object2nb = detect_objects(
            image, processor, model, text_prompts, device,
            box_threshold=box_threshold, text_threshold=text_threshold
        )
        
        # Add condition to reject unused images
        is_valid_sample = True
        for obj, min_num in object_nb_map.items():
            if obj not in object2nb or object2nb[obj] < min_num:
                is_valid_sample = False
                break
        if not is_valid_sample:
            continue
        
        # 保存原始图像
        origin_path = os.path.join(dirs['origin'], basename)
        shutil.copy(img_path, origin_path)
        
        # 保存可视化结果
        debug_path = os.path.join(dirs['debug'], basename)
        visualize_detections(image, detections).save(debug_path)
        
        # 保存YOLO格式标签
        label_path = os.path.join(dirs['label'], f"{basename}.txt")
        with open(label_path, 'w') as f:
            img_w, img_h = image.size
            for d in detections:
                label = d['label']
                if label not in class2id_map:
                    continue
                
                x1, y1, x2, y2 = d['box']
                # 转换到YOLO格式
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                f.write(f"{class2id_map[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return [os.path.join(dirs['debug'], f) for f in os.listdir(dirs['debug'])]


def create_video_ffmpeg(image_dir, output_video_path, fps=10):
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
    
def check_path(path):
    if not os.path.exists(path):
        print(f'路径 {path} 不存在')
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="基于Grounding DINO的目标检测系统")
    parser.add_argument("--clips-txt", required=True, help="包含cone clip的目录")
    parser.add_argument("--output-dir", default="detection_results", help="输出目录")
    parser.add_argument("--video-path", default="detection_video.mp4", help="输出视频路径")
    parser.add_argument("--prompts", required=True, nargs="+", help="检测提示词列表")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base", help="模型ID")
    parser.add_argument("--box-threshold", type=float, default=0.4, help="边界框置信度阈值")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="文本置信度阈值")
    parser.add_argument("--device", default="cpu", help="运行设备（cuda/cpu）")
    parser.add_argument("--cameras", type=str, default="side_right_camera", help="摄像头名称，多个用逗号分隔")
    parser.add_argument("--object-nbs", type=str, default="cone,2", help="目标名称和数量，多个用逗号分隔")
    
    args = parser.parse_args()
    
    check_path(args.clips_txt)
    check_path(args.output_dir)
    
    # Load model
    processor, model, device = load_model(args.model, args.device)
    
    # Get minimum object nb 
    object_nbs = args.object_nbs.split(',')
    object_nb_map = {}
    for i in range(len(object_nbs) // 2):
        object_nb_map[object_nbs[2 * i]] = int(object_nbs[2 * i + 1])
    
    # Create class to ID map
    class2id_map = {label: idx for idx, label in enumerate(args.prompts)}
    
    with open(args.clips_txt, 'r') as f:
        clips = f.readlines()
    
    # Process each clip
    for clip in tqdm(clips):
        clip = clip.strip()
        if not os.path.exists(clip):
            print(f'路径 {clip} 不存在')
            continue
        # /mnt/juicefs/obstacle_visual_auto_labeling/raw_data/pdb-l4e-c0002/xxxx
        vehicle_name = clip.split("/")[5]
        for camera in args.cameras.split(','):
            img_dir = f'{clip}/raw_images/{camera}'
            process_directory(
                img_dir,
                args.output_dir,
                args.prompts,
                vehicle_name,
                camera,
                class2id_map,
                object_nb_map,
                processor, model, device,
                args.box_threshold,
                args.text_threshold
            )
    
    # 创建视频
    create_video_ffmpeg(f'{args.output_dir}/debug_result', args.video_path)
    print(f"处理完成！结果保存在 {args.output_dir} 中")

if __name__ == "__main__":
    main()