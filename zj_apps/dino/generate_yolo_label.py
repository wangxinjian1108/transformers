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

def process_directory(img_dir, output_dir, text_prompts, model_id="IDEA-Research/grounding-dino-tiny", 
                     box_threshold=0.4, text_threshold=0.3, device=None):
    """处理目录中的所有图像"""
    # 创建输出目录
    dirs = {
        'origin': os.path.join(output_dir, 'origin_img'),
        'debug': os.path.join(output_dir, 'debug_result'),
        'label': os.path.join(output_dir, 'labeling_result')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 创建类别到ID的映射
    class2id_map = {label: idx for idx, label in enumerate(text_prompts)}
    
    # 加载模型
    processor, model, device = load_model(model_id, device)
    
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
        
        # 保存原始图像
        origin_path = os.path.join(dirs['origin'], basename)
        shutil.copy(img_path, origin_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"打开 {img_path} 失败: {e}")
            continue
            
        # 目标检测
        detections = detect_objects(
            image, processor, model, text_prompts, device,
            box_threshold=box_threshold, text_threshold=text_threshold
        )
        
        # 保存可视化结果
        debug_path = os.path.join(dirs['debug'], f"debug_{basename}")
        visualize_detections(image, detections).save(debug_path)
        
        # 保存YOLO格式标签
        label_path = os.path.join(dirs['label'], f"{os.path.splitext(basename)[0]}.txt")
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

def create_video(image_files, output_video_path, fps=5):
    """创建结果视频"""
    if not image_files:
        print("没有可用于创建视频的图像")
        return
    
    first_img = cv2.imread(image_files[0])
    h, w = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
    video.release()

def main():
    parser = argparse.ArgumentParser(description="基于Grounding DINO的目标检测系统")
    parser.add_argument("--img-dir", required=True, help="输入图像目录")
    parser.add_argument("--output-dir", default="detection_results", help="输出目录")
    parser.add_argument("--video-path", default="detection_video.mp4", help="输出视频路径")
    parser.add_argument("--prompts", required=True, nargs="+", help="检测提示词列表")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="模型ID")
    parser.add_argument("--box-threshold", type=float, default=0.4, help="边界框置信度阈值")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="文本置信度阈值")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率")
    parser.add_argument("--device", default="cpu", help="运行设备（cuda/cpu）")
    
    args = parser.parse_args()
    
    # 处理图像目录
    result_files = process_directory(
        args.img_dir,
        args.output_dir,
        args.prompts,
        model_id=args.model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device
    )
    
    # 创建视频
    if result_files:
        create_video(result_files, args.video_path, args.fps)
        print(f"处理完成！结果保存在 {args.output_dir} 中")

if __name__ == "__main__":
    main()