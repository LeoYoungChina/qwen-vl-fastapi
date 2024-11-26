from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
import base64
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import time

app = FastAPI()

# 加载模型和处理器
model_dir = "/home/muos/models/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

# 检查模型所在设备
device = next(model.parameters()).device

def encode_image_base64(image_data):
    """将图片数据编码为base64字符串"""
    return base64.b64encode(image_data).decode('utf-8')

def extract_frames(video_path, num_frames=1):
    """从视频中抽取指定数量的帧"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(frame)
    cap.release()
    return frames

def generate_response(video_frames, user_text):
    # 构建消息结构
    video_paths = [f"data:image/jpeg;base64,{encode_image_base64(cv2.imencode('.jpg', frame)[1].tobytes())}" for frame in video_frames]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_paths,
                    "fps": 1.0,
                },
                {"type": "text", "text": user_text}
            ]
        }
    ]
    
    # 准备推理数据
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    if not image_inputs and not video_inputs:
        raise ValueError("image, image_url or video should in content.")
    
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128000000, temperature=0.7, top_k=50)
    
    end_time = time.time()
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    del inputs
    torch.cuda.empty_cache()
    
    elapsed_time = end_time - start_time
    
    return output_text[0], elapsed_time

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), user_input: str = ""):
    file_content = await file.read()
    
    if file.content_type.startswith('image'):
        image_data = file_content
        image_base64 = encode_image_base64(image_data)
        image_url = f"data:image/jpeg;base64,{image_base64}"
        video_frames = [cv2.imdecode(np.frombuffer(image_data, np.uint8), 1)]
    elif file.content_type.startswith('video'):
        video_path = file.filename
        with open(video_path, 'wb') as f:
            f.write(file_content)
        video_frames = extract_frames(video_path, num_frames=1)
    
    response, elapsed_time = generate_response(video_frames, user_input)
    
    return {"description": response, "time_taken": f"{elapsed_time:.2f} seconds"}