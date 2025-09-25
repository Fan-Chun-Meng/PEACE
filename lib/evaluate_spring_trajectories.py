import os
import pandas as pd
import random


import os
import re
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor
from transformers.models.llava import LlavaForConditionalGeneration

def fake_evaluate_spring_trajectories(image_folder, system_params_dict, confidence_threshold=70):
    """
    模拟大模型输出，用于开发环境调试。
    输出格式保持与大模型一致。
    """
    results = []
    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        params = system_params_dict.get(img_name, {
            "box_size": "Unknown",
            "vel_norm": "Unknown",
            "interaction_strength": "Unknown",
            "spring_prob": "Unknown"
        })

        # 模拟回答文本
        answer = (
            f"Based on the image, the trajectories appear plausible.\n"
            f"Confidence score: {random.randint(60, 100)}"
        )
        confidence = int(re.search(r"(\d{1,3})", answer).group(1))

        results.append({
            "image": img_name,
            "raw_answer": answer,
            "confidence": confidence,
            "box_size": params["box_size"],
            "vel_norm": params["vel_norm"],
            "interaction_strength": params["interaction_strength"],
            "spring_prob": params["spring_prob"]
        })

    df = pd.DataFrame(results)
    high_conf_samples = df[df["confidence"] >= confidence_threshold]
    return df, high_conf_samples




def evaluate_spring_trajectories(
        image_folder,
        system_params_dict,
        model_id="/public/huggingface-models/llava-hf/llava-1.5-7b-hf",
        device="cuda",
        max_new_tokens=200,
        confidence_threshold=70
):
    """
    批量评估图片中弹簧系统轨迹的物理可靠性，返回包含置信度和系统参数的列表。
    可以根据置信度筛选样本进行训练数据增强。
    """

    # 加载模型
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 提取置信度函数
    def extract_confidence(answer: str):
        pattern = r"(?:confidence\s*score|confidence|score)\s*(?:is|of|=|:)?\s*(?:around|approximately)?\s*(\d{1,3})(?:\s*%)?"
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
        # fallback
        answer_lower = answer.lower()
        if "plausible" in answer_lower:
            return 80
        elif "unreliable" in answer_lower or "not plausible" in answer_lower:
            return 20
        else:
            return 50

    results = []

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            params = system_params_dict.get(img_name, {
                "box_size": "Unknown",
                "vel_norm": "Unknown",
                "interaction_strength": "Unknown",
                "spring_prob": "Unknown"
            })

            prompt = (
                "USER: <image>\n"
                "You are a physics expert. The image shows the predicted trajectory of a spring system with 10 particles.\n"
                "The spring system parameters for this simulation are:\n"
                f"- box_size: {params['box_size']}\n"
                f"- vel_norm: {params['vel_norm']}\n"
                f"- interaction_strength: {params['interaction_strength']}\n"
                f"- spring_prob: {params['spring_prob']}\n"
                "Carefully analyze the trajectories and determine whether they are physically plausible:\n"
                "1. Is the trajectory physically plausible and follows expected physics laws?\n"
                "2. Give a confidence score between 0 and 100 based on what you observe.\n"
                "IMPORTANT: Always provide the confidence as a number in the format: 'Confidence score: XX' (0-100).\n"
                "Do not give default answers; base your conclusion on the actual trajectories in the image.\n"
                "ASSISTANT:"
            )

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            confidence = extract_confidence(answer)

            results.append({
                "image": img_name,
                "raw_answer": answer,
                "confidence": confidence,
                "box_size": params["box_size"],
                "vel_norm": params["vel_norm"],
                "interaction_strength": params["interaction_strength"],
                "spring_prob": params["spring_prob"]
            })
            print(f"[INFO] {img_name}: confidence={confidence}")

        except Exception as e:
            print(f"[ERROR] {img_name} failed: {e}")
            results.append({
                "image": img_name,
                "raw_answer": "ERROR",
                "confidence": None,
                "box_size": params.get("box_size", "Unknown"),
                "vel_norm": params.get("vel_norm", "Unknown"),
                "interaction_strength": params.get("interaction_strength", "Unknown"),
                "spring_prob": params.get("spring_prob", "Unknown")
            })

    # 转为 DataFrame
    df = pd.DataFrame(results)

    # 返回高置信度样本
    high_conf_samples = df[df["confidence"] >= confidence_threshold]

    return df, high_conf_samples
