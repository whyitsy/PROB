import clip
import torch
import itertools
from torch.nn import functional as F

#OWOD splits
VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]
UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES={}


T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

VOC_COCO_CLASS_NAMES["OWDETR"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


VOC_CLASS_NAMES = [
"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]
VOC_COCO_CLASS_NAMES["TOWOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES["VOC2007"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))





def build_clip_text_features(class_names, prompt_templates=None):
    """
    构建已知类的CLIP文本特征，支持prompt ensemble提升稳定性
    Args:
        class_names: 已知类名称列表，如["person", "car", "cat", ...]
        prompt_templates:  prompt模板列表，默认用CLIP原生通用模板
    Returns:
        text_features: 归一化后的已知类文本特征，shape [num_classes, 512]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # 强制冻结CLIP所有参数
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    
    if prompt_templates is None:
        # 通用prompt ensemble，适配目标检测场景，提升鲁棒性
        prompt_templates = [
            "a photo of a {}.",
            "a photo of a {} in the scene.",
            "a photo of a small {}.",
            "a photo of a large {}.",
            "a photo of a {} in the background."
        ]
    
    text_features = []
    for class_name in class_names:
        # 对每个类生成多个prompt的文本特征，取平均
        prompts = [template.format(class_name) for template in prompt_templates]
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            class_text_features = clip_model.encode_text(text_tokens)
            class_text_features = F.normalize(class_text_features, dim=-1).mean(dim=0)
        text_features.append(class_text_features)
    
    text_features = torch.stack(text_features, dim=0)
    return F.normalize(text_features, dim=-1)

if __name__ == "__main__":
    # print(dir(clip))  # 应包含 load, tokenize, available_models 等
    # 预计算文本特征，训练中全程固定不变
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # clip_text_features = build_clip_text_features(VOC_COCO_CLASS_NAMES["TOWOD"]).to(device)
    # num_known_classes = len(VOC_COCO_CLASS_NAMES["TOWOD"])
    # torch.save(clip_text_features, "clip_text_feature/TOWOD_clip_text_features.pth") # 81, 512
    
    clip_text_features = build_clip_text_features(VOC_COCO_CLASS_NAMES["OWDETR"]).to(device)
    num_known_classes = len(VOC_COCO_CLASS_NAMES["OWDETR"])
    torch.save(clip_text_features, "clip_text_feature/OWDETR_clip_text_features.pth") # 81, 512
    print("clip_text_features shape:", clip_text_features.shape) 