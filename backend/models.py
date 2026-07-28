import torch
import timm
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Config ---
YOLO_COLOR_PATH = "d:/EECU_Year4-1/Senior_Project/annotation/Cocoa_code_v2/YOLO/weight_yolo/color/best_v11s_color_merge_new_old_data.pt"
YOLO_DEFECT_PATH = "d:/EECU_Year4-1/Senior_Project/annotation/Cocoa_code_v2/YOLO/weight_yolo/defect/best_v11s_defect_merge_new_old_data.pt"

VIT_COLOR_PATH = "d:/EECU_Year4-1/Senior_Project/annotation/Cocoa_code_v2/ViT/weight_Vit/color_raw/convnext_best_tiny.pth"
VIT_DEFECT_PATH = "d:/EECU_Year4-1/Senior_Project/annotation/Cocoa_code_v2/ViT/weight_Vit/defect_raw/convnext_defect_tiny_best.pth"

BASE_CONVNEXT_MODEL = "convnext_tiny"

CLASS_NAMES_COLOR = {0: "Purple Bean", 1: "Brown Bean"} # Translated for API, can be Thai if preferred
CLASS_NAMES_DEFECT = {
    0: "Normal", 1: "Sprouted",
    2: "Slaty", 3: "Moldy"
}

# Thai names for display if needed
CLASS_NAMES_COLOR_TH = {0: "เมล็ดสีม่วง", 1: "เมล็ดสีน้ำตาล"}
CLASS_NAMES_DEFECT_TH = {
    0: "เมล็ดปกติ", 1: "เมล็ดงอก",
    2: "เมล็ดสีเทาหินชนวน", 3: "เมล็ดขึ้นรา"
}

# --- Pydantic Models ---
class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] # [x1, y1, x2, y2]

class PredictionResponse(BaseModel):
    detections: List[DetectionResult]
    summary: Dict[str, int]
    grade: Optional[str] = None
    image_width: int
    image_height: int

# --- Model Wrappers ---

class CocoaModels:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading models on {self.device}...")
        
        # Load YOLO
        self.yolo_color = YOLO(YOLO_COLOR_PATH)
        # self.yolo_defect = YOLO(YOLO_DEFECT_PATH) # Not used in main logic of app_convnext.py for detection, only color yolo is used for detection

        # Load ConvNeXt
        self.convnext_color = self._load_convnext(VIT_COLOR_PATH, len(CLASS_NAMES_COLOR))
        self.convnext_defect = self._load_convnext(VIT_DEFECT_PATH, len(CLASS_NAMES_DEFECT))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Models loaded.")

    def _load_convnext(self, path, num_classes):
        model = timm.create_model(
            BASE_CONVNEXT_MODEL,
            pretrained=False,
            num_classes=num_classes
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_yolo_only(self, image: np.ndarray) -> PredictionResponse:
        # YOLO v11 inference
        results = self.yolo_color.predict(image, conf=0.25, iou=0.45, verbose=False)[0]
        
        detections = []
        summary = {}
        
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            
            class_name = CLASS_NAMES_COLOR_TH.get(cls_id, str(cls_id))
            
            detections.append(DetectionResult(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=xyxy
            ))
            summary[class_name] = summary.get(class_name, 0) + 1
            
        return PredictionResponse(
            detections=detections,
            summary=summary,
            image_width=image.shape[1],
            image_height=image.shape[0]
        )

    def predict_hybrid(self, image: np.ndarray) -> PredictionResponse:
        # 1. Detect with YOLO
        results = self.yolo_color.predict(image, conf=0.25, iou=0.45, verbose=False)[0]
        
        detections = []
        summary = {}
        
        # Prepare crops for batch processing or single processing
        # For simplicity, process one by one as in original code
        
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Crop
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
                
            # Decide which model to use? 
            # The original code runs BOTH color and defect models for every crop?
            # "Live-1 (ViT สี)" and "Live-2 (ViT defect)"
            # But for a single image prediction, we probably want to show one result per bean.
            # However, a bean has BOTH a color and a defect status?
            # Or are they mutually exclusive classes in the final output?
            
            # Looking at app_convnext.py:
            # It calculates "grade" based on counts from BOTH models.
            # But it draws boxes on TWO separate frames (frame1, frame2).
            
            # For this web app, let's return the "Defect" classification as primary if it's not normal?
            # Or maybe return both?
            # The user request says "แยกด้วยมาตราฐาน 2 ข้อ คือ สีและคุณภาพ" (Separate by 2 standards: Color and Quality).
            
            # Let's use the Color model to get Purple/Brown
            # And Defect model to get Normal/Sprouted/Slaty/Moldy
            
            # Wait, "Slaty" (สีเทาหินชนวน) is a color/quality thing.
            
            # Let's run both and combine info?
            # Or just run the Defect model?
            # The original code runs both independently and calculates grade.
            
            # Let's run both.
            
            # Predict Color
            idx_color, label_color = self._predict_crop(self.convnext_color, crop, CLASS_NAMES_COLOR_TH)
            
            # Predict Defect
            idx_defect, label_defect = self._predict_crop(self.convnext_defect, crop, CLASS_NAMES_DEFECT_TH)
            
            # Combine label? e.g. "Purple / Normal"
            combined_label = f"{label_color} / {label_defect}"
            
            detections.append(DetectionResult(
                class_id=idx_defect, # Use defect ID as primary for now?
                class_name=combined_label,
                confidence=1.0, # ConvNeXt doesn't give conf easily without softmax, just use 1.0
                bbox=xyxy
            ))
            
            summary[label_color] = summary.get(label_color, 0) + 1
            summary[label_defect] = summary.get(label_defect, 0) + 1

        # Calculate Grade
        grade = self._calculate_grade(summary)

        return PredictionResponse(
            detections=detections,
            summary=summary,
            grade=grade,
            image_width=image.shape[1],
            image_height=image.shape[0]
        )

    def _predict_crop(self, model, crop, class_dict):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(input_tensor)
        idx = logits.argmax(dim=1).item()
        return idx, class_dict.get(idx, str(idx))

    def _calculate_grade(self, summary):
        # Logic from check_grade in app_convnext.py
        # th = [("พิเศษ", 3, 3, 2.5), ("ชั้น 1", 3, 5, 3), ("ชั้น 2", 4, 8, 5)]
        # c1 = Moldy %
        # c3 = Sprouted %
        # c2 = Purple + Slaty %
        
        total = 0
        # Sum all defect counts to get total beans?
        # Or just use the number of detections?
        # In app_convnext.py, total2 = sum(cnt_defect.values())
        
        # We can estimate total from summary of defect classes
        total = sum(summary.get(name, 0) for name in CLASS_NAMES_DEFECT_TH.values())
        
        if total == 0:
            return "No Beans"
            
        moldy = summary.get("เมล็ดขึ้นรา", 0)
        sprouted = summary.get("เมล็ดงอก", 0)
        purple = summary.get("เมล็ดสีม่วง", 0)
        slaty = summary.get("เมล็ดสีเทาหินชนวน", 0)
        
        c1 = 100 * moldy / total
        c3 = 100 * sprouted / total
        c2 = 100 * (purple + slaty) / total
        
        th = [("พิเศษ", 3, 3, 2.5), ("ชั้น 1", 3, 5, 3), ("ชั้น 2", 4, 8, 5)]
        
        if c1 > 4 or c2 > 8 or c3 > 5: return "ตกเกรด"
        for g, m1, m2, m3 in th:
            if c1 <= m1 and c2 <= m2 and c3 <= m3: return g
        return "ตกเกรด"

# Global instance
cocoa_models = CocoaModels()
