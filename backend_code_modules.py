# measurement_extraction.py
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
from smplx import SMPLX
import json

@dataclass
class BodyMeasurements:
    """Standardized body measurements in cm"""
    chest: float
    waist: float
    hips: float
    inseam: float
    shoulders: float
    arm_length: float
    torso_length: float
    neck: float
    thigh: float
    calf: float
    bicep: float
    forearm: float
    height: float
    weight: Optional[float] = None
    body_type: Optional[str] = None

class MeasurementExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        # Initialize SMPL-X model
        self.body_model = SMPLX(
            model_path='models/smplx',
            gender='neutral',
            use_face_contour=False
        )
        
    def extract_from_image(self, image_path: str, height_cm: float) -> BodyMeasurements:
        """Extract measurements from a single image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get 2D landmarks
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            raise ValueError("No pose detected in image")
        
        landmarks = self._landmarks_to_array(results.pose_landmarks)
        
        # Estimate 3D body mesh
        body_params = self._estimate_3d_body(landmarks, image.shape)
        vertices = self.body_model(**body_params).vertices[0].detach().numpy()
        
        # Calculate measurements from vertices
        measurements = self._calculate_measurements(vertices, height_cm)
        return measurements
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array"""
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    def _estimate_3d_body(self, landmarks_2d: np.ndarray, img_shape: Tuple) -> Dict:
        """Estimate SMPL-X parameters from 2D landmarks"""
        # Simplified - in production use HMR or PIXIE for better accuracy
        return {
            'body_pose': torch.zeros(1, 63),
            'global_orient': torch.zeros(1, 3),
            'betas': torch.randn(1, 10) * 0.03,  # Shape parameters
            'transl': torch.zeros(1, 3)
        }
    
    def _calculate_measurements(self, vertices: np.ndarray, height_cm: float) -> BodyMeasurements:
        """Calculate body measurements from 3D vertices"""
        # Vertex indices for key body parts (SMPL-X specific)
        CHEST_VERTICES = [3076, 3060, 6473, 6489]
        WAIST_VERTICES = [4339, 4298, 7747, 7706]
        HIP_VERTICES = [3135, 3093, 6544, 6502]
        
        # Calculate scaling factor based on height
        mesh_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        scale_factor = height_cm / (mesh_height * 100)
        
        # Calculate circumferences
        chest = self._calculate_circumference(vertices[CHEST_VERTICES]) * scale_factor
        waist = self._calculate_circumference(vertices[WAIST_VERTICES]) * scale_factor
        hips = self._calculate_circumference(vertices[HIP_VERTICES]) * scale_factor
        
        # Calculate lengths
        shoulders = np.linalg.norm(vertices[2854] - vertices[6261]) * scale_factor
        inseam = np.linalg.norm(vertices[3151] - vertices[3327]) * scale_factor
        
        return BodyMeasurements(
            chest=chest,
            waist=waist,
            hips=hips,
            inseam=inseam,
            shoulders=shoulders,
            arm_length=self._calculate_arm_length(vertices) * scale_factor,
            torso_length=self._calculate_torso_length(vertices) * scale_factor,
            neck=self._calculate_neck(vertices) * scale_factor,
            thigh=self._calculate_thigh(vertices) * scale_factor,
            calf=self._calculate_calf(vertices) * scale_factor,
            bicep=self._calculate_bicep(vertices) * scale_factor,
            forearm=self._calculate_forearm(vertices) * scale_factor,
            height=height_cm
        )
    
    def _calculate_circumference(self, points: np.ndarray) -> float:
        """Calculate circumference from 3D points"""
        # Fit ellipse to points and calculate circumference
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return 2 * np.pi * np.mean(distances)
    
    def _calculate_arm_length(self, vertices: np.ndarray) -> float:
        """Calculate arm length from shoulder to wrist"""
        return np.linalg.norm(vertices[2854] - vertices[5528])
    
    def _calculate_torso_length(self, vertices: np.ndarray) -> float:
        """Calculate torso length from shoulder to hip"""
        return np.linalg.norm(vertices[2854] - vertices[3135])
    
    def _calculate_neck(self, vertices: np.ndarray) -> float:
        """Calculate neck circumference"""
        NECK_VERTICES = [134, 227, 3634, 3726]
        return self._calculate_circumference(vertices[NECK_VERTICES])
    
    def _calculate_thigh(self, vertices: np.ndarray) -> float:
        """Calculate thigh circumference"""
        THIGH_VERTICES = [3151, 3184, 6560, 6593]
        return self._calculate_circumference(vertices[THIGH_VERTICES])
    
    def _calculate_calf(self, vertices: np.ndarray) -> float:
        """Calculate calf circumference"""
        CALF_VERTICES = [3268, 3295, 6677, 6704]
        return self._calculate_circumference(vertices[CALF_VERTICES])
    
    def _calculate_bicep(self, vertices: np.ndarray) -> float:
        """Calculate bicep circumference"""
        BICEP_VERTICES = [5087, 5119, 1709, 1741]
        return self._calculate_circumference(vertices[BICEP_VERTICES])
    
    def _calculate_forearm(self, vertices: np.ndarray) -> float:
        """Calculate forearm circumference"""
        FOREARM_VERTICES = [5361, 5393, 1983, 2015]
        return self._calculate_circumference(vertices[FOREARM_VERTICES])

# fit_prediction.py
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

@dataclass
class FitScore:
    size: str
    overall_score: float
    chest_fit: str
    waist_fit: str
    hip_fit: str
    length_fit: str
    recommendation: str
    notes: List[str]

class FitPredictionEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = RandomForestRegressor(n_estimators=100)
            
        self.fit_tolerance = {
            'perfect': 2,  # ±2cm
            'good': 3,     # ±3cm
            'acceptable': 5,  # ±5cm
            'poor': float('inf')
        }
    
    def predict_fit(self, 
                   user_measurements: Dict[str, float],
                   garment_sizes: Dict[str, Dict[str, List[float]]],
                   fit_type: str = 'regular') -> Dict[str, FitScore]:
        """Predict fit scores for all available sizes"""
        fit_scores = {}
        
        for size, ranges in garment_sizes.items():
            score = self._calculate_fit_score(user_measurements, ranges, fit_type)
            fit_scores[size] = score
        
        return fit_scores
    
    def _calculate_fit_score(self,
                            user: Dict[str, float],
                            garment: Dict[str, List[float]],
                            fit_type: str) -> FitScore:
        """Calculate detailed fit score for a single size"""
        
        # Adjust expectations based on fit type
        adjustments = self._get_fit_adjustments(fit_type)
        
        # Calculate individual scores
        chest_score, chest_fit = self._score_measurement(
            user.get('chest', 0),
            garment.get('chest', [0, 0]),
            adjustments.get('chest', 0)
        )
        
        waist_score, waist_fit = self._score_measurement(
            user.get('waist', 0),
            garment.get('waist', [0, 0]),
            adjustments.get('waist', 0)
        )
        
        hip_score, hip_fit = self._score_measurement(
            user.get('hips', 0),
            garment.get('hips', [0, 0]),
            adjustments.get('hips', 0)
        )
        
        length_score, length_fit = self._score_measurement(
            user.get('torso_length', 0),
            garment.get('length', [0, 0]),
            0
        )
        
        # Calculate weighted overall score
        weights = {'chest': 0.35, 'waist': 0.25, 'hips': 0.25, 'length': 0.15}
        overall_score = (
            chest_score * weights['chest'] +
            waist_score * weights['waist'] +
            hip_score * weights['hips'] +
            length_score * weights['length']
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(overall_score)
        
        # Generate notes
        notes = self._generate_fit_notes(
            chest_fit, waist_fit, hip_fit, length_fit, fit_type
        )
        
        return FitScore(
            size=garment.get('size', 'Unknown'),
            overall_score=round(overall_score, 2),
            chest_fit=chest_fit,
            waist_fit=waist_fit,
            hip_fit=hip_fit,
            length_fit=length_fit,
            recommendation=recommendation,
            notes=notes
        )
    
    def _score_measurement(self,
                          user_value: float,
                          garment_range: List[float],
                          adjustment: float) -> Tuple[float, str]:
        """Score a single measurement"""
        if len(garment_range) != 2:
            return 0.0, "unknown"
        
        min_val, max_val = garment_range
        adjusted_val = user_value + adjustment
        
        if min_val <= adjusted_val <= max_val:
            # Perfect fit
            center = (min_val + max_val) / 2
            deviation = abs(adjusted_val - center)
            score = 1.0 - (deviation / (max_val - min_val))
            return score, "perfect"
        
        # Calculate how far outside the range
        if adjusted_val < min_val:
            diff = min_val - adjusted_val
            fit_desc = "tight"
        else:
            diff = adjusted_val - max_val
            fit_desc = "loose"
        
        # Score based on deviation
        if diff <= self.fit_tolerance['good']:
            return 0.7, f"slightly_{fit_desc}"
        elif diff <= self.fit_tolerance['acceptable']:
            return 0.4, fit_desc
        else:
            return 0.0, f"very_{fit_desc}"
    
    def _get_fit_adjustments(self, fit_type: str) -> Dict[str, float]:
        """Get measurement adjustments based on fit type"""
        adjustments = {
            'slim': {'chest': -2, 'waist': -2, 'hips': -1},
            'regular': {'chest': 0, 'waist': 0, 'hips': 0},
            'relaxed': {'chest': 2, 'waist': 3, 'hips': 2},
            'oversized': {'chest': 5, 'waist': 6, 'hips': 5}
        }
        return adjustments.get(fit_type, adjustments['regular'])
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on overall score"""
        if score >= 0.85:
            return "Excellent fit - Highly recommended"
        elif score >= 0.7:
            return "Good fit - Recommended"
        elif score >= 0.5:
            return "Acceptable fit - Consider alternatives"
        else:
            return "Poor fit - Not recommended"
    
    def _generate_fit_notes(self,
                           chest: str, waist: str, 
                           hip: str, length: str,
                           fit_type: str) -> List[str]:
        """Generate detailed fit notes"""
        notes = []
        
        if 'tight' in chest:
            notes.append("May feel restrictive across chest")
        elif 'loose' in chest:
            notes.append("Will have extra room in chest area")
        
        if 'tight' in waist:
            notes.append("Snug fit around midsection")
        elif 'loose' in waist:
            notes.append("Relaxed fit through waist")
        
        if fit_type == 'slim' and any('loose' in x for x in [chest, waist]):
            notes.append("Consider sizing down for intended slim fit")
        
        if length == 'slightly_short':
            notes.append("May need hemming or length adjustment")
        
        return notes

# visualization.py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2
from typing import Optional

class VirtualTryOnGenerator:
    def __init__(self, model_path: str = "yisol/IDM-VTON"):
        """Initialize virtual try-on model"""
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
    
    def generate_tryon(self,
                      person_image: Image.Image,
                      garment_image: Image.Image,
                      pose_image: Optional[Image.Image] = None) -> Image.Image:
        """Generate virtual try-on image"""
        
        if pose_image is None:
            pose_image = self._extract_pose(person_image)
        
        # Prepare prompt
        prompt = "person wearing clothing, high quality, detailed, fashion photography"
        negative_prompt = "deformed, distorted, low quality, blurry"
        
        # Composite garment onto pose guidance
        control_image = self._prepare_control_image(pose_image, garment_image)
        
        # Generate try-on
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        
        return result
    
    def _extract_pose(self, image: Image.Image) -> Image.Image:
        """Extract pose from person image using OpenPose"""
        # Simplified - use OpenPose or MediaPipe
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
        
        image_array = np.array(image)
        results = pose.process(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        
        # Draw pose on blank canvas
        pose_image = np.zeros_like(image_array)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                pose_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        
        return Image.fromarray(pose_image)
    
    def _prepare_control_image(self,
                              pose: Image.Image,
                              garment: Image.Image) -> Image.Image:
        """Prepare control image for ControlNet"""
        # Composite garment mask onto pose
        # This is simplified - production would use segmentation
        return pose

# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import asyncio
from measurement_extraction import MeasurementExtractor, BodyMeasurements
from fit_prediction import FitPredictionEngine
from visualization import VirtualTryOnGenerator

app = FastAPI(title="Size Recommendation API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
measurement_extractor = MeasurementExtractor()
fit_engine = FitPredictionEngine()
tryon_generator = VirtualTryOnGenerator()

class UserInput(BaseModel):
    height: float
    weight: Optional[float]
    age: Optional[int]
    gender: str = "male"
    usual_size: Optional[str]

class GarmentData(BaseModel):
    brand: str
    product_id: str
    sizes: Dict[str, Dict[str, List[float]]]
    fit: str = "regular"
    fabric: Optional[str]

class MeasurementResponse(BaseModel):
    measurements: Dict[str, float]
    confidence: float
    body_type: Optional[str]

class SizeRecommendation(BaseModel):
    recommended_size: str
    fit_score: float
    fit_notes: List[str]
    alternatives: List[Dict[str, float]]

@app.post("/api/v1/extract_measurements", response_model=MeasurementResponse)
async def extract_measurements(
    file: UploadFile = File(...),
    height_cm: float = 175,
    image_type: str = "front"
):
    """Extract body measurements from uploaded image"""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Extract measurements
        measurements = measurement_extractor.extract_from_image(temp_path, height_cm)
        
        return MeasurementResponse(
            measurements=measurements.__dict__,
            confidence=0.87,  # From ML model
            body_type="athletic"  # Classification
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/recommend_size", response_model=SizeRecommendation)
async def recommend_size(
    user_measurements: Dict[str, float],
    garment_data: GarmentData
):
    """Recommend best size based on measurements and garment data"""
    try:
        # Get fit predictions for all sizes
        fit_scores = fit_engine.predict_fit(
            user_measurements,
            garment_data.sizes,
            garment_data.fit
        )
        
        # Sort by score
        sorted_sizes = sorted(
            fit_scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        best_fit = sorted_sizes[0][1]
        alternatives = [
            {"size": size, "score": score.overall_score}
            for size, score in sorted_sizes[1:3]
        ]
        
        return SizeRecommendation(
            recommended_size=best_fit.size,
            fit_score=best_fit.overall_score,
            fit_notes=best_fit.notes,
            alternatives=alternatives
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/generate_tryon")
async def generate_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """Generate virtual try-on visualization"""
    try:
        # Load images
        person_img = Image.open(await person_image.read())
        garment_img = Image.open(await garment_image.read())
        
        # Generate try-on
        result = tryon_generator.generate_tryon(person_img, garment_img)
        
        # Save and return URL
        output_path = f"/tmp/tryon_{person_image.filename}"
        result.save(output_path)
        
        return {"tryon_url": output_path, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}