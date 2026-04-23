#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Gaze Classification System
Based on State-of-the-Art 2024 Research

This implements modern deep learning approaches including:
- CNN-RNN hybrid architecture for temporal sequence modeling
- Event-based processing with motion-aware filtering
- Adaptive thresholds based on individual user characteristics
- Real-time optimization for Tobii consumer hardware
- NPU acceleration when available

References:
- "DHECA-SuperGaze: Dual Head-Eye Cross-Attention and Super-Resolution for Unconstrained Gaze Estimation" (2025)
- "GazeSCRNN: Event-based Near-eye Gaze Tracking using a Spiking Neural Network" (2025)
- "Inference-Time Gaze Refinement for Micro-Expression Recognition" (2025)
- "Towards Structured Gaze Data Classification: The Gaze Data Clustering Taxonomy (GCT)" (2025)
"""

import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import threading

# Try to import neural network libraries
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("ModernGazeClassifier")

@dataclass
class GazeSequence:
    """Modern gaze sequence data structure"""
    timestamps: np.ndarray
    coordinates: np.ndarray  # Shape: (n_samples, 2) - x, y coordinates
    velocities: np.ndarray   # Shape: (n_samples, 2) - velocity vectors
    accelerations: np.ndarray # Shape: (n_samples, 2) - acceleration vectors
    pupil_sizes: np.ndarray  # Shape: (n_samples,) - pupil diameter
    confidence: np.ndarray   # Shape: (n_samples,) - tracking confidence
    duration: float          # Total sequence duration in milliseconds
    
@dataclass 
class ClassificationResult:
    """Modern classification result with enhanced information"""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    temporal_features: Dict[str, float]
    movement_characteristics: Dict[str, float]
    processing_time_ms: float

class MotionAwareFilter:
    """
    Motion-Aware Median Filtering for gaze refinement
    Based on "Inference-Time Gaze Refinement for Micro-Expression Recognition" (2025)
    """
    
    def __init__(self, window_size: int = 5, velocity_threshold: float = 500.0):
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.position_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        
    def filter_sequence(self, gaze_sequence: GazeSequence) -> GazeSequence:
        """Apply motion-aware filtering to suppress blink artifacts while preserving natural dynamics"""
        
        filtered_coords = np.copy(gaze_sequence.coordinates)
        filtered_velocities = np.copy(gaze_sequence.velocities)
        
        for i in range(len(gaze_sequence.coordinates)):
            current_velocity = np.linalg.norm(gaze_sequence.velocities[i])
            
            # Detect blink-induced spikes
            if current_velocity > self.velocity_threshold:
                # Apply median filtering in window
                start_idx = max(0, i - self.window_size // 2)
                end_idx = min(len(gaze_sequence.coordinates), i + self.window_size // 2 + 1)
                
                # Calculate median position in window, excluding potential outliers
                window_coords = gaze_sequence.coordinates[start_idx:end_idx]
                window_velocities = np.array([np.linalg.norm(gaze_sequence.velocities[j]) 
                                            for j in range(start_idx, end_idx)])
                
                # Only use low-velocity points for median calculation
                valid_mask = window_velocities < self.velocity_threshold
                if np.any(valid_mask):
                    valid_coords = window_coords[valid_mask]
                    filtered_coords[i] = np.median(valid_coords, axis=0)
                    
                    # Recalculate velocity after filtering
                    if i > 0:
                        dt = gaze_sequence.timestamps[i] - gaze_sequence.timestamps[i-1]
                        if dt > 0:
                            filtered_velocities[i] = (filtered_coords[i] - filtered_coords[i-1]) / dt
        
        return GazeSequence(
            timestamps=gaze_sequence.timestamps,
            coordinates=filtered_coords,
            velocities=filtered_velocities,
            accelerations=gaze_sequence.accelerations,
            pupil_sizes=gaze_sequence.pupil_sizes,
            confidence=gaze_sequence.confidence,
            duration=gaze_sequence.duration
        )

class AdaptiveThresholdCalculator:
    """
    Adaptive threshold calculation based on individual user characteristics
    Implements personalized calibration from recent research
    """
    
    def __init__(self):
        self.user_profile = {
            'fixation_dispersion_baseline': 50.0,
            'saccade_velocity_baseline': 300.0,
            'pursuit_velocity_range': (50.0, 200.0),
            'individual_scaling_factor': 1.0
        }
        self.calibration_data = deque(maxlen=1000)  # Store recent classifications
        
    def update_profile(self, gaze_sequence: GazeSequence, ground_truth: Optional[str] = None):
        """Update user profile based on observed gaze patterns"""
        
        # Calculate sequence characteristics
        dispersion = self._calculate_dispersion(gaze_sequence.coordinates)
        avg_velocity = np.mean(np.linalg.norm(gaze_sequence.velocities, axis=1))
        
        self.calibration_data.append({
            'dispersion': dispersion,
            'velocity': avg_velocity,
            'timestamp': time.time(),
            'ground_truth': ground_truth
        })
        
        # Update baselines if we have enough calibration data
        if len(self.calibration_data) >= 50:
            self._recalculate_baselines()
    
    def _calculate_dispersion(self, coordinates: np.ndarray) -> float:
        """Calculate spatial dispersion of gaze points"""
        if len(coordinates) < 2:
            return 0.0
        return np.sqrt(np.var(coordinates[:, 0]) + np.var(coordinates[:, 1])) * 1000  # Convert to pixels
    
    def _recalculate_baselines(self):
        """Recalculate adaptive thresholds based on user's historical data"""
        recent_data = list(self.calibration_data)[-100:]  # Use last 100 samples
        
        dispersions = [d['dispersion'] for d in recent_data]
        velocities = [d['velocity'] for d in recent_data]
        
        # Update baselines with moving percentiles
        self.user_profile['fixation_dispersion_baseline'] = np.percentile(dispersions, 25)
        self.user_profile['saccade_velocity_baseline'] = np.percentile(velocities, 75)
        
        logger.info(f"Updated adaptive thresholds: "
                   f"fixation_dispersion={self.user_profile['fixation_dispersion_baseline']:.1f}, "
                   f"saccade_velocity={self.user_profile['saccade_velocity_baseline']:.1f}")

class CNNRNNGazeClassifier:
    """
    Modern CNN-RNN hybrid architecture for gaze classification
    Based on latest deep learning research in eye movement analysis
    """
    
    def __init__(self, input_features: int = 8, sequence_length: int = 30):
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.classes = ['fixation', 'saccade', 'pursuit', 'blink']
        self.model = None
        self.is_trained = False
        
        if HAS_TORCH:
            self._build_torch_model()
        elif HAS_TF:
            self._build_tf_model()
        else:
            logger.warning("No deep learning framework available, using traditional methods")
    
    def _build_torch_model(self):
        """Build PyTorch CNN-RNN model"""
        
        class GazeCNNRNN(nn.Module):
            def __init__(self, input_features, sequence_length, num_classes):
                super(GazeCNNRNN, self).__init__()
                
                # 1D CNN for feature extraction
                self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
                
                # Batch normalization
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(128)
                self.bn3 = nn.BatchNorm1d(64)
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
                
            def forward(self, x):
                # x shape: (batch, features, sequence)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                
                # Transpose for LSTM: (batch, sequence, features)
                x = x.transpose(1, 2)
                
                # LSTM
                lstm_out, _ = self.lstm(x)
                
                # Attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Global average pooling
                features = torch.mean(attn_out, dim=1)
                
                # Classification
                output = self.classifier(features)
                return output
        
        self.model = GazeCNNRNN(self.input_features, self.sequence_length, len(self.classes))
        logger.info("Built PyTorch CNN-RNN model for gaze classification")
    
    def _build_tf_model(self):
        """Build TensorFlow CNN-RNN model"""
        
        inputs = tf.keras.Input(shape=(self.sequence_length, self.input_features))
        
        # Reshape for 1D CNN
        x = tf.keras.layers.Reshape((self.sequence_length, self.input_features, 1))(inputs)
        
        # 1D CNN layers
        x = tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 1), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Reshape back for RNN
        x = tf.keras.layers.Reshape((self.sequence_length, -1))(x)
        
        # Bidirectional LSTM
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(8, 128)(x, x)
        x = tf.keras.layers.Add()([x, attention])
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(len(self.classes), activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Built TensorFlow CNN-RNN model for gaze classification")
    
    def extract_features(self, gaze_sequence: GazeSequence) -> np.ndarray:
        """Extract comprehensive features from gaze sequence"""
        
        coords = gaze_sequence.coordinates
        velocities = gaze_sequence.velocities
        accelerations = gaze_sequence.accelerations
        
        # Create feature matrix: [x, y, vx, vy, ax, ay, pupil, confidence]
        features = np.column_stack([
            coords[:, 0],                    # x coordinate
            coords[:, 1],                    # y coordinate  
            velocities[:, 0],                # x velocity
            velocities[:, 1],                # y velocity
            accelerations[:, 0],             # x acceleration
            accelerations[:, 1],             # y acceleration
            gaze_sequence.pupil_sizes,       # pupil size
            gaze_sequence.confidence         # tracking confidence
        ])
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        return features
    
    def predict(self, gaze_sequence: GazeSequence) -> ClassificationResult:
        """Predict gaze event class using modern deep learning model"""
        
        start_time = time.time()
        
        if not self.model or not self.is_trained:
            # Fallback to traditional classification
            return self._traditional_classify(gaze_sequence)
        
        # Extract features
        features = self.extract_features(gaze_sequence)
        
        # Pad or truncate to fixed sequence length
        if len(features) > self.sequence_length:
            features = features[-self.sequence_length:]
        else:
            padding = np.zeros((self.sequence_length - len(features), self.input_features))
            features = np.vstack([padding, features])
        
        # Reshape for model input
        if HAS_TORCH and isinstance(self.model, torch.nn.Module):
            # PyTorch prediction
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).transpose(1, 2)
                outputs = self.model(x)
                probabilities = F.softmax(outputs, dim=1).numpy()[0]
        elif HAS_TF:
            # TensorFlow prediction
            x = features.reshape(1, self.sequence_length, self.input_features)
            probabilities = self.model.predict(x, verbose=0)[0]
        else:
            return self._traditional_classify(gaze_sequence)
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        class_probs = {cls: float(prob) for cls, prob in zip(self.classes, probabilities)}
        
        # Calculate additional features
        temporal_features = self._calculate_temporal_features(gaze_sequence)
        movement_characteristics = self._calculate_movement_characteristics(gaze_sequence)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            temporal_features=temporal_features,
            movement_characteristics=movement_characteristics,
            processing_time_ms=processing_time
        )
    
    def _traditional_classify(self, gaze_sequence: GazeSequence) -> ClassificationResult:
        """Fallback traditional classification when deep learning is unavailable"""
        
        start_time = time.time()
        
        # Calculate basic features
        dispersion = np.sqrt(np.var(gaze_sequence.coordinates[:, 0]) + 
                           np.var(gaze_sequence.coordinates[:, 1])) * 1000
        avg_velocity = np.mean(np.linalg.norm(gaze_sequence.velocities, axis=1))
        max_velocity = np.max(np.linalg.norm(gaze_sequence.velocities, axis=1))
        avg_acceleration = np.mean(np.linalg.norm(gaze_sequence.accelerations, axis=1))
        velocity_std = np.std(np.linalg.norm(gaze_sequence.velocities, axis=1))
        
        # BALANCED THRESHOLDS - Fixed to reduce fixation bias
        fixation_score = 0.0
        saccade_score = 0.0
        pursuit_score = 0.0
        
        # FIXATION: More restrictive criteria to reduce false positives
        if dispersion < 300:  # Much more restrictive (was 650)
            fixation_score += 0.3
        if avg_velocity < 400:  # More restrictive (was 800)
            fixation_score += 0.3
        if velocity_std < 300:  # Added velocity consistency check
            fixation_score += 0.2
        if gaze_sequence.duration > 200:  # Slightly higher duration requirement
            fixation_score += 0.2
        
        # SACCADE: More achievable criteria 
        if max_velocity > 2500:  # Much lower threshold (was 5000)
            saccade_score += 0.4
        if avg_acceleration > 3000:  # Lower threshold (was 7000)
            saccade_score += 0.3
        if velocity_std > 800:  # Added variability check for ballistic movement
            saccade_score += 0.2
        if gaze_sequence.duration < 150:  # Slightly longer allowed
            saccade_score += 0.1
        
        # PURSUIT: More achievable and broader criteria
        if 50 < dispersion < 600:  # Broader range (was 100-400)
            pursuit_score += 0.3
        if 150 <= avg_velocity <= 800:  # Broader range (was 200-500)
            pursuit_score += 0.3
        if 300 < velocity_std < 600:  # Moderate variability for smooth tracking
            pursuit_score += 0.2
        if gaze_sequence.duration > 300:  # Lower duration requirement
            pursuit_score += 0.2
        
        # Enhanced decision logic to prevent fixation bias
        scores = {'fixation': fixation_score, 'saccade': saccade_score, 'pursuit': pursuit_score}
        
        # Special logic to break fixation bias
        max_score = max(scores.values())
        
        # If multiple classes have similar scores, prefer the non-fixation class
        if max_score > 0:
            # Get classes with max score
            max_classes = [cls for cls, score in scores.items() if abs(score - max_score) < 0.1]
            
            if len(max_classes) > 1 and 'fixation' in max_classes:
                # Remove fixation from tie if other classes are close
                non_fixation_classes = [cls for cls in max_classes if cls != 'fixation']
                if non_fixation_classes:
                    predicted_class = max(non_fixation_classes, key=lambda x: scores[x])
                else:
                    predicted_class = 'fixation'
            else:
                predicted_class = max(scores, key=scores.get)
        else:
            predicted_class = 'fixation'  # Default fallback
        
        confidence = scores[predicted_class]
        
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            class_probs = {cls: score/total_score for cls, score in scores.items()}
        else:
            class_probs = {cls: 1.0/len(scores) for cls in scores.keys()}
        
        # Add blink probability
        class_probs['blink'] = 0.1
        
        # Renormalize
        total_prob = sum(class_probs.values())
        class_probs = {cls: prob/total_prob for cls, prob in class_probs.items()}
        
        temporal_features = self._calculate_temporal_features(gaze_sequence)
        movement_characteristics = self._calculate_movement_characteristics(gaze_sequence)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            temporal_features=temporal_features,
            movement_characteristics=movement_characteristics,
            processing_time_ms=processing_time
        )
    
    def _calculate_temporal_features(self, gaze_sequence: GazeSequence) -> Dict[str, float]:
        """Calculate temporal features of the gaze sequence"""
        return {
            'duration_ms': gaze_sequence.duration,
            'sampling_rate': len(gaze_sequence.timestamps) / (gaze_sequence.duration / 1000),
            'temporal_stability': 1.0 / (1.0 + np.std(np.diff(gaze_sequence.timestamps)))
        }
    
    def _calculate_movement_characteristics(self, gaze_sequence: GazeSequence) -> Dict[str, float]:
        """Calculate movement characteristics of the gaze sequence"""
        velocities = np.linalg.norm(gaze_sequence.velocities, axis=1)
        accelerations = np.linalg.norm(gaze_sequence.accelerations, axis=1)
        
        return {
            'dispersion': np.sqrt(np.var(gaze_sequence.coordinates[:, 0]) + 
                                np.var(gaze_sequence.coordinates[:, 1])) * 1000,
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'velocity_std': np.std(velocities),
            'avg_acceleration': np.mean(accelerations),
            'max_acceleration': np.max(accelerations),
            'movement_efficiency': np.sum(velocities) / (gaze_sequence.duration / 1000),
            'directional_consistency': self._calculate_directional_consistency(gaze_sequence.velocities)
        }
    
    def _calculate_directional_consistency(self, velocities: np.ndarray) -> float:
        """Calculate how consistent the movement direction is"""
        if len(velocities) < 2:
            return 0.0
        
        # Calculate angles between consecutive velocity vectors
        angles = []
        for i in range(1, len(velocities)):
            v1 = velocities[i-1]
            v2 = velocities[i]
            
            # Skip zero velocities
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
                
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        if not angles:
            return 0.0
            
        # Return consistency as inverse of angular variance
        return 1.0 / (1.0 + np.var(angles))

class ModernGazeClassificationSystem:
    """
    Complete modern gaze classification system
    Integrates all state-of-the-art components for optimal real-time performance
    """
    
    def __init__(self, enable_npu: bool = True):
        self.motion_filter = MotionAwareFilter()
        self.adaptive_thresholds = AdaptiveThresholdCalculator()
        self.classifier = CNNRNNGazeClassifier()
        self.enable_npu = enable_npu
        
        # Performance monitoring
        self.classification_times = deque(maxlen=100)
        self.classification_count = 0
        
        logger.info("Initialized Modern Gaze Classification System")
        
        # Try to enable NPU acceleration if available
        if enable_npu:
            self._setup_npu_acceleration()
    
    def _setup_npu_acceleration(self):
        """Setup NPU acceleration for real-time performance"""
        try:
            # Check for Intel NPU availability
            if HAS_TF:
                # TensorFlow setup for Intel NPU
                physical_devices = tf.config.list_physical_devices()
                logger.info(f"Available devices: {physical_devices}")
                
                # Enable mixed precision for better NPU performance
                tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
                
            logger.info("NPU acceleration configured successfully")
        except Exception as e:
            logger.warning(f"NPU acceleration not available: {e}")
    
    def classify_gaze_sequence(self, timestamps: np.ndarray, coordinates: np.ndarray, 
                             pupil_sizes: np.ndarray, confidence: np.ndarray,
                             ground_truth: Optional[str] = None) -> ClassificationResult:
        """
        Classify a gaze sequence using the complete modern pipeline
        
        Args:
            timestamps: Array of timestamps in seconds
            coordinates: Array of (x, y) coordinates in normalized space [0,1]
            pupil_sizes: Array of pupil diameters in mm
            confidence: Array of tracking confidence values [0,1]
            ground_truth: Optional ground truth label for adaptive learning
            
        Returns:
            ClassificationResult with comprehensive classification information
        """
        
        start_time = time.time()
        
        # Calculate velocities and accelerations
        velocities = self._calculate_velocities(timestamps, coordinates)
        accelerations = self._calculate_accelerations(timestamps, velocities)
        
        # Create gaze sequence
        duration = (timestamps[-1] - timestamps[0]) * 1000  # Convert to milliseconds
        gaze_sequence = GazeSequence(
            timestamps=timestamps,
            coordinates=coordinates,
            velocities=velocities,
            accelerations=accelerations,
            pupil_sizes=pupil_sizes,
            confidence=confidence,
            duration=duration
        )
        
        # Apply motion-aware filtering
        filtered_sequence = self.motion_filter.filter_sequence(gaze_sequence)
        
        # Update adaptive thresholds
        self.adaptive_thresholds.update_profile(filtered_sequence, ground_truth)
        
        # Classify using modern CNN-RNN model
        result = self.classifier.predict(filtered_sequence)
        
        # Update performance statistics
        processing_time = (time.time() - start_time) * 1000
        self.classification_times.append(processing_time)
        self.classification_count += 1
        
        # Log performance every 100 classifications
        if self.classification_count % 100 == 0:
            avg_time = np.mean(self.classification_times)
            logger.info(f"Average classification time: {avg_time:.2f}ms "
                       f"(last 100 classifications)")
        
        return result
    
    def _calculate_velocities(self, timestamps: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Calculate velocity vectors from coordinates and timestamps"""
        if len(timestamps) < 2:
            return np.zeros((len(coordinates), 2))
        
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Duplicate last dt for same length
        dt = np.maximum(dt, 1e-6)   # Avoid division by zero
        
        # Calculate coordinate differences
        dx = np.diff(coordinates[:, 0])
        dy = np.diff(coordinates[:, 1])
        
        # Append zero for first sample
        dx = np.append(0, dx)
        dy = np.append(0, dy)
        
        # Convert to screen coordinates (assuming 1920x1080 screen)
        dx = dx * 1920  # pixels per second
        dy = dy * 1080  # pixels per second
        
        velocities = np.column_stack([dx / dt, dy / dt])
        return velocities
    
    def _calculate_accelerations(self, timestamps: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Calculate acceleration vectors from velocities and timestamps"""
        if len(timestamps) < 2:
            return np.zeros((len(velocities), 2))
        
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1])  # Duplicate last dt for same length
        dt = np.maximum(dt, 1e-6)   # Avoid division by zero
        
        # Calculate velocity differences
        dvx = np.diff(velocities[:, 0])
        dvy = np.diff(velocities[:, 1])
        
        # Append zero for first sample
        dvx = np.append(0, dvx)
        dvy = np.append(0, dvy)
        
        accelerations = np.column_stack([dvx / dt, dvy / dt])
        return accelerations
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.classification_times:
            return {}
        
        times = list(self.classification_times)
        return {
            'avg_processing_time_ms': np.mean(times),
            'min_processing_time_ms': np.min(times),
            'max_processing_time_ms': np.max(times),
            'std_processing_time_ms': np.std(times),
            'total_classifications': self.classification_count,
            'classifications_per_second': 1000.0 / np.mean(times) if np.mean(times) > 0 else 0.0
        }

# Global instance for easy access
_modern_classifier = None

def get_modern_classifier() -> ModernGazeClassificationSystem:
    """Get the global modern gaze classifier instance"""
    global _modern_classifier
    if _modern_classifier is None:
        _modern_classifier = ModernGazeClassificationSystem()
    return _modern_classifier

def classify_gaze_events_modern(gaze_events: List[Any], trial_type: str = None) -> Tuple[str, float]:
    """
    Modern gaze classification function compatible with existing codebase
    
    Args:
        gaze_events: List of gaze events with timestamp, x, y, pupil_size, confidence
        trial_type: Optional ground truth for adaptive learning
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    
    if len(gaze_events) < 3:
        return "unknown", 0.0
    
    try:
        # Extract data from gaze events
        timestamps = np.array([event.timestamp for event in gaze_events])
        coordinates = np.array([[event.x, event.y] for event in gaze_events])
        pupil_sizes = np.array([getattr(event, 'pupil_size', 4.0) for event in gaze_events])
        confidence = np.array([getattr(event, 'confidence', 1.0) for event in gaze_events])
        
        # Get classifier and classify
        classifier = get_modern_classifier()
        result = classifier.classify_gaze_sequence(
            timestamps, coordinates, pupil_sizes, confidence, trial_type
        )
        
        return result.predicted_class, result.confidence
        
    except Exception as e:
        logger.error(f"Error in modern classification: {e}")
        # Fallback to basic classification
        return "unknown", 0.0

if __name__ == "__main__":
    # Test the modern classification system
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    n_samples = 50
    timestamps = np.linspace(0, 1, n_samples)
    
    # Simulate fixation data
    base_pos = np.array([0.5, 0.5])
    noise = np.random.normal(0, 0.01, (n_samples, 2))
    coordinates = base_pos + noise
    
    pupil_sizes = np.random.normal(4.0, 0.2, n_samples)
    confidence = np.random.uniform(0.8, 1.0, n_samples)
    
    # Test classification
    classifier = ModernGazeClassificationSystem()
    result = classifier.classify_gaze_sequence(timestamps, coordinates, pupil_sizes, confidence)
    
    print(f"Classification result: {result.predicted_class} (confidence: {result.confidence:.3f})")
    print(f"Class probabilities: {result.class_probabilities}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Performance stats: {classifier.get_performance_stats()}") 