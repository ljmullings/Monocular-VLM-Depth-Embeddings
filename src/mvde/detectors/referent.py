"""Text referent selection for identifying objects from natural language."""

from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass

from .yolo import BoundingBox


@dataclass
class ReferentMatch:
    """Represents a match between text and a detected object."""
    object_box: BoundingBox
    confidence: float
    matched_text: str
    reasoning: str


class ReferentSelector:
    """Select objects based on natural language descriptions."""
    
    def __init__(self):
        """Initialize referent selector."""
        # Common spatial relationship keywords
        self.spatial_keywords = {
            "left": ["left", "leftmost", "on the left"],
            "right": ["right", "rightmost", "on the right"], 
            "center": ["center", "middle", "central"],
            "top": ["top", "upper", "above"],
            "bottom": ["bottom", "lower", "below"],
            "front": ["front", "foreground", "closest", "nearest"],
            "back": ["back", "background", "farthest", "distant"],
        }
        
        # Color keywords
        self.color_keywords = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "gray", "grey", "brown", "silver", "gold"
        ]
        
        # Size keywords  
        self.size_keywords = {
            "large": ["large", "big", "huge", "giant"],
            "small": ["small", "tiny", "little", "mini"],
        }
    
    def select_referent(
        self, 
        text: str, 
        detected_objects: List[BoundingBox],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[ReferentMatch]:
        """
        Select the most likely object referent from text description.
        
        Args:
            text: Natural language description
            detected_objects: List of detected objects
            image_size: Optional image dimensions for spatial reasoning
            
        Returns:
            Best matching object or None
        """
        if not detected_objects:
            return None
        
        text_lower = text.lower()
        candidates = []
        
        # Extract potential object names and attributes
        object_matches = self._match_object_classes(text_lower, detected_objects)
        
        for obj in object_matches:
            confidence = 0.0
            reasoning_parts = []
            
            # Class name match
            if obj.class_name.lower() in text_lower:
                confidence += 0.5
                reasoning_parts.append(f"class match: {obj.class_name}")
            
            # Color matching (if we had color detection)
            color_score = self._match_color(text_lower)
            if color_score > 0:
                confidence += 0.2 * color_score
                reasoning_parts.append("color mentioned")
            
            # Spatial positioning
            if image_size:
                spatial_score = self._match_spatial_position(
                    text_lower, obj, detected_objects, image_size
                )
                confidence += 0.3 * spatial_score
                if spatial_score > 0:
                    reasoning_parts.append("spatial match")
            
            # Size matching
            size_score = self._match_size(text_lower, obj, detected_objects)
            confidence += 0.2 * size_score
            if size_score > 0:
                reasoning_parts.append("size match")
            
            # Boost confidence for higher detection confidence
            confidence *= (0.5 + 0.5 * obj.confidence)
            
            candidates.append(ReferentMatch(
                object_box=obj,
                confidence=confidence,
                matched_text=text,
                reasoning="; ".join(reasoning_parts)
            ))
        
        # Return highest confidence match
        if candidates:
            best_match = max(candidates, key=lambda x: x.confidence)
            return best_match if best_match.confidence > 0.1 else None
        
        return None
    
    def _match_object_classes(
        self, 
        text: str, 
        objects: List[BoundingBox]
    ) -> List[BoundingBox]:
        """Find objects whose class names appear in the text."""
        matches = []
        for obj in objects:
            if obj.class_name.lower() in text:
                matches.append(obj)
        
        # If no exact class matches, return all objects
        return matches if matches else objects
    
    def _match_color(self, text: str) -> float:
        """Check if color keywords are mentioned in text."""
        for color in self.color_keywords:
            if color in text:
                return 1.0
        return 0.0
    
    def _match_spatial_position(
        self, 
        text: str, 
        target_obj: BoundingBox, 
        all_objects: List[BoundingBox],
        image_size: Tuple[int, int]
    ) -> float:
        """Score spatial position matches."""
        img_width, img_height = image_size
        center_x, center_y = target_obj.center
        
        score = 0.0
        
        # Left/right positioning
        if any(keyword in text for keyword in self.spatial_keywords["left"]):
            if center_x < img_width * 0.33:
                score += 1.0
        elif any(keyword in text for keyword in self.spatial_keywords["right"]):
            if center_x > img_width * 0.67:
                score += 1.0
        elif any(keyword in text for keyword in self.spatial_keywords["center"]):
            if 0.33 * img_width < center_x < 0.67 * img_width:
                score += 1.0
        
        # Top/bottom positioning
        if any(keyword in text for keyword in self.spatial_keywords["top"]):
            if center_y < img_height * 0.33:
                score += 1.0
        elif any(keyword in text for keyword in self.spatial_keywords["bottom"]):
            if center_y > img_height * 0.67:
                score += 1.0
        
        return min(score, 1.0)
    
    def _match_size(
        self, 
        text: str, 
        target_obj: BoundingBox, 
        all_objects: List[BoundingBox]
    ) -> float:
        """Score size-based matches."""
        if len(all_objects) < 2:
            return 0.0
        
        # Calculate relative size
        areas = [obj.area for obj in all_objects]
        target_area = target_obj.area
        area_rank = sorted(areas, reverse=True).index(target_area)
        
        # Large object keywords
        if any(keyword in text for keyword in self.size_keywords["large"]):
            if area_rank < len(areas) * 0.3:  # Top 30% by size
                return 1.0
        
        # Small object keywords
        if any(keyword in text for keyword in self.size_keywords["small"]):
            if area_rank > len(areas) * 0.7:  # Bottom 30% by size
                return 1.0
        
        return 0.0
    
    def get_all_candidates(
        self, 
        text: str, 
        detected_objects: List[BoundingBox],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[ReferentMatch]:
        """Get all candidate matches sorted by confidence."""
        # TODO: Implement to return all candidates, not just the best
        best_match = self.select_referent(text, detected_objects, image_size)
        return [best_match] if best_match else []
