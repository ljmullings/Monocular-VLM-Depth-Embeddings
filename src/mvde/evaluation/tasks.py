"""QA-style evaluation tasks for depth reasoning."""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class QAExample:
    """A question-answer example for evaluation."""
    image_path: str
    question: str
    answer: str
    question_type: str  # "distance", "relative", "counting", etc.
    metadata: Dict[str, Any] = None


class QAEvaluator:
    """Evaluator for question-answering tasks involving depth reasoning."""
    
    def __init__(self):
        """Initialize QA evaluator."""
        self.question_types = {
            "distance": "How far is the {object}?",
            "relative": "Which is closer, the {object1} or the {object2}?",
            "counting": "How many {objects} are within {distance} meters?",
            "spatial": "What objects are to the left of the {object}?",
        }
    
    def create_distance_questions(
        self,
        objects: List[Dict[str, Any]],
        image_path: str,
    ) -> List[QAExample]:
        """Create distance estimation questions."""
        questions = []
        
        for obj in objects:
            question = f"How far is the {obj['class_name']}?"
            answer = f"{obj['distance']:.1f} meters"
            
            questions.append(QAExample(
                image_path=image_path,
                question=question,
                answer=answer,
                question_type="distance",
                metadata={"object": obj}
            ))
        
        return questions
    
    def create_relative_questions(
        self,
        objects: List[Dict[str, Any]],
        image_path: str,
    ) -> List[QAExample]:
        """Create relative distance questions."""
        questions = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i + 1:], i + 1):
                question = f"Which is closer, the {obj1['class_name']} or the {obj2['class_name']}?"
                
                if obj1['distance'] < obj2['distance']:
                    answer = obj1['class_name']
                else:
                    answer = obj2['class_name']
                
                questions.append(QAExample(
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    question_type="relative",
                    metadata={"object1": obj1, "object2": obj2}
                ))
        
        return questions
    
    def create_counting_questions(
        self,
        objects: List[Dict[str, Any]],
        image_path: str,
        distance_thresholds: List[float] = [5.0, 10.0, 20.0],
    ) -> List[QAExample]:
        """Create counting questions based on distance."""
        questions = []
        
        # Group objects by class
        class_groups = {}
        for obj in objects:
            class_name = obj['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(obj)
        
        for class_name, class_objects in class_groups.items():
            if len(class_objects) > 1:  # Only create questions for multi-instance classes
                for threshold in distance_thresholds:
                    count = sum(1 for obj in class_objects if obj['distance'] <= threshold)
                    
                    question = f"How many {class_name}s are within {threshold} meters?"
                    answer = str(count)
                    
                    questions.append(QAExample(
                        image_path=image_path,
                        question=question,
                        answer=answer,
                        question_type="counting",
                        metadata={
                            "class_name": class_name,
                            "threshold": threshold,
                            "objects": class_objects
                        }
                    ))
        
        return questions
    
    def evaluate_answers(
        self,
        predictions: List[str],
        ground_truth: List[str],
        question_types: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate predicted answers against ground truth.
        
        Args:
            predictions: Predicted answers
            ground_truth: Ground truth answers
            question_types: Types of questions
            
        Returns:
            Evaluation metrics by question type
        """
        results = {
            "overall_accuracy": 0.0,
            "distance_accuracy": 0.0,
            "relative_accuracy": 0.0,
            "counting_accuracy": 0.0,
        }
        
        if len(predictions) != len(ground_truth):
            return results
        
        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if self._match_answer(p, g))
        results["overall_accuracy"] = correct / len(predictions)
        
        # Per-type accuracy
        for question_type in ["distance", "relative", "counting"]:
            type_indices = [i for i, qt in enumerate(question_types) if qt == question_type]
            
            if type_indices:
                type_correct = sum(
                    1 for i in type_indices 
                    if self._match_answer(predictions[i], ground_truth[i])
                )
                results[f"{question_type}_accuracy"] = type_correct / len(type_indices)
        
        return results
    
    def _match_answer(self, prediction: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_clean = prediction.lower().strip()
        gt_clean = ground_truth.lower().strip()
        
        # Exact match
        if pred_clean == gt_clean:
            return True
        
        # Fuzzy matching for distance answers
        if "meter" in gt_clean:
            try:
                pred_value = float(pred_clean.split()[0])
                gt_value = float(gt_clean.split()[0])
                # Allow 10% tolerance for distance estimates
                return abs(pred_value - gt_value) / gt_value < 0.1
            except:
                return False
        
        # Partial matching for object names
        if gt_clean in pred_clean or pred_clean in gt_clean:
            return True
        
        return False
    
    def generate_evaluation_dataset(
        self,
        image_objects_list: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> List[QAExample]:
        """Generate a complete evaluation dataset."""
        all_questions = []
        
        for image_path, objects in image_objects_list:
            # Distance questions
            all_questions.extend(
                self.create_distance_questions(objects, image_path)
            )
            
            # Relative questions
            all_questions.extend(
                self.create_relative_questions(objects, image_path)
            )
            
            # Counting questions
            all_questions.extend(
                self.create_counting_questions(objects, image_path)
            )
        
        return all_questions
