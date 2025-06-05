import numpy as np
import json
import random
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict
from typing import List, Dict, Tuple, Any

class RealisticClassifierErrorSimulator:
    """
    Simulates realistic classifier errors to achieve target F1 scores.
    Models sliding window classifier with class-specific confusion patterns.
    """
    
    def __init__(self, window_size_seconds: float = 2.0, overlap_ratio: float = 0.9):
        self.window_size = window_size_seconds
        self.overlap_ratio = overlap_ratio
        self.step_size = window_size_seconds * (1 - overlap_ratio)  # 0.2 seconds
        
        # Define realistic class confusion patterns
        self.class_confusions = {
            'faceoff': ['groundball', 'pass'],  # Often confused with ground scrambles
            'pass': ['shot', 'catch'],          # Motion similarity
            'catch': ['groundball', 'pass'],    # Hand movements similar
            'shot': ['pass', 'groundball'],     # Throwing motions
            'groundball': ['catch', 'faceoff'], # Low positioning similar
            'save_attempt': ['groundball', 'catch']  # Goalie movements
        }
        
        # Class-specific error tendencies (some classes harder to detect)
        self.class_difficulty = {
            'faceoff': 0.85,      # Easier - distinct starting position
            'shot': 0.80,         # Easier - clear throwing motion
            'save_attempt': 0.82, # Easier - goalie specific
            'pass': 0.75,         # Medium - can look like shot
            'catch': 0.70,        # Harder - quick hand movements
            'groundball': 0.65    # Hardest - often obscured/quick
        }

    def create_sliding_windows(self, events: List[Dict], game_duration: float) -> List[Dict]:
        """Convert events to sliding window format that mimics classifier input."""
        windows = []
        current_time = 0.0
        
        while current_time < game_duration:
            window_end = current_time + self.window_size
            
            # Find events in this window
            window_events = [
                e for e in events 
                if current_time <= e['timestamp'] < window_end
            ]
            
            # For simplicity, take the most prominent action in window
            # (Real classifier would use features from entire window)
            primary_action = None
            primary_player = None
            primary_facing = None
            
            if window_events:
                # Use the event closest to window center
                window_center = current_time + self.window_size / 2
                closest_event = min(window_events, 
                                  key=lambda e: abs(e['timestamp'] - window_center))
                primary_action = closest_event['action']
                primary_player = closest_event['player']
                primary_facing = closest_event['facing']
            
            windows.append({
                'window_start': current_time,
                'window_end': window_end,
                'predicted_action': primary_action,
                'predicted_player': primary_player,
                'predicted_facing': primary_facing,
                'confidence': random.uniform(0.6, 0.95) if primary_action else 0.1
            })
            
            current_time += self.step_size
        
        return windows

    def generate_confusion_matrix(self, target_f1: float, classes: List[str]) -> np.ndarray:
        """Generate a realistic confusion matrix for target F1 score."""
        n_classes = len(classes)
        confusion = np.zeros((n_classes, n_classes))
        
        for i, true_class in enumerate(classes):
            base_accuracy = self.class_difficulty.get(true_class, 0.75)
            
            # Adjust base accuracy to achieve target F1
            # F1 is roughly related to diagonal dominance
            adjusted_accuracy = self._adjust_accuracy_for_f1(base_accuracy, target_f1)
            
            # Diagonal (correct predictions)
            confusion[i, i] = adjusted_accuracy
            
            # Off-diagonal (errors) - distribute based on class confusions
            remaining_mass = 1.0 - adjusted_accuracy
            
            if true_class in self.class_confusions:
                confused_classes = self.class_confusions[true_class]
                # Distribute errors primarily among confused classes
                for confused_class in confused_classes:
                    if confused_class in classes:
                        j = classes.index(confused_class)
                        confusion[i, j] = remaining_mass / len(confused_classes) * 0.7
                
                # Distribute remaining error mass to other classes
                other_classes = [c for c in classes if c not in confused_classes and c != true_class]
                if other_classes:
                    remaining_error = remaining_mass * 0.3
                    for other_class in other_classes:
                        j = classes.index(other_class)
                        confusion[i, j] = remaining_error / len(other_classes)
            else:
                # Uniform distribution if no specific confusions
                for j in range(n_classes):
                    if i != j:
                        confusion[i, j] = remaining_mass / (n_classes - 1)
        
        # Normalize rows to sum to 1
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
        return confusion

    def _adjust_accuracy_for_f1(self, base_accuracy: float, target_f1: float) -> float:
        """Adjust per-class accuracy to approximate target micro F1."""
        # Simplified relationship: F1 ≈ (precision + recall) / 2 ≈ accuracy for balanced data
        return min(0.95, max(0.1, base_accuracy * (target_f1 / 0.8)))

    def apply_classifier_errors(self, windows: List[Dict], target_f1: float, 
                              all_actions: List[str], all_players: List[str]) -> List[Dict]:
        """Apply realistic classifier errors to achieve target F1 score."""
        
        # Extract non-null actions for confusion matrix
        valid_windows = [w for w in windows if w['predicted_action'] is not None]
        if not valid_windows:
            return windows
        
        actions = [w['predicted_action'] for w in valid_windows]
        unique_actions = list(set(actions))
        
        # Generate confusion matrix
        confusion_matrix = self.generate_confusion_matrix(target_f1, unique_actions)
        
        # Apply errors to each window
        corrupted_windows = []
        for window in windows:
            if window['predicted_action'] is None:
                # Handle null predictions - randomly assign or keep null
                if random.random() < (1 - target_f1):  # More nulls = lower F1
                    window_copy = window.copy()
                    window_copy['predicted_action'] = random.choice(all_actions)
                    window_copy['predicted_player'] = random.choice(all_players)
                    window_copy['confidence'] = random.uniform(0.3, 0.7)
                    corrupted_windows.append(window_copy)
                else:
                    corrupted_windows.append(window.copy())
            else:
                # Apply confusion matrix
                true_action = window['predicted_action']
                if true_action in unique_actions:
                    true_idx = unique_actions.index(true_action)
                    
                    # Sample from confusion matrix row
                    predicted_idx = np.random.choice(
                        len(unique_actions), 
                        p=confusion_matrix[true_idx]
                    )
                    predicted_action = unique_actions[predicted_idx]
                    
                    window_copy = window.copy()
                    window_copy['predicted_action'] = predicted_action
                    
                    # If action changed, potentially change player/facing too
                    if predicted_action != true_action:
                        if random.random() < 0.3:  # 30% chance to also misclassify player
                            window_copy['predicted_player'] = random.choice(all_players)
                        if random.random() < 0.2:  # 20% chance to misclassify facing
                            facing_options = [p for p in all_players if p != window_copy['predicted_player']]
                            if facing_options:
                                window_copy['predicted_facing'] = random.choice(facing_options)
                        window_copy['confidence'] = random.uniform(0.4, 0.8)
                    
                    corrupted_windows.append(window_copy)
                else:
                    corrupted_windows.append(window.copy())
        
        return corrupted_windows

    def windows_to_events(self, windows: List[Dict]) -> List[Dict]:
        """Convert windowed predictions back to event format."""
        events = []
        
        for window in windows:
            if window['predicted_action'] is not None:
                # Use window center as timestamp
                timestamp = window['window_start'] + self.window_size / 2
                
                event = {
                    'timestamp': round(timestamp, 2),
                    'player': window['predicted_player'],
                    'action': window['predicted_action'],
                    'facing': window['predicted_facing'],
                    'confidence': window.get('confidence', 0.5)
                }
                events.append(event)
        
        # Remove duplicates and sort
        events = self._deduplicate_events(events)
        events.sort(key=lambda e: e['timestamp'])
        
        return events

    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Remove near-duplicate events (within 0.5 seconds with same action/player)."""
        if not events:
            return events
        
        deduplicated = [events[0]]
        
        for current_event in events[1:]:
            should_add = True
            
            for existing_event in deduplicated:
                time_diff = abs(current_event['timestamp'] - existing_event['timestamp'])
                same_player = current_event['player'] == existing_event['player']
                same_action = current_event['action'] == existing_event['action']
                
                if time_diff < 0.5 and same_player and same_action:
                    should_add = False
                    break
            
            if should_add:
                deduplicated.append(current_event)
        
        return deduplicated

    def simulate_classifier_predictions(self, ground_truth_events: List[Dict], 
                                      target_f1: float, game_duration: float,
                                      all_players: List[str], all_actions: List[str]) -> Tuple[List[Dict], Dict]:
        """
        Main method to simulate classifier predictions with target F1 score.
        Uses iterative direct event corruption to achieve precise F1 targets.
        """
        
        best_predictions = None
        best_f1_diff = float('inf')
        best_metadata = None
        
        # Try multiple iterations to get close to target F1
        for attempt in range(50):  # Increased iterations for better accuracy
            # Generate predicted events using direct corruption
            predicted_events = self._generate_corrupted_events(
                ground_truth_events, target_f1, game_duration, all_players, all_actions, attempt
            )
            
            # Calculate actual achieved F1 score
            actual_f1 = self._calculate_achieved_f1(ground_truth_events, predicted_events, all_actions)
            
            # Track best result
            f1_diff = abs(actual_f1 - target_f1)
            if f1_diff < best_f1_diff:
                best_f1_diff = f1_diff
                best_predictions = predicted_events
                best_metadata = {
                    'target_f1_score': target_f1,
                    'achieved_f1_score': actual_f1,
                    'original_event_count': len(ground_truth_events),
                    'predicted_event_count': len(predicted_events),
                    'simulation_method': 'direct_event_corruption_with_iterative_targeting',
                    'iterations_attempted': attempt + 1,
                    'f1_difference': f1_diff
                }
            
            # If we're close enough, stop early
            if f1_diff < 0.02:  # Within 2% of target
                break
        
        return best_predictions, best_metadata

    def _calculate_achieved_f1(self, ground_truth: List[Dict], predicted: List[Dict], 
                             all_actions: List[str]) -> float:
        """Calculate actual F1 score using proper event signature matching."""
        try:
            if not ground_truth and not predicted:
                return 1.0
            if not ground_truth or not predicted:
                return 0.0
            
            # Create event signatures for matching (time_bin, action, player)
            def event_signature(event, time_bin_size=0.5):
                time_bin = int(event['timestamp'] / time_bin_size)
                return (time_bin, event['action'], event['player'])
            
            gt_signatures = set(event_signature(e) for e in ground_truth)
            pred_signatures = set(event_signature(e) for e in predicted)
            
            # Calculate precision, recall, F1
            true_positives = len(gt_signatures & pred_signatures)
            false_positives = len(pred_signatures - gt_signatures)
            false_negatives = len(gt_signatures - pred_signatures)
            
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
        
        except Exception as e:
            print(f"Warning: Could not calculate F1 score: {e}")
            return 0.0

    def _generate_corrupted_events(self, ground_truth_events: List[Dict], target_f1: float,
                                 game_duration: float, all_players: List[str], 
                                 all_actions: List[str], attempt: int) -> List[Dict]:
        """Generate corrupted events using direct manipulation to achieve target F1."""
        
        predicted_events = []
        
        # Start with ground truth events and corrupt them
        for gt_event in ground_truth_events:
            # Determine if this event should be kept, corrupted, or dropped
            corruption_rate = self._get_corruption_rate(target_f1, attempt)
            
            if random.random() < corruption_rate:
                # Corrupt this event
                if random.random() < 0.3:  # 30% chance to drop the event entirely
                    continue  # Skip this event (false negative)
                else:
                    # Corrupt the event
                    corrupted_event = self._corrupt_event(gt_event, all_players, all_actions)
                    predicted_events.append(corrupted_event)
            else:
                # Keep the event unchanged
                predicted_events.append(gt_event.copy())
        
        # Add some false positive events
        false_positive_rate = self._get_false_positive_rate(target_f1, attempt)
        num_false_positives = int(len(ground_truth_events) * false_positive_rate)
        
        for _ in range(num_false_positives):
            player = random.choice(all_players)
            facing_options = [p for p in all_players if p != player]
            fake_event = {
                'timestamp': random.uniform(0, game_duration),
                'player': player,
                'action': random.choice(all_actions),
                'facing': random.choice(facing_options) if facing_options else player
            }
            predicted_events.append(fake_event)
        
        # Sort by timestamp
        predicted_events.sort(key=lambda e: e['timestamp'])
        
        return predicted_events

    def _get_corruption_rate(self, target_f1: float, attempt: int) -> float:
        """Calculate corruption rate to achieve target F1."""
        base_rate = 1.0 - target_f1  # Higher F1 = lower corruption
        
        # Add variation based on attempt to explore different rates
        variation = (attempt % 10) * 0.02 - 0.1  # Range from -0.1 to +0.08
        adjusted_rate = base_rate + variation
        
        return max(0.05, min(0.95, adjusted_rate))

    def _get_false_positive_rate(self, target_f1: float, attempt: int) -> float:
        """Calculate false positive rate to achieve target F1."""
        base_rate = (1.0 - target_f1) * 0.5  # Lower F1 = more false positives
        
        # Add variation based on attempt
        variation = (attempt % 20) * 0.01 - 0.1  # Range from -0.1 to +0.09
        adjusted_rate = base_rate + variation
        
        return max(0.0, min(0.8, adjusted_rate))

    def _corrupt_event(self, event: Dict, all_players: List[str], all_actions: List[str]) -> Dict:
        """Corrupt a single event based on realistic confusion patterns."""
        corrupted = event.copy()
        
        # Corrupt action based on class confusions
        if event['action'] in self.class_confusions:
            if random.random() < 0.7:  # 70% chance to use realistic confusion
                confused_actions = self.class_confusions[event['action']]
                if confused_actions:  # Make sure there are confused actions available
                    corrupted['action'] = random.choice(confused_actions)
                else:
                    corrupted['action'] = random.choice(all_actions)
            else:
                corrupted['action'] = random.choice(all_actions)
        else:
            corrupted['action'] = random.choice(all_actions)
        
        # Sometimes corrupt player too
        if random.random() < 0.2:
            corrupted['player'] = random.choice(all_players)
        
        # Sometimes corrupt facing
        if random.random() < 0.15:
            facing_options = [p for p in all_players if p != corrupted['player']]
            if facing_options:
                corrupted['facing'] = random.choice(facing_options)
        
        # Add small timing noise
        time_noise = random.uniform(-0.5, 0.5)
        corrupted['timestamp'] = max(0, event['timestamp'] + time_noise)
        
        return corrupted

# Example usage function
def generate_error_levels(ground_truth_file: str, output_dir: str = "."):
    """Generate multiple error levels from 50% to 100% F1 in 5% increments."""
    
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    simulator = RealisticClassifierErrorSimulator()
    
    all_actions = ground_truth_data['game_setup']['sim_actions']
    all_players = (ground_truth_data['game_setup']['team_A'] + 
                  ground_truth_data['game_setup']['team_B'] + 
                  [ground_truth_data['game_setup']['goalie']])
    
    f1_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for target_f1 in f1_levels:
        print(f"Generating predictions for F1 = {target_f1:.2f}")
        
        predicted_events, metadata = simulator.simulate_classifier_predictions(
            ground_truth_data['events'],
            target_f1,
            ground_truth_data['game_setup']['duration_seconds'],
            all_players,
            all_actions
        )
        
        # Create output data structure
        output_data = {
            'game_setup': ground_truth_data['game_setup'],
            'final_score': ground_truth_data['final_score'],
            'predicted_events': predicted_events,
            'metadata': metadata
        }
        
        # Save to file
        output_filename = f"{output_dir}/2v2_predictions_f1_{int(target_f1*100):02d}.json"
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"  → Saved {len(predicted_events)} predicted events to {output_filename}")
        print(f"  → Achieved F1: {metadata['achieved_f1_score']:.3f} (target: {target_f1:.3f}, diff: {metadata['f1_difference']:.3f})")

if __name__ == "__main__":
    generate_error_levels("2v2_groundtruth.json") 