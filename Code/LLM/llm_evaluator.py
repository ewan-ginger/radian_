import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
import os
import random
from collections import defaultdict, Counter
import copy

# LangChain Google Generative AI imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    print("Warning: LangChain Google Generative AI not available. Install with: pip install langchain-google-genai")
    ChatGoogleGenerativeAI = None
    HumanMessage = SystemMessage = None

class LacrosseLLMEvaluator:
    """
    Evaluates LLM's ability to reconstruct ground truth gameplay from noisy predictions.
    Uses Google Gemini via LangChain integration.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20", temperature: float = 0.0, test_mode: bool = False, consensus_runs: int = 1, similarity_threshold: float = 0.8):
        if test_mode:
            # Test mode: simulate ideal LLM behavior
            self.llm = None
            self.test_mode = True
            print("Running in TEST MODE: Simulating ideal conservative LLM corrections")
        elif ChatGoogleGenerativeAI is None:
            raise ImportError("LangChain Google Generative AI not available. Please install langchain-google-genai")
        else:
            # Check for Google API key
            if not os.getenv('GOOGLE_API_KEY'):
                print("Warning: GOOGLE_API_KEY not found in environment variables")
                print("You can set it with: export GOOGLE_API_KEY='your-key-here'")
                print("For now, creating evaluator without LLM (metrics only)")
                self.llm = None
                self.test_mode = False
            else:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=None,
                        timeout=None,
                        max_retries=3,
                        # Convert older messages format for better compatibility
                        convert_system_message_to_human=True
                    )
                    self.test_mode = False
                    
                except Exception as e:
                    print(f"Warning: Could not initialize ChatGoogleGenerativeAI: {e}")
                    self.llm = None
                    self.test_mode = False
        
        # Consensus parameters
        self.consensus_runs = consensus_runs
        self.similarity_threshold = similarity_threshold
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        return """You are a meticulous and **conservative** Lacrosse Game Analyst AI. Your task is to critically evaluate a sequence of predicted lacrosse game events, identify errors by following a specific step-by-step analysis process provided by the user, and then reconstruct an accurate and logically sound gameplay sequence.

**Your Guiding Principles for Correction:**
1.  **Accuracy and Logic:** The reconstructed sequence must be accurate according to lacrosse rules and logically coherent.
2.  **Conservatism & Minimal Change:** This is PARAMOUNT. Make only the **fewest necessary changes** to the `predicted_events`. It is better to leave a slightly imperfect event or a minor inconsistency than to make a speculative change that introduces new errors. Prefer modifying existing events over deleting or adding new ones.
3.  **High-Confidence Corrections:** Only make changes you are highly confident will fix a clear, verifiable error based on the provided lacrosse logic. Avoid guesswork. If a potential issue is ambiguous, or the correction is not obvious, lean towards keeping the original event.
4.  **Fidelity to Input:** Preserve as much of the original `predicted_events` as possible. Your goal is refinement, not a complete rewrite.
5.  **Justification:** All changes must be clearly justified based on the lacrosse rules and these principles in your analysis.

Your ultimate goal is to **modestly improve the accuracy and coherence** of the event list by applying these principles, primarily by fixing clear, unambiguous errors.

Game Structure:
Starts with a faceoff (or can after a goal).
Possession is key. Only the player with possession (or goalie for saves) can initiate most actions.
Goals reset possession to a faceoff.

Action Sequences & Logic:
Faceoff: Two players (one from each team). Followed by a groundball to determine possession. The facing field indicates the opponent.
Pass: Player A passes to Player B (teammate). facing is Player B.
Catch: Player B catches from Player A. facing is Player A. A pass should almost always be followed by a catch by the intended receiver or a groundball if the pass is incomplete, very shortly after.
Shot: An offensive player shoots. facing is always the Goalie.
Save Attempt: Made by the Goalie. facing is the shooter. Follows a shot very closely. Can result in a "Save" (goalie gains possession) or "Goal".
Groundball: A player picks up a loose ball.
After faceoff: One of the faceoff players or a teammate/opponent.
After missed shot: Often the shooter, a teammate, an opponent, or the goalie. facing is often the player who shot or a player involved in the contest for the ball.
After incomplete pass: Can be either team. facing is often the passer.
After turnover (lost ball): facing is the player who recovered it, or who they took it from.
Possession Chain: Events should logically follow possession. If Player_1 has the ball, the next action involving the ball should be by Player_1 (pass, shot) or against Player_1 (turnover leading to groundball).

Timestamps:
Events are chronological.
Pass -> Catch: Very short delay (e.g., 0.1-0.5s).
Shot -> Save Attempt: Very short delay (e.g., 0.1-0.6s).
Events like "movement" are not explicitly logged but implied between actions with larger time gaps.

Team Logic:
Players on team_A pass to other players on team_A.
Possession dictates offensive/defensive roles, though this isn't explicitly in the event log, it's implied by who is shooting.

Goal and Score Consistency (Internal Logic):
A goal is typically inferred when an offensive player's `shot` is followed by a `save_attempt` by the `Goalie`, and the `Goalie` does *not* immediately gain possession (e.g., via a `groundball` or `pass` by the goalie within a very short timeframe, like ~1-2 seconds).
A successfully inferred goal should then be followed by a `faceoff` to restart play.
During your analysis, especially in STEP 6, consider if the sequence of events implies goals consistently with this logic. The number of inferred goals should correspond to the number of subsequent restart `faceoff` events (excluding the initial game start faceoff). If inferring a goal based on shot/save_attempt outcome would require many speculative or drastic changes to the event sequence, prioritize the local logical flow of events and the principle of minimal change. You can note any remaining logical discrepancies around goal scoring or faceoffs in your analysis, but do not invent goals or saves without strong contextual evidence from the events themselves.

Error Types to Look For (based on your prediction file):
Misclassification: shot becomes save_attempt (e.g., timestamp 23.41 GT vs. 23.73 Pred).
Incorrect facing: save_attempt facing Player_3 instead of the shooter.
Missing events: A catch after a pass.
Spurious events: An action that doesn't fit the sequence.
Timestamp drift: Minor shifts are okay, but large ones breaking sequences are problematic.
Incorrect player for action: e.g., a field player making a save_attempt.
When these are identified, apply your conservative correction strategy.

**Core Task:**
1.  Strictly follow the user's "CHAIN OF THOUGHT ANALYSIS" steps (STEP 1 through STEP 6). Adhere to the detailed instructions within each step, especially the conservative correction strategy in STEP 6.
2.  Use your lacrosse knowledge (summarized below) and the **Guiding Principles for Correction** (above) to inform your analysis and decisions, especially for STEP 4 (Missing Critical Events), STEP 5 (Sequence Validation), and STEP 6 (Correction Decisions).
3.  Based on your analysis and decisions, populate the "analysis" section of the output JSON.
4.  Generate the `reconstructed_events` list, reflecting only high-confidence, justified, and minimal corrections made.
5.  Accurately report the `corrections_made` (counts and detailed descriptions of each change) in the output JSON.
6.  The reconstructed event sequence should strive for logical consistency regarding game progression (e.g., goals leading to faceoffs), but not at the expense of making speculative or drastic changes to the events themselves.

**Essential Lacrosse Logic Reference (Use this to guide your CORRECTION DECISIONS in STEP 6):**
When analyzing potential errors and deciding on corrections, refer to this logic. The goal is to ensure the `reconstructed_events` adhere to these principles, but corrections should be made conservatively, prioritizing minimal changes.

*   **Game Structure & Teams:**
    *   Teams are `team_A` and `team_B`. The `goalie` defends against the offensive team.
    *   Actions available: `faceoff`, `pass`, `catch`, `shot`, `save_attempt`, `groundball`.

*   **Event Sequences & Possession:**
    *   **Faceoff:**
        *   Occurs at game start and after inferred goals.
        *   Involves one player from each team (e.g., `Player_A` vs `Player_B`). Both will have a `faceoff` action at the same timestamp, `facing` each other.
        *   Is IMMEDIATELY followed by a `groundball` event to determine possession. The player who wins this `groundball` gains possession. If a `faceoff` event is present but the subsequent `groundball` to establish possession is clearly missing and no other event explains possession, adding the `groundball` by one of the faceoff players or a nearby player is a high-confidence correction if it resolves a clear break in logic.
    *   **Pass & Catch:**
        *   A `pass` by PlayerX (`facing` TeammateY) should be followed by a `catch` by TeammateY (`facing` PlayerX) shortly after.
        *   If no `catch` by the intended receiver, it's an incomplete pass, resulting in a `groundball`.
        *   If a `pass` to TeammateY is present, and no `catch` by TeammateY (or an immediate `groundball` by anyone) follows shortly:
            1.  **Consider Modifying:** Can a subsequent nearby event by TeammateY plausibly be modified to be a `catch`?
            2.  **Consider Adding Catch:** If subsequent actions by TeammateY strongly imply they gained possession (e.g., they immediately pass or shoot), adding a `catch` by TeammateY is a reasonable, high-confidence correction.
            3.  **Consider Adding Groundball:** If possession seems to change to an opponent or become contested immediately after the pass, adding a `groundball` might be more appropriate than forcing a catch.
            4.  **If Ambiguous:** If the outcome of the pass is unclear and no strong evidence supports a specific correction, it may be better to leave it and note the ambiguity.
    *   **Shot & Consequences:**
        *   A `shot` by an offensive player must be `facing` the `Goalie`.
        *   A `shot` is ALWAYS followed by:
            *   A `save_attempt` by the `Goalie` (`facing` the shooter).
            *   If the `save_attempt` is successful (a "Save"): The `Goalie` gains possession (often indicated by a subsequent `groundball` or `pass` by the goalie).
            *   If the `save_attempt` is unsuccessful (a "Goal" is inferred): Play then restarts with a `faceoff`. (See "Goal and Score Consistency (Internal Logic)")
            *   If the `shot` is off-target (missed, no save needed): Results in a `groundball`.
        *   A `save_attempt` event should always be present after a `shot`. If missing:
            1.  **High Priority Correction:** Insert a `save_attempt` by the `Goalie`, `facing` the shooter, with a minimal timestamp delay (e.g., 0.1-0.6s after the shot). This is a critical event.
            2.  Then, evaluate the outcome based on subsequent events:
                *   If Goalie then makes a `pass` or gets a `groundball` very close to the goal quickly: Likely a "Save" where goalie retains/gains possession.
                *   If no immediate goalie possession event occurs, and a `faceoff` is expected based on game flow: This *may* imply a "Goal". Be cautious; only infer a goal if it's a logical outcome and doesn't contradict other strong evidence. Ensure a subsequent `faceoff` is present or can be high-confidence inserted.
                *   If a `groundball` by a field player occurs some distance/time away: Could be a rebound off a save or a missed shot resulting in a loose ball.
    *   **Groundball:**
        *   Indicates a player picking up a loose ball.
        *   Occurs after: faceoffs, incomplete passes, missed shots, uncontrolled saves, or turnovers. Verify `player` and `facing` make sense contextually. Modifications are preferred if only these are slightly off.
    *   **Turnovers:** A player losing possession without a pass/shot (e.g., dropped ball, stripped). This typically results in a `groundball` event for the recovering player (often an opponent). The `facing` field on the `groundball` might indicate the player who lost it. If a player clearly loses possession illogically (e.g., PlayerA passes, then PlayerA immediately shoots without an intervening event from another player), a turnover might have occurred, often leading to a `groundball`. Correct by inserting a `groundball` if high confidence; otherwise, note the anomaly.

*   **Player Actions & Roles:**
    *   The `Goalie` typically does NOT perform `faceoff`, offensive `shot` (unless a very rare, specific situation like an end-of-game desperation play, which is unlikely in typical sequences), or `catch` as a primary receiver in open play. Goalies primarily `save_attempt` and may `pass` to clear or pick up `groundball`s near their crease.
    *   Field players perform `faceoff`, `pass`, `catch`, `shot`, `groundball`.
    *   If a Goalie is listed for `faceoff`, offensive `shot`, or field `catch` as primary receiver, this is a strong candidate for modification (e.g., changing `player` to a likely field player, or `action` if context strongly supports it).

*   **Timestamps & `facing` Field:**
    *   Events must be chronological.
    *   Directly related events (pass -> catch, shot -> save_attempt, faceoff -> faceoff_groundball) have very small time gaps.
    *   `facing` field context:
        *   `pass`: `player`=passer, `facing`=intended receiver.
        *   `catch`: `player`=receiver, `facing`=passer.
        *   `shot`: `player`=shooter, `facing`=`Goalie`.
        *   `save_attempt`: `player`=`Goalie`, `facing`=shooter.
        *   `faceoff`: `player`=faceoff player, `facing`=opposing faceoff player.
        *   `groundball`: `player`=recovering player, `facing`=often player who last had possession or was involved in prior action (e.g., faceoff winner for the initial groundball, shooter for a rebound groundball).
    *   Use `facing` to validate event pairings. Incorrect `facing` is often a good candidate for a high-confidence modification.

*   **Error Types to Consider during Analysis:**
    *   Misclassified Actions: An action that doesn't fit the context (e.g., `catch` without `pass`).
    *   Missing Events: Logical gaps (e.g., `pass` directly to `shot` by same player; `shot` with no `save_attempt`/outcome).
    *   Inserted/Ghost Events: Extraneous events that break logical flow.
    *   Incorrect Player/Facing: Player performing an unlikely action, or `facing` field illogical.
    *   Timing Issues: Actions out of order, or impossible concurrency.
    *   Apply your **Guiding Principles for Correction** when addressing these. Not every identified anomaly warrants a change if the change is speculative.

**Output Requirements:**
*   You MUST generate a JSON object adhering to the structure specified in the user's "REQUIRED JSON OUTPUT" section.
*   Your "analysis" section should clearly reflect the findings from each step of the user's "CHAIN OF THOUGHT ANALYSIS", including justifications for decisions made in STEP 6.
*   The `reconstructed_events` should be your best effort at the true, corrected sequence, applying the principle of minimal change.
*   The `corrections_made` section must accurately reflect the number and nature of changes, with clear descriptions for each.
*   Every decision in "STEP 6 - CORRECTION DECISIONS" and every modification, insertion, or deletion contributing to `reconstructed_events` must be justified by lacrosse logic and the Guiding Principles, aiming to improve the sequence's accuracy and consistency with minimal, high-confidence changes.

Focus on making logical, justifiable, and **conservative** corrections that will demonstrably improve the accuracy and coherence of the game sequence by fixing clear errors, rather than broadly reinterpreting the events."""

    def _create_evaluation_prompt(self, predicted_events: List[Dict], game_setup: Dict, final_score: Dict) -> str:
        """Create the specific prompt for evaluating a game sequence."""
        
        event_count = len(predicted_events)
        
        prompt = f"""
=== LACROSSE CORRECTION TASK ===

GAME DETAILS:
- Teams: {game_setup['team_A']} vs {game_setup['team_B']}
- Duration: {game_setup['duration_seconds']} seconds  
- Current events: {event_count}

DETECTED EVENTS TO ANALYZE:

```json
{json.dumps(predicted_events, indent=2)}

=== CHAIN OF THOUGHT ANALYSIS ===
Please analyze using this step-by-step process, adhering to the Guiding Principles for Correction (Conservatism, Minimal Change, High-Confidence, Fidelity to Input, Justification) from your System Prompt:
STEP 1 - DUPLICATE CHECK:
Look for identical or near-identical events (same player, same action, very close timestamp, e.g., within 0.5s) that are clearly redundant. List any found.
STEP 2 - IMPOSSIBLE PLAYER/ACTION CHECK:
Identify events where the player performing an action is highly improbable or violates role expectations (e.g., Goalie performing faceoff, shot, or a field catch as a primary receiver; field player making a save_attempt). List any found.
STEP 3 - TIMING CONFLICTS & ORDERING:
Look for a single player performing multiple distinct actions at the exact same timestamp that are physically impossible simultaneously. Also, check for gross violations of chronological event order that break fundamental sequences (e.g., catch before pass, save before shot). List any found.
STEP 4 - MISSING CRITICAL EVENTS:
Based on lacrosse logic (e.g., pass usually followed by catch/groundball; shot always by save_attempt/outcome), identify any high-probability missing events that create a clear logical break.
Examples:
Pass by PlayerA to PlayerB, but no subsequent catch by PlayerB or groundball.
Shot by PlayerX, but no subsequent save_attempt by Goalie.
Faceoff without a subsequent groundball to determine possession.
List what appears to be critically missing and why.
STEP 5 - SEQUENCE VALIDATION & FACING FIELD CHECKS:
Review the overall flow. Does it make sense?
Do faceoffs generally occur at the start and after inferred goals?
Does possession flow logically?
Is the facing field consistent for related events (e.g., shooter facing Goalie, Goalie facing shooter on save_attempt; passer facing receiver, receiver facing passer)?
List any significant logical inconsistencies or facing field errors.
STEP 6 - CORRECTION DECISIONS & STRATEGY:
For each potential issue identified in STEPS 1-5, meticulously decide on a course of action. Your primary goal here is to be conservative and apply the Principle of Minimal Change.
a. Re-evaluate the issue: Is it a definite error, or could it be plausible under some interpretation?
b. Correction Options (apply with conservatism):
* Modification (Preferred if possible): Can the event be corrected by changing player, action, facing, or slightly adjusting timestamp (e.g., to resolve a pass/catch timing issue or correct a facing field)? This is often the least disruptive fix. Only modify if you have high confidence in the specific change.
* Insertion: Is a critical event provably missing based on unbreakable lacrosse logic (e.g., a save_attempt immediately after every shot is mandatory; a groundball after a faceoff to establish possession)? Only add events if their absence creates a major logical break AND their details (player, action, approximate time) are strongly implied by surrounding events and core lacrosse rules. Be very cautious about adding events like goals unless evidence from the sequence itself is strong (shot -> save_attempt -> no goalie possession -> faceoff needed).
* Deletion: Is an event clearly spurious (e.g., a true duplicate missed in STEP 1), nonsensical in its context with no possibility of plausible modification, or directly contradictory to an unchangeable, high-confidence event? Deletion should be used sparingly.
c. Apply Guiding Principles:
* Minimal Change: Choose the option that resolves the issue with the least disruption to the original sequence.
* High Confidence: If a fix is speculative or you have low confidence, it is better to leave the original event as-is and note the ambiguity in your analysis. Do not guess.
* Fidelity to Input: Strive to keep as much of the original data as possible.
d. Justify Your Decision: For every correction made (or conscious decision not to correct an identified potential issue), briefly explain why in "step_6_decisions". Base justifications on specific lacrosse logic, the Guiding Principles, and how the change (or lack thereof) improves the sequence's integrity.
e. Internal Goal/Faceoff Logic Check: After proposing all corrections, briefly consider if the sequence of events (shots, save attempts, subsequent possession, and faceoffs) is internally consistent regarding inferred goals, as outlined in your System Prompt's "Goal and Score Consistency (Internal Logic)" section. Note any significant, unresolvable discrepancies, but do not force changes solely to create a 'perfect' goal-faceoff count if it requires speculative event manipulation.
Summarize your specific decisions, the rationale for each, and any uncorrected ambiguities in the "step_6_decisions" field of the JSON output.
=== REQUIRED JSON OUTPUT ===
{{
    "analysis": {{
        "step_1_duplicates": "List any clear duplicates found or 'None found'. Justify removals if any made in STEP 6.",
        "step_2_impossible_players": "List any impossible player/action attributions or 'None found'. Justify changes if any made in STEP 6.", 
        "step_3_timing_conflicts": "List any timing/ordering conflicts or 'None found'. Justify changes if any made in STEP 6.",
        "step_4_missing_events": "List any clearly missing critical events or 'None found'. Justify additions if any made in STEP 6.",
        "step_5_sequence_issues": "List any significant sequence/facing field problems or 'None found'. Justify changes if any made in STEP 6.",
        "step_6_decisions": "Detailed summary of correction decisions: what was changed, what was kept as-is despite potential issue, and why for each, referencing specific lacrosse logic and your guiding principles (conservatism, minimal change, high-confidence). Explain how changes (or lack thereof) relate to internal game progression logic (e.g., goal inference, faceoffs)."
    }},
    "reconstructed_events": [
        // Example: {{"timestamp": 1.0, "player": "Player_1", "action": "faceoff", "facing": "Player_X"}},
        // Example: {{"timestamp": 2.5, "player": "Player_2", "action": "pass", "facing": "Player_Y"}}
    ],
    "corrections_made": {{
        "events_removed": 0,
        "events_added": 0, 
        "events_modified": 0,
        "detailed_changes": [
            // Example: "Modified event at original_timestamp 23.73: Changed action from 'save_attempt' to 'shot' for Player_X because original was illogical for player role.",
            // Example: "Added event: 'catch' at timestamp 10.5 by Player_Y, facing Player_Z, to complete pass from Player_Z at 10.2.",
            // Example: "Removed event at original_timestamp 5.0: Duplicate 'faceoff' action by Player_A."
            // If no changes: "No changes made as all corrections were deemed too speculative or issues were minor."
        ]
    }}
}}


=== SUCCESS CRITERIA ===
Your success will be judged by how well your reconstructed_events:
Adhere strictly to lacrosse logic.
Faithfully represent the likely true game events by making only minimal, necessary, and high-confidence alterations to the input.
Demonstrably fix clear, unambiguous errors from the input without introducing new ones.
The overall goal is a net improvement in the quality, accuracy, and logical coherence of the event list, achieved conservatively.
MAKE CORRECTIONS THAT WILL MEASURABLY IMPROVE ACCURACY THROUGH CAREFUL, JUSTIFIED, AND MINIMAL CHANGES!"""
        return prompt

    async def evaluate_single_game(self, predicted_events: List[Dict], game_setup: Dict, 
                                 final_score: Dict, ground_truth_events: List[Dict]) -> Dict[str, Any]:
        """Evaluate LLM reconstruction for a single game."""
        
        # Always calculate metrics, even without LLM
        basic_metrics = self._calculate_reconstruction_metrics(
            ground_truth_events, 
            predicted_events,  # Use original predictions as baseline
            predicted_events
        )
        
        if self.test_mode:
            # Use simulated ideal corrections
            reconstructed_data = self._simulate_ideal_corrections(predicted_events, ground_truth_events)
            
            # Calculate evaluation metrics
            metrics = self._calculate_reconstruction_metrics(
                ground_truth_events, 
                reconstructed_data.get('reconstructed_events', []),
                predicted_events
            )
            
            return {
                'llm_response': reconstructed_data,
                'metrics': metrics,
                'raw_llm_output': 'Simulated ideal conservative corrections'
            }
        
        elif self.llm is None:
            return {
                'llm_response': {'reconstructed_events': predicted_events, 'corrections_made': 'No LLM available - using original predictions'},
                'metrics': basic_metrics,
                'raw_llm_output': 'LLM not available'
            }
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(predicted_events, game_setup, final_score)
        
        # Create messages using LangChain format
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=eval_prompt)
        ]
        
        try:
            # Use LangChain's invoke method
            response = self.llm.invoke(messages)
            llm_output = response.content
            
            # Parse LLM response
            reconstructed_data = self._parse_llm_response(llm_output)
            
            # Calculate evaluation metrics
            metrics = self._calculate_reconstruction_metrics(
                ground_truth_events, 
                reconstructed_data.get('reconstructed_events', []),
                predicted_events
            )
            
            return {
                'llm_response': reconstructed_data,
                'metrics': metrics,
                'raw_llm_output': llm_output
            }
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                'llm_response': {'reconstructed_events': [], 'corrections_made': f'Error occurred: {str(e)}'},
                'metrics': basic_metrics,
                'error': str(e)
            }

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response."""
        try:
            # Find JSON block in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {'reconstructed_events': [], 'corrections_made': 'Could not parse response'}
                
        except json.JSONDecodeError:
            return {'reconstructed_events': [], 'corrections_made': 'Invalid JSON in response'}

    def _calculate_reconstruction_metrics(self, ground_truth: List[Dict], 
                                        reconstructed: List[Dict], 
                                        original_predictions: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive metrics comparing reconstruction to ground truth and original predictions."""
        
        metrics = {}
        
        # Basic count metrics
        metrics['gt_event_count'] = len(ground_truth)
        metrics['reconstructed_event_count'] = len(reconstructed)
        metrics['original_prediction_count'] = len(original_predictions)
        
        # === LLM INPUT vs OUTPUT COMPARISON ===
        input_output_metrics = self._calculate_input_output_metrics(original_predictions, reconstructed)
        for key, value in input_output_metrics.items():
            metrics[f'input_output_{key}'] = value
        
        # === LLM OUTPUT vs GROUND TRUTH COMPARISON ===
        output_gt_metrics = self._calculate_output_groundtruth_metrics(ground_truth, reconstructed)
        for key, value in output_gt_metrics.items():
            metrics[f'output_gt_{key}'] = value
        
        # === ORIGINAL INPUT vs GROUND TRUTH COMPARISON ===
        input_gt_metrics = self._calculate_output_groundtruth_metrics(ground_truth, original_predictions)
        for key, value in input_gt_metrics.items():
            metrics[f'input_gt_{key}'] = value
        
        # === IMPROVEMENT METRICS ===
        metrics['f1_improvement'] = metrics['output_gt_f1'] - metrics['input_gt_f1']
        metrics['precision_improvement'] = metrics['output_gt_precision'] - metrics['input_gt_precision']
        metrics['recall_improvement'] = metrics['output_gt_recall'] - metrics['input_gt_recall']
        metrics['accuracy_improvement'] = metrics['output_gt_accuracy'] - metrics['input_gt_accuracy']
        
        # === SEQUENCE-LEVEL METRICS ===
        gt_actions = [e['action'] for e in ground_truth]
        recon_actions = [e['action'] for e in reconstructed]
        orig_actions = [e['action'] for e in original_predictions]
        
        # Action sequence similarity
        metrics['output_gt_sequence_similarity'] = SequenceMatcher(None, gt_actions, recon_actions).ratio()
        metrics['input_gt_sequence_similarity'] = SequenceMatcher(None, gt_actions, orig_actions).ratio()
        metrics['input_output_sequence_similarity'] = SequenceMatcher(None, orig_actions, recon_actions).ratio()
        
        # Temporal alignment metrics
        metrics['output_gt_temporal_accuracy'] = self._calculate_temporal_accuracy(ground_truth, reconstructed)
        metrics['input_gt_temporal_accuracy'] = self._calculate_temporal_accuracy(ground_truth, original_predictions)
        
        # Event-level matching within time windows
        metrics['output_gt_event_match_rate'] = self._match_events_by_time(ground_truth, reconstructed) / max(len(ground_truth), 1)
        metrics['input_gt_event_match_rate'] = self._match_events_by_time(ground_truth, original_predictions) / max(len(ground_truth), 1)
        
        return metrics

    def _calculate_input_output_metrics(self, input_events: List[Dict], output_events: List[Dict]) -> Dict[str, float]:
        """Calculate metrics comparing LLM input to LLM output."""
        if not input_events and not output_events:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}
        if not input_events or not output_events:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Create event signatures for matching
        def event_signature(event):
            time_bin = int(event['timestamp'] / 0.5)  # 0.5s bins
            return (time_bin, event['action'], event['player'])
        
        input_sigs = set(event_signature(e) for e in input_events)
        output_sigs = set(event_signature(e) for e in output_events)
        
        # Calculate metrics
        tp = len(input_sigs & output_sigs)  # Events kept by LLM
        fp = len(output_sigs - input_sigs)  # Events added by LLM
        fn = len(input_sigs - output_sigs)  # Events removed by LLM
        
        precision = tp / max(tp + fp, 1)  # How many output events were from input
        recall = tp / max(tp + fn, 1)     # How many input events were kept
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy = tp / max(len(input_sigs | output_sigs), 1)  # Overall agreement
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'events_added': fp,
            'events_removed': fn,
            'events_kept': tp
        }

    def _calculate_output_groundtruth_metrics(self, ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, float]:
        """Calculate metrics comparing predictions to ground truth."""
        if not ground_truth and not predictions:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}
        if not ground_truth or not predictions:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Create event signatures for matching
        def event_signature(event):
            time_bin = int(event['timestamp'] / 0.5)  # 0.5s bins
            return (time_bin, event['action'], event['player'])
        
        gt_sigs = set(event_signature(e) for e in ground_truth)
        pred_sigs = set(event_signature(e) for e in predictions)
        
        # Calculate metrics
        tp = len(gt_sigs & pred_sigs)     # Correctly predicted events
        fp = len(pred_sigs - gt_sigs)     # False positive events
        fn = len(gt_sigs - pred_sigs)     # Missed events
        
        precision = tp / max(tp + fp, 1)  # Accuracy of predictions
        recall = tp / max(tp + fn, 1)     # Coverage of ground truth
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy = tp / max(len(gt_sigs | pred_sigs), 1)  # Overall correctness
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def _match_events_by_time(self, gt_events: List[Dict], recon_events: List[Dict], 
                            time_window: float = 2.0) -> int:
        """Count events that match within a time window."""
        matched = 0
        
        for gt_event in gt_events:
            for recon_event in recon_events:
                time_diff = abs(gt_event['timestamp'] - recon_event['timestamp'])
                if (time_diff <= time_window and 
                    gt_event['action'] == recon_event['action'] and
                    gt_event['player'] == recon_event['player']):
                    matched += 1
                    break
        
        return matched

    def _calculate_temporal_accuracy(self, gt_events: List[Dict], recon_events: List[Dict]) -> float:
        """Calculate how well the timing of events matches."""
        if not gt_events or not recon_events:
            return 0.0
        
        # Compare relative timing of matching action types
        total_comparisons = 0
        correct_orderings = 0
        
        for i, gt1 in enumerate(gt_events[:-1]):
            for j, gt2 in enumerate(gt_events[i+1:], i+1):
                # Find corresponding events in reconstruction
                recon1_idx = self._find_matching_event(gt1, recon_events)
                recon2_idx = self._find_matching_event(gt2, recon_events)
                
                if recon1_idx is not None and recon2_idx is not None:
                    gt_order = gt1['timestamp'] < gt2['timestamp']
                    recon_order = recon_events[recon1_idx]['timestamp'] < recon_events[recon2_idx]['timestamp']
                    
                    if gt_order == recon_order:
                        correct_orderings += 1
                    total_comparisons += 1
        
        return correct_orderings / max(total_comparisons, 1)

    def _find_matching_event(self, target_event: Dict, event_list: List[Dict]) -> int:
        """Find the best matching event in a list."""
        for i, event in enumerate(event_list):
            if (event['action'] == target_event['action'] and 
                event['player'] == target_event['player']):
                return i
        return None

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict for error cases."""
        return {
            'gt_event_count': 0,
            'reconstructed_event_count': 0,
            'original_prediction_count': 0,
            'action_sequence_similarity': 0.0,
            'event_match_rate': 0.0,
            'action_accuracy': 0.0,
            'improvement_over_original': 0.0,
            'temporal_accuracy': 0.0
        }

    def calculate_sequence_similarity(self, events1: List[Dict], events2: List[Dict]) -> float:
        """Calculate similarity between two event sequences."""
        if not events1 and not events2:
            return 1.0
        if not events1 or not events2:
            return 0.0
            
        # Convert to comparable signatures
        def to_signature(events):
            return [(e['action'], e['player'], round(e['timestamp'], 1)) for e in events]
        
        sig1 = to_signature(events1)
        sig2 = to_signature(events2)
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, sig1, sig2).ratio()
        
        # Also consider length similarity
        length_similarity = 1.0 - abs(len(events1) - len(events2)) / max(len(events1), len(events2), 1)
        
        # Weighted combination
        return 0.7 * similarity + 0.3 * length_similarity
    
    def find_consensus_groups(self, responses: List[Dict[str, Any]]) -> List[List[int]]:
        """Group responses by similarity and find consensus groups."""
        if not responses:
            return []
            
        # Extract reconstructed events from each response
        event_sequences = []
        for response in responses:
            events = response.get('llm_response', {}).get('reconstructed_events', [])
            event_sequences.append(events)
        
        # Create similarity matrix
        n = len(event_sequences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                similarity = self.calculate_sequence_similarity(event_sequences[i], event_sequences[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Group sequences by similarity threshold
        groups = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
                
            # Start new group
            group = [i]
            assigned.add(i)
            
            # Find similar sequences
            for j in range(i + 1, n):
                if j not in assigned and similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        # Sort groups by size (largest first)
        groups.sort(key=len, reverse=True)
        
        return groups
    
    def average_responses_from_group(self, responses: List[Dict[str, Any]], group_indices: List[int]) -> Dict[str, Any]:
        """Average responses from a consensus group."""
        if len(group_indices) == 1:
            return responses[group_indices[0]]
        
        group_responses = [responses[idx] for idx in group_indices]
        
        # Average the reconstructed events using consensus
        averaged_events = self._average_event_sequences(group_responses)
        
        # Average the metrics
        averaged_metrics = self._average_metrics(group_responses)
        
        # Aggregate corrections made
        aggregated_corrections = self._aggregate_corrections(group_responses)
        
        # Combine analysis text
        combined_analysis = self._combine_analysis(group_responses)
        
        return {
            'llm_response': {
                'reconstructed_events': averaged_events,
                'corrections_made': aggregated_corrections,
                'analysis': combined_analysis
            },
            'metrics': averaged_metrics,
            'raw_llm_output': f'Averaged from {len(group_responses)} consensus responses'
        }
    
    def _average_event_sequences(self, responses: List[Dict[str, Any]]) -> List[Dict]:
        """Average event sequences from multiple responses using consensus."""
        all_event_sequences = []
        for response in responses:
            events = response.get('llm_response', {}).get('reconstructed_events', [])
            all_event_sequences.append(events)
        
        if not all_event_sequences:
            return []
        
        # Find consensus events - events that appear in multiple responses
        event_candidates = []
        
        # Collect all unique events with their frequencies
        event_frequency = defaultdict(list)  # signature -> list of (response_idx, event)
        
        for resp_idx, events in enumerate(all_event_sequences):
            for event in events:
                # Create a signature for matching similar events
                signature = (
                    event['action'],
                    event['player'],
                    round(event['timestamp'], 1)  # Round to 0.1s for matching
                )
                event_frequency[signature].append((resp_idx, event))
        
        # Build consensus events
        consensus_events = []
        
        for signature, event_instances in event_frequency.items():
            # Only include events that appear in multiple responses OR in majority of responses
            min_consensus = max(1, len(responses) // 2)  # At least half the responses
            
            if len(event_instances) >= min_consensus:
                # Average the timestamps and other details
                timestamps = [inst[1]['timestamp'] for inst in event_instances]
                avg_timestamp = sum(timestamps) / len(timestamps)
                
                # Use most common values for discrete fields
                actions = [inst[1]['action'] for inst in event_instances]
                players = [inst[1]['player'] for inst in event_instances]
                facings = [inst[1]['facing'] for inst in event_instances]
                
                consensus_event = {
                    'timestamp': avg_timestamp,
                    'action': Counter(actions).most_common(1)[0][0],
                    'player': Counter(players).most_common(1)[0][0],
                    'facing': Counter(facings).most_common(1)[0][0]
                }
                
                consensus_events.append(consensus_event)
        
        # Sort by timestamp
        consensus_events.sort(key=lambda x: x['timestamp'])
        
        return consensus_events
    
    def _average_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average metrics from multiple responses."""
        if not responses:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for response in responses:
            metrics = response.get('metrics', {})
            all_keys.update(metrics.keys())
        
        averaged_metrics = {}
        
        for key in all_keys:
            values = []
            for response in responses:
                metrics = response.get('metrics', {})
                if key in metrics:
                    values.append(metrics[key])
            
            if values:
                averaged_metrics[key] = sum(values) / len(values)
            else:
                averaged_metrics[key] = 0.0
        
        return averaged_metrics
    
    def _aggregate_corrections(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate corrections made from multiple responses."""
        total_removed = 0
        total_added = 0
        total_modified = 0
        all_detailed_changes = []
        
        for response in responses:
            corrections = response.get('llm_response', {}).get('corrections_made', {})
            
            if isinstance(corrections, dict):
                total_removed += corrections.get('events_removed', 0)
                total_added += corrections.get('events_added', 0)
                total_modified += corrections.get('events_modified', 0)
                
                detailed = corrections.get('detailed_changes', [])
                if isinstance(detailed, list):
                    all_detailed_changes.extend(detailed)
        
        # Average the counts
        num_responses = len(responses)
        return {
            'events_removed': total_removed / num_responses,
            'events_added': total_added / num_responses,
            'events_modified': total_modified / num_responses,
            'detailed_changes': [f"Averaged from {num_responses} consensus responses"] + all_detailed_changes[:5]  # Include a few examples
        }
    
    def _combine_analysis(self, responses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Combine analysis from multiple responses."""
        combined = {
            'step_1_duplicates': f"Consensus from {len(responses)} responses",
            'step_2_impossible_players': f"Consensus from {len(responses)} responses", 
            'step_3_timing_conflicts': f"Consensus from {len(responses)} responses",
            'step_4_missing_events': f"Consensus from {len(responses)} responses",
            'step_5_sequence_issues': f"Consensus from {len(responses)} responses",
            'step_6_decisions': f"Averaged decision from {len(responses)} similar responses with {self.similarity_threshold:.1%} similarity threshold"
        }
        
        return combined
    
    async def evaluate_single_game_with_consensus(self, predicted_events: List[Dict], 
                                                game_setup: Dict, final_score: Dict, 
                                                ground_truth_events: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single game using consensus from multiple LLM runs with averaging."""
        
        if self.consensus_runs <= 1:
            # No consensus needed, use single evaluation
            return await self.evaluate_single_game(predicted_events, game_setup, final_score, ground_truth_events)
        
        print(f"  🔄 Running {self.consensus_runs} iterations for consensus averaging...")
        
        # Run multiple evaluations
        responses = []
        for i in range(self.consensus_runs):
            try:
                response = await self.evaluate_single_game(
                    predicted_events, game_setup, final_score, ground_truth_events
                )
                responses.append(response)
                
                # Progress indicator
                if (i + 1) % 3 == 0 or (i + 1) == self.consensus_runs:
                    print(f"    ⏳ Completed {i + 1}/{self.consensus_runs} iterations")
                    
            except Exception as e:
                print(f"    ⚠️  Iteration {i + 1} failed: {e}")
                continue
        
        if not responses:
            print("    ❌ All iterations failed")
            return await self.evaluate_single_game(predicted_events, game_setup, final_score, ground_truth_events)
        
        # Find consensus groups
        print(f"  🔍 Analyzing consensus from {len(responses)} successful runs...")
        groups = self.find_consensus_groups(responses)
        
        # Report consensus analysis
        print(f"    📊 Found {len(groups)} distinct response groups:")
        for i, group in enumerate(groups):
            size = len(group)
            percentage = (size / len(responses)) * 100
            print(f"      Group {i+1}: {size} responses ({percentage:.1f}%)")
        
        # Average responses from largest consensus group
        if groups:
            largest_group = groups[0]
            consensus_response = self.average_responses_from_group(responses, largest_group)
            
            # Add consensus metadata
            consensus_response['consensus_metadata'] = {
                'total_runs': len(responses),
                'consensus_size': len(largest_group),
                'consensus_percentage': (len(largest_group) / len(responses)) * 100,
                'num_distinct_groups': len(groups),
                'group_sizes': [len(group) for group in groups],
                'averaging_method': 'consensus_averaging'
            }
            
            consensus_size = len(largest_group)
            consensus_pct = (consensus_size / len(responses)) * 100
            print(f"  ✅ Averaged consensus from {consensus_size}/{len(responses)} runs ({consensus_pct:.1f}%)")
            
            return consensus_response
        else:
            # Fallback to first response if no groups found
            print("  ⚠️  No consensus groups found, using first response")
            return responses[0]
    
    async def evaluate_all_f1_levels_with_consensus(self, ground_truth_file: str, predictions_dir: str = ".") -> Dict[str, Any]:
        """Evaluate LLM performance across all F1 levels using consensus averaging approach."""
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
        
        results = {}
        
        # Progress tracking
        running_improvements = []
        running_conservatism = []
        running_consensus = []
        
        # Process each F1 level
        f1_levels = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        
        print(f"🎯 Evaluating {len(f1_levels)} F1 levels with {self.consensus_runs}-run consensus averaging...\n")
        
        for i, f1_level in enumerate(f1_levels, 1):
            predictions_file = f"{predictions_dir}/2v2_predictions_f1_{f1_level:02d}.json"
            
            try:
                with open(predictions_file, 'r') as f:
                    predictions_data = json.load(f)
                
                print(f"📈 Evaluating F1 level {f1_level}% with consensus averaging...")
                
                # Use consensus evaluation with averaging
                evaluation = await self.evaluate_single_game_with_consensus(
                    predictions_data['predicted_events'],
                    predictions_data['game_setup'],
                    predictions_data['final_score'],
                    ground_truth_data['events']
                )
                
                results[f"f1_{f1_level}"] = {
                    'target_f1': f1_level / 100.0,
                    'achieved_f1': predictions_data['metadata'].get('achieved_f1_score', f1_level / 100.0),
                    'evaluation': evaluation
                }
                
                # Extract key metrics
                metrics = evaluation['metrics']
                consensus_meta = evaluation.get('consensus_metadata', {})
                
                ml_f1 = metrics.get('input_gt_f1', 0)
                llm_f1 = metrics.get('output_gt_f1', 0)
                f1_improvement = metrics.get('f1_improvement', 0)
                conservatism = metrics.get('input_output_f1', 0)
                consensus_pct = consensus_meta.get('consensus_percentage', 0)
                
                # Performance indicators
                performance_icon = "🟢" if f1_improvement > 0.01 else "🟡" if f1_improvement > -0.01 else "🔴"
                conservatism_icon = "🟢" if conservatism > 0.95 else "🟡" if conservatism > 0.85 else "🔴"
                consensus_icon = "🟢" if consensus_pct > 70 else "🟡" if consensus_pct > 50 else "🔴"
                
                # Corrections info (now averaged)
                corrections = evaluation['llm_response'].get('corrections_made', {})
                if isinstance(corrections, dict):
                    removed = corrections.get('events_removed', 0)
                    added = corrections.get('events_added', 0)
                    modified = corrections.get('events_modified', 0)
                    changes_summary = f"≈{removed+added+modified:.1f}"  # Approximate since averaged
                else:
                    changes_summary = "unknown"
                
                print(f"  ✅ Complete! ML: {ml_f1:.3f} → LLM: {llm_f1:.3f} | Improvement: {f1_improvement:+.3f} {performance_icon}")
                print(f"      Conservatism: {conservatism:.1%} {conservatism_icon} | Consensus: {consensus_pct:.1f}% {consensus_icon} | Avg Changes: {changes_summary}")
                
                # Update running metrics
                running_improvements.append(f1_improvement)
                running_conservatism.append(conservatism)
                running_consensus.append(consensus_pct)
                
                # Show running progress
                if i % 3 == 0 or i == len(f1_levels):
                    avg_improvement = sum(running_improvements) / len(running_improvements)
                    avg_conservatism = sum(running_conservatism) / len(running_conservatism)
                    avg_consensus = sum(running_consensus) / len(running_consensus)
                    trend_icon = "📈" if avg_improvement > 0 else "📉" if avg_improvement < 0 else "➡️"
                    
                    print(f"    📊 Progress ({i}/{len(f1_levels)}): Avg Improvement: {avg_improvement:+.3f} {trend_icon}")
                    print(f"        Avg Conservatism: {avg_conservatism:.1%} | Avg Consensus: {avg_consensus:.1f}%")
                    if i < len(f1_levels):
                        print()  # Add space before next batch
                
            except FileNotFoundError:
                print(f"  ❌ Predictions file not found for F1 {f1_level}%")
                continue
            except Exception as e:
                print(f"  ❌ Error processing F1 {f1_level}%: {e}")
                continue
        
        return results

    def evaluate_all_f1_levels(self, ground_truth_file: str, predictions_dir: str = ".") -> Dict[str, Any]:
        """Evaluate LLM performance across all F1 levels - synchronous version (original method)."""
        
        if self.consensus_runs > 1:
            # Use consensus if configured
            return asyncio.run(self.evaluate_all_f1_levels_with_consensus(ground_truth_file, predictions_dir))
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
        
        results = {}
        
        # Progress tracking
        running_improvements = []
        running_conservatism = []
        
        # Process each F1 level
        f1_levels = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        
        print(f"🎯 Evaluating {len(f1_levels)} F1 levels with real-time feedback...\n")
        
        for i, f1_level in enumerate(f1_levels, 1):
            predictions_file = f"{predictions_dir}/2v2_predictions_f1_{f1_level:02d}.json"
            
            try:
                with open(predictions_file, 'r') as f:
                    predictions_data = json.load(f)
                
                print(f"Evaluating F1 level {f1_level}%...")
                
                # Use sync version
                evaluation = asyncio.run(self.evaluate_single_game(
                    predictions_data['predicted_events'],
                    predictions_data['game_setup'],
                    predictions_data['final_score'],
                    ground_truth_data['events']
                ))
                
                results[f"f1_{f1_level}"] = {
                    'target_f1': f1_level / 100.0,
                    'achieved_f1': predictions_data['metadata'].get('achieved_f1_score', f1_level / 100.0),
                    'evaluation': evaluation
                }
                
                # Print immediate performance feedback
                metrics = evaluation['metrics']
                
                # Extract key metrics
                ml_f1 = metrics.get('input_gt_f1', 0)
                llm_f1 = metrics.get('output_gt_f1', 0)
                f1_improvement = metrics.get('f1_improvement', 0)
                conservatism = metrics.get('input_output_f1', 0)
                
                # Performance indicators
                performance_icon = "🟢" if f1_improvement > 0.01 else "🟡" if f1_improvement > -0.01 else "🔴"
                conservatism_icon = "🟢" if conservatism > 0.95 else "🟡" if conservatism > 0.85 else "🔴"
                
                # Corrections info
                corrections = evaluation['llm_response'].get('corrections_made', {})
                if isinstance(corrections, dict):
                    removed = corrections.get('events_removed', 0)
                    added = corrections.get('events_added', 0)
                    modified = corrections.get('events_modified', 0)
                    changes_summary = f"±{removed+added+modified}"
                else:
                    changes_summary = "unknown"
                
                print(f"  ✅ Complete! ML: {ml_f1:.3f} → LLM: {llm_f1:.3f} | Improvement: {f1_improvement:+.3f} {performance_icon} | Conservatism: {conservatism:.1%} {conservatism_icon} | Changes: {changes_summary}")
                
                # Update running metrics
                running_improvements.append(f1_improvement)
                running_conservatism.append(conservatism)
                
                # Show running progress every few evaluations
                if i % 3 == 0 or i == len(f1_levels):
                    avg_improvement = sum(running_improvements) / len(running_improvements)
                    avg_conservatism = sum(running_conservatism) / len(running_conservatism)
                    trend_icon = "📈" if avg_improvement > 0 else "📉" if avg_improvement < 0 else "➡️"
                    
                    print(f"    📊 Progress ({i}/{len(f1_levels)}): Avg Improvement: {avg_improvement:+.3f} {trend_icon} | Avg Conservatism: {avg_conservatism:.1%}")
                    if i < len(f1_levels):
                        print()  # Add space before next batch
                
            except FileNotFoundError:
                print(f"  ❌ Predictions file not found for F1 {f1_level}%")
                continue
            except Exception as e:
                print(f"  ❌ Error processing F1 {f1_level}%: {e}")
                continue
        
        return results

    def _simulate_ideal_corrections(self, predicted_events: List[Dict], ground_truth_events: List[Dict]) -> Dict[str, Any]:
        """Simulate what an ideal conservative LLM would do."""
        random.seed(42)  # Consistent results
        
        corrected_events = predicted_events.copy()
        corrections = {'events_removed': 0, 'events_added': 0, 'reasoning': 'Simulated ideal conservative corrections'}
        
        # Remove some obvious duplicates (simulate removing ~10% of bad events)
        if len(corrected_events) > len(ground_truth_events):
            excess = len(corrected_events) - len(ground_truth_events)
            excess_to_remove = min(excess, max(1, len(corrected_events) // 10))
            
            # Remove events that are furthest from ground truth events
            events_to_remove = []
            for i, pred_event in enumerate(corrected_events):
                # Find closest ground truth event
                closest_distance = float('inf')
                for gt_event in ground_truth_events:
                    distance = abs(pred_event['timestamp'] - gt_event['timestamp'])
                    if pred_event['action'] == gt_event['action'] and pred_event['player'] == gt_event['player']:
                        distance *= 0.1  # Much closer if exact match
                    closest_distance = min(closest_distance, distance)
                
                events_to_remove.append((i, closest_distance))
            
            # Remove events with largest distances (worst matches)
            events_to_remove.sort(key=lambda x: x[1], reverse=True)
            indices_to_remove = [x[0] for x in events_to_remove[:excess_to_remove]]
            
            corrected_events = [e for i, e in enumerate(corrected_events) if i not in indices_to_remove]
            corrections['events_removed'] = excess_to_remove
        
        # Occasionally add a missing critical event (simulate adding ~5% good events)
        elif len(corrected_events) < len(ground_truth_events) and random.random() < 0.3:
            # Find a ground truth event that might be missing
            for gt_event in ground_truth_events:
                found_match = False
                for pred_event in corrected_events:
                    if (abs(gt_event['timestamp'] - pred_event['timestamp']) < 1.0 and
                        gt_event['action'] == pred_event['action'] and
                        gt_event['player'] == pred_event['player']):
                        found_match = True
                        break
                
                if not found_match and len(corrected_events) < len(predicted_events) + 2:
                    # Add this missing event
                    corrected_events.append(gt_event.copy())
                    corrections['events_added'] += 1
                    break
        
        # Sort by timestamp
        corrected_events.sort(key=lambda x: x['timestamp'])
        
        return {
            'reconstructed_events': corrected_events,
            'corrections_made': corrections
        }

# Example usage
def run_full_evaluation(consensus_runs: int = 1, similarity_threshold: float = 0.8):
    """Run the complete evaluation pipeline with optional consensus."""
    
    # First generate error levels
    print("Generating classifier predictions with different F1 scores...")
    try:
        from classifier_error_simulator import generate_error_levels
        generate_error_levels("2v2_groundtruth.json")
    except ImportError:
        print("Error: classifier_error_simulator not found. Please run that first.")
        return
    
    # Then evaluate LLM performance
    if consensus_runs > 1:
        print(f"\nEvaluating LLM reconstruction performance with {consensus_runs}-run consensus...")
        evaluator = LacrosseLLMEvaluator(consensus_runs=consensus_runs, similarity_threshold=similarity_threshold)
        results = evaluator.evaluate_all_f1_levels("2v2_groundtruth.json")
        
        # Save results with consensus suffix
        output_file = "llm_evaluation_results_consensus.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n💾 Consensus results saved to: {output_file}")
        
    else:
        print("\nEvaluating LLM reconstruction performance...")
        evaluator = LacrosseLLMEvaluator()
        results = evaluator.evaluate_all_f1_levels("2v2_groundtruth.json")
        
        # Save results
        with open("llm_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=4)

    # Print summary
    print("\nCOMPREHENSIVE EVALUATION SUMMARY:")
    print("=" * 140)
    
    if consensus_runs > 1:
        # Enhanced summary for consensus results
        print(f"{'F1 Level':>8} | {'Input F1':>8} | {'ML→GT Quality':>16} | {'LLM→GT Quality':>17} | {'Conservatism':>12} | {'Consensus':>12} | {'Improvements':>15}")
        print(f"{'':>8} | {'Target':>8} | {'P':>4} {'R':>4} {'F1':>4} {'A':>4} | {'P':>4} {'R':>4} {'F1':>4} {'A':>4} | {'F1':>4} {'Chg':>4} {'%':>3} | {'%':>4} {'Grps':>4} {'Sz':>3} | {'ΔF1':>4} {'ΔP':>4} {'ΔR':>4} {'ΔA':>4}")
        print("-" * 140)
        
        consensus_scores = []
        
        for f1_key, result in results.items():
            target_f1 = result['target_f1']
            evaluation = result['evaluation']
            metrics = evaluation['metrics']
            consensus_meta = evaluation.get('consensus_metadata', {})
            
            # ML Baseline Quality (Input→GT)
            ml_p = metrics.get('input_gt_precision', 0)
            ml_r = metrics.get('input_gt_recall', 0)
            ml_f1 = metrics.get('input_gt_f1', 0)
            ml_a = metrics.get('input_gt_accuracy', 0)
            
            # LLM Final Quality (Output→GT)
            llm_p = metrics.get('output_gt_precision', 0)
            llm_r = metrics.get('output_gt_recall', 0)
            llm_f1 = metrics.get('output_gt_f1', 0)
            llm_a = metrics.get('output_gt_accuracy', 0)
            
            # Conservatism (Input→Output)
            cons_f1 = metrics.get('input_output_f1', 0)
            
            # Calculate change magnitude
            corrections = evaluation['llm_response'].get('corrections_made', {})
            if isinstance(corrections, dict):
                total_changes = corrections.get('events_removed', 0) + corrections.get('events_added', 0) + corrections.get('events_modified', 0)
            else:
                total_changes = 0
            
            # Consensus info
            consensus_pct = consensus_meta.get('consensus_percentage', 0)
            num_groups = consensus_meta.get('num_distinct_groups', 0)
            consensus_size = consensus_meta.get('consensus_size', 0)
            
            consensus_scores.append(consensus_pct)
            
            # Improvements
            f1_imp = metrics.get('f1_improvement', 0)
            p_imp = metrics.get('precision_improvement', 0)
            r_imp = metrics.get('recall_improvement', 0)
            a_imp = metrics.get('accuracy_improvement', 0)
            
            # Format and print
            f1_level_str = f"{int(target_f1 * 100)}%"
            
            print(f"{f1_level_str:>8} | {target_f1:>8.2f} | "
                  f"{ml_p:>4.2f} {ml_r:>4.2f} {ml_f1:>4.2f} {ml_a:>4.2f} | "
                  f"{llm_p:>4.2f} {llm_r:>4.2f} {llm_f1:>4.2f} {llm_a:>4.2f} | "
                  f"{cons_f1:>4.2f} {total_changes:>4.1f} {cons_f1*100:>3.0f} | "
                  f"{consensus_pct:>4.0f} {num_groups:>4.0f} {consensus_size:>3.0f} | "
                  f"{f1_imp:>+4.2f} {p_imp:>+4.2f} {r_imp:>+4.2f} {a_imp:>+4.2f}")
        
        print("-" * 140)
        
        # Overall consensus statistics
        avg_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0
        
        print(f"\n📈 CONSENSUS STATISTICS:")
        print(f"  • Average consensus: {avg_consensus:.1f}%")
        print(f"  • High consensus (>70%): {sum(1 for c in consensus_scores if c > 70)} / {len(consensus_scores)} levels")
        print(f"  • Low consensus (<50%): {sum(1 for c in consensus_scores if c < 50)} / {len(consensus_scores)} levels")
        print(f"  • Consensus runs per level: {consensus_runs}")
        print(f"  • Similarity threshold: {similarity_threshold:.1%}")
        
    else:
        # Original summary
        print(f"{'F1 Level':>8} | {'Input F1':>8} | {'ML Baseline Quality':>20} | {'LLM Final Quality':>18} | {'LLM Conservatism':>16} | {'LLM Improvements':>15}")
        print(f"{'':>8} | {'':>8} | {'(ML Predictions → GT)':>20} | {'(LLM Output → GT)':>18} | {'(Changes Made)':>16} | {'(LLM vs ML)':>15}")
        print(f"{'':>8} | {'':>8} | {'P':>4} {'R':>4} {'F1':>4} {'A':>4} | {'P':>4} {'R':>4} {'F1':>4} {'A':>4} | {'P':>4} {'R':>4} {'F1':>4} {'A':>4} | {'ΔF1':>4} {'ΔP':>4} {'ΔR':>4} {'ΔA':>4}")
        print("-" * 140)
        
        for f1_key, result in results.items():
            target_f1 = result['target_f1']
            achieved_f1 = result['achieved_f1']
            metrics = result['evaluation']['metrics']
            
            # ML Baseline Quality (Input→GT)
            ml_p = metrics.get('input_gt_precision', 0)
            ml_r = metrics.get('input_gt_recall', 0)
            ml_f1 = metrics.get('input_gt_f1', 0)
            ml_a = metrics.get('input_gt_accuracy', 0)
            
            # LLM Final Quality (Output→GT)
            llm_p = metrics.get('output_gt_precision', 0)
            llm_r = metrics.get('output_gt_recall', 0)
            llm_f1 = metrics.get('output_gt_f1', 0)
            llm_a = metrics.get('output_gt_accuracy', 0)
            
            # LLM Conservatism (Input→Output) - How much LLM changed the input
            # 100% = no changes, 0% = completely different
            change_p = metrics.get('input_output_precision', 0)
            change_r = metrics.get('input_output_recall', 0)
            change_f1 = metrics.get('input_output_f1', 0)
            change_a = metrics.get('input_output_accuracy', 0)
            
            # Improvements (LLM vs ML)
            imp_f1 = metrics.get('f1_improvement', 0)
            imp_p = metrics.get('precision_improvement', 0)
            imp_r = metrics.get('recall_improvement', 0)
            imp_a = metrics.get('accuracy_improvement', 0)
            
            print(f"{target_f1*100:7.0f}% | {achieved_f1:7.3f} | "
                  f"{ml_p*100:3.0f} {ml_r*100:3.0f} {ml_f1*100:3.0f} {ml_a*100:3.0f} | "
                  f"{llm_p*100:3.0f} {llm_r*100:3.0f} {llm_f1*100:3.0f} {llm_a*100:3.0f} | "
                  f"{change_p*100:3.0f} {change_r*100:3.0f} {change_f1*100:3.0f} {change_a*100:3.0f} | "
                  f"{imp_f1*100:+3.0f} {imp_p*100:+3.0f} {imp_r*100:+3.0f} {imp_a*100:+3.0f}")
    
    print("\nCLEAR EXPLANATION OF METRICS:")
    print("• ML Baseline Quality: How good the original ML predictions were compared to ground truth")
    print("• LLM Final Quality: How good the LLM's corrected version is compared to ground truth") 
    print("• LLM Conservatism: How much the LLM changed the input (100% = no changes, 0% = totally different)")
    print("• LLM Improvements: Direct improvement from LLM corrections (positive = better, negative = worse)")
    print("• All metrics: P=Precision(%), R=Recall(%), F1=F1-Score(%), A=Accuracy(%)")
    print("• 🎯 GOAL: LLM Final Quality > ML Baseline Quality (positive improvements)")
    print("• 🎯 IDEAL: High conservatism (95-99%) with small positive improvements (+1 to +5%)")
    
    if consensus_runs > 1:
        print("• 📊 CONSENSUS: Higher % = more agreement between LLM runs (>70% is very good)")
    
    # Additional insights
    avg_f1_improvement = sum(r['evaluation']['metrics'].get('f1_improvement', 0) for r in results.values()) / len(results)
    avg_precision_improvement = sum(r['evaluation']['metrics'].get('precision_improvement', 0) for r in results.values()) / len(results)
    avg_conservatism = sum(r['evaluation']['metrics'].get('input_output_f1', 0) for r in results.values()) / len(results)
    
    print(f"\nOVERALL PERFORMANCE SUMMARY:")
    print(f"• Average F1 improvement: {avg_f1_improvement:+.3f} {'🟢' if avg_f1_improvement > 0 else '🔴' if avg_f1_improvement < 0 else '🟡'}")
    print(f"• Average Precision improvement: {avg_precision_improvement:+.3f}")
    print(f"• Average LLM conservatism: {avg_conservatism:.1%} {'🟢' if avg_conservatism > 0.9 else '🟡' if avg_conservatism > 0.8 else '🔴'}")
    best_f1 = max(results.keys(), key=lambda k: results[k]['evaluation']['metrics'].get('f1_improvement', 0)).replace('f1_', '')
    print(f"• Best performing F1 level: {best_f1}%")

def run_consensus_evaluation(consensus_runs: int = 10, similarity_threshold: float = 0.8):
    """Run evaluation specifically with consensus averaging."""
    return run_full_evaluation(consensus_runs=consensus_runs, similarity_threshold=similarity_threshold)

if __name__ == "__main__":
    # Default behavior - single run
    run_full_evaluation() 