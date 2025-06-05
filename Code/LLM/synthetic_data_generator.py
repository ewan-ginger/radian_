import json
import random

# --- Configuration ---
GAME_DURATION = 60.0  # seconds
MAX_STAMINA = 100.0
BASE_FATIGUE_RATE = 5.0  # Stamina points lost per action
BASE_RECOVERY_RATE = 2.0  # Stamina points recovered per second of rest

# Player IDs
TEAM_A_PLAYER_IDS = ["Player_1", "Player_2"]
TEAM_B_PLAYER_IDS = ["Player_3", "Player_4"]
GOALIE_ID = "Goalie"
ALL_FIELD_PLAYER_IDS = TEAM_A_PLAYER_IDS + TEAM_B_PLAYER_IDS

# Player data - will be randomized for each game
PLAYERS_DATA = {}

def initialize_random_players():
    """Initialize players with random skills for game variety."""
    global PLAYERS_DATA
    PLAYERS_DATA = {
        "Player_1": {
            "passing_skill": random.uniform(0.7, 0.9), "shooting_skill": random.uniform(0.7, 0.9),
            "faceoff_skill": random.uniform(0.6, 0.8), "ball_handling_skill": random.uniform(0.7, 0.9),
            "stamina": MAX_STAMINA, "fatigue_rate": BASE_FATIGUE_RATE, "recovery_rate": BASE_RECOVERY_RATE
        },
        "Player_2": {
            "passing_skill": random.uniform(0.7, 0.9), "shooting_skill": random.uniform(0.7, 0.9),
            "faceoff_skill": random.uniform(0.6, 0.8), "ball_handling_skill": random.uniform(0.7, 0.9),
            "stamina": MAX_STAMINA, "fatigue_rate": BASE_FATIGUE_RATE, "recovery_rate": BASE_RECOVERY_RATE
        },
        "Player_3": {
            "passing_skill": random.uniform(0.7, 0.9), "shooting_skill": random.uniform(0.7, 0.9),
            "faceoff_skill": random.uniform(0.6, 0.8), "ball_handling_skill": random.uniform(0.7, 0.9),
            "stamina": MAX_STAMINA, "fatigue_rate": BASE_FATIGUE_RATE, "recovery_rate": BASE_RECOVERY_RATE
        },
        "Player_4": {
            "passing_skill": random.uniform(0.7, 0.9), "shooting_skill": random.uniform(0.7, 0.9),
            "faceoff_skill": random.uniform(0.6, 0.8), "ball_handling_skill": random.uniform(0.7, 0.9),
            "stamina": MAX_STAMINA, "fatigue_rate": BASE_FATIGUE_RATE, "recovery_rate": BASE_RECOVERY_RATE
        },
        "Goalie": {
            "passing_skill": random.uniform(0.8, 1.0), "shooting_skill": 0.0, "faceoff_skill": 0.0,
            "ball_handling_skill": random.uniform(0.8, 1.0), "stamina": MAX_STAMINA,
            "fatigue_rate": BASE_FATIGUE_RATE * 0.5, "recovery_rate": BASE_RECOVERY_RATE * 1.5
        }
    }

# Initialize with default random values
initialize_random_players()

# --- Action Definitions ---
ACTION_FACEOFF = "faceoff"
ACTION_PASS = "pass"
ACTION_CATCH = "catch"
ACTION_SHOT = "shot"
ACTION_SAVE_ATTEMPT = "save_attempt"
ACTION_GROUNDBALL = "groundball"

# --- Time Delays (min, max) seconds ---
TIME_FACEOFF_EXECUTION = (0.5, 1.0)
TIME_PASS_EXECUTION = (0.2, 1.5)  # More variable
TIME_CATCH_EXECUTION = (0.1, 0.5)  # More variable
TIME_SHOT_EXECUTION = (0.2, 1.0)  # More variable
TIME_SAVE_ATTEMPT_EXECUTION = (0.1, 0.6)  # More variable
TIME_GROUNDBALL_PICKUP_EXECUTION = (0.8, 2.0)  # More variable
TIME_CLEAR_EXECUTION = (1.0, 3.0)  # Time for clearing the ball

DELAY_POST_FACEOFF_WIN = (1.0, 4.0)  # More variable
DELAY_PLAYER_POSSESSION_MOVEMENT = (1.0, 10.0)  # More variable
DELAY_GOALIE_POST_SAVE_LOOK = (1.0, 3.0)
DELAY_AFTER_GOAL_RESET = (5.0, 10.0)  # More variable
DELAY_AFTER_EVENT_FOR_GROUNDBALL = (0.5, 2.0)  # More variable

# --- Base Probabilities ---
BASE_FACEOFF_WIN_RATE = 0.5
BASE_PASS_COMPLETION_RATE = 0.80
BASE_SHOT_ON_GOAL_RATE = 0.65
BASE_GOALIE_SAVE_RATE_ON_TARGET = 0.60
BASE_TURNOVER_RATE_DURING_MOVEMENT = 0.08
BASE_CLEAR_SUCCESS_RATE = 0.70  # Base chance of successful clear
BASE_CLEAR_PASS_SUCCESS_RATE = 0.85  # Base chance of successful clear pass

# --- New Probabilities for Variability ---
CHANCE_OF_DEAD_TIME = 0.2  # 20% chance of dead time after any event
DEAD_TIME_RANGE = (0.5, 3.0)  # Random dead time between 0.5 and 3 seconds
CHANCE_OF_BURST_EVENTS = 0.1  # 10% chance of burst events (multiple quick actions)
BURST_EVENT_COUNT = (2, 4)  # Number of events in a burst

# --- Helper Functions ---
def get_player_attributes(player_id):
    """Get a player's attributes from PLAYERS_DATA."""
    return PLAYERS_DATA.get(player_id, {
        "passing_skill": 0.5,
        "shooting_skill": 0.5,
        "faceoff_skill": 0.5,
        "ball_handling_skill": 0.5,
        "stamina": MAX_STAMINA,
        "fatigue_rate": BASE_FATIGUE_RATE,
        "recovery_rate": BASE_RECOVERY_RATE
    })

def get_fatigue_modifier(player_id):
    """Calculate performance modifier based on current stamina."""
    player = get_player_attributes(player_id)
    stamina_ratio = player["stamina"] / MAX_STAMINA
    # Linear scaling from 0.5 (exhausted) to 1.0 (full stamina)
    return 0.5 + (0.5 * stamina_ratio)

def apply_fatigue(player_id, action_type):
    """Apply fatigue to a player based on the action performed."""
    if player_id is None:  # Handle None case
        return
    player = get_player_attributes(player_id)
    fatigue_multiplier = {
        ACTION_FACEOFF: 1.5,
        ACTION_PASS: 1.0,
        ACTION_SHOT: 1.2,
        "movement": 0.8
    }.get(action_type, 1.0)
    
    fatigue_amount = player["fatigue_rate"] * fatigue_multiplier
    PLAYERS_DATA[player_id]["stamina"] = max(0, player["stamina"] - fatigue_amount)

def recover_stamina(player_id, time_seconds):
    """Recover stamina for a player during rest periods."""
    if player_id is None:  # Handle None case
        return
    player = get_player_attributes(player_id)
    recovery_amount = player["recovery_rate"] * time_seconds
    PLAYERS_DATA[player_id]["stamina"] = min(MAX_STAMINA, player["stamina"] + recovery_amount)

def get_teammate(player_with_ball, team_players_list):
    """Get a random teammate from the player's team."""
    if player_with_ball not in team_players_list and player_with_ball != GOALIE_ID:
        print(f"WARNING: {player_with_ball} not in their supposed team {team_players_list} for get_teammate.")
        return random.choice([p for p in ALL_FIELD_PLAYER_IDS if p != player_with_ball])
    
    options = [p for p in team_players_list if p != player_with_ball]
    if not options:
        if player_with_ball == GOALIE_ID and team_players_list:
            return random.choice(team_players_list)
        print(f"ERROR: No teammates found for {player_with_ball} in {team_players_list}")
        return random.choice([p for p in ALL_FIELD_PLAYER_IDS if p != player_with_ball])
    return random.choice(options)

def get_random_player_from_team(team_list):
    """Get a random player from the specified team."""
    return random.choice(team_list)

def log_event(events_list, timestamp, player, action, facing, details=None):
    """Log a game event."""
    event = {
        "timestamp": round(timestamp, 2),
        "player": player,
        "action": action,
        "facing": facing
    }
    events_list.append(event)
    print(f"{event['timestamp']:.2f}s | {player} | {action} | Facing: {facing}" + (f" | Details: {details}" if details else ""))

def add_random_dead_time(current_time):
    """Add random dead time after an event."""
    if random.random() < CHANCE_OF_DEAD_TIME:
        dead_time = random.uniform(DEAD_TIME_RANGE[0], DEAD_TIME_RANGE[1])
        return current_time + dead_time
    return current_time

def should_have_burst_events():
    """Determine if there should be a burst of quick events."""
    return random.random() < CHANCE_OF_BURST_EVENTS

# --- Main Game Simulation ---
def generate_lacrosse_game_data_v3():
    # Initialize random player skills for this game
    initialize_random_players()
    
    events = []
    current_time = 0.0
    game_score = {"TeamA": 0, "TeamB": 0}

    possession_player = None
    offensive_team_id = None # Renamed from attacking_team_id

    # --- Helper function to get player's team ---
    def get_team_of_player(p_id):
        if p_id in TEAM_A_PLAYER_IDS: return "TeamA"
        if p_id in TEAM_B_PLAYER_IDS: return "TeamB"
        if p_id == GOALIE_ID:
            # Goalie is on the team that is currently non-offensive
            if offensive_team_id == "TeamA": return "TeamB"
            if offensive_team_id == "TeamB": return "TeamA"
            return None # Should be set if offensive_team_id is known
        return None

    def get_players_of_team(team_id_str): # Wrapper for existing get_team_players_by_id
        return get_team_players_by_id(team_id_str)

    def get_offensive_team_players(): # Renamed
        return TEAM_A_PLAYER_IDS if offensive_team_id == "TeamA" else TEAM_B_PLAYER_IDS

    def get_non_offensive_team_players(): # Renamed
        return TEAM_B_PLAYER_IDS if offensive_team_id == "TeamA" else TEAM_A_PLAYER_IDS
    
    def get_team_players_by_id(team_id_str):
        if team_id_str == "TeamA": return TEAM_A_PLAYER_IDS
        if team_id_str == "TeamB": return TEAM_B_PLAYER_IDS
        return []

    def initiate_faceoff():
        nonlocal current_time, possession_player, offensive_team_id # offensive_team_id updated here
        
        if current_time > 0:
                 current_time += random.uniform(DELAY_AFTER_GOAL_RESET[0], DELAY_AFTER_GOAL_RESET[1])
        if current_time >= GAME_DURATION: return False

        faceoff_player_A = random.choice(TEAM_A_PLAYER_IDS)
        faceoff_player_B = random.choice(TEAM_B_PLAYER_IDS)
        
        player_A_skill = get_player_attributes(faceoff_player_A)["faceoff_skill"] * get_fatigue_modifier(faceoff_player_A)
        player_B_skill = get_player_attributes(faceoff_player_B)["faceoff_skill"] * get_fatigue_modifier(faceoff_player_B)
        
        action_time = random.uniform(TIME_FACEOFF_EXECUTION[0], TIME_FACEOFF_EXECUTION[1])
        log_event(events, current_time, faceoff_player_A, ACTION_FACEOFF, faceoff_player_B)
        log_event(events, current_time, faceoff_player_B, ACTION_FACEOFF, faceoff_player_A)
        
        apply_fatigue(faceoff_player_A, ACTION_FACEOFF)
        apply_fatigue(faceoff_player_B, ACTION_FACEOFF)
        
        current_time += action_time
        if current_time >= GAME_DURATION: return False

        total_skill = player_A_skill + player_B_skill
        faceoff_winner_on_draw = faceoff_player_A if random.random() < (player_A_skill / total_skill) else faceoff_player_B
        faceoff_winner_on_draw_team = get_team_of_player(faceoff_winner_on_draw)

        current_time += random.uniform(0.2, 0.5)
        if current_time >= GAME_DURATION: return False

        gb_roll = random.random()
        if gb_roll < 0.7:
            gb_winner = faceoff_winner_on_draw
        elif gb_roll < 0.9:
            gb_winner = get_teammate(faceoff_winner_on_draw, get_players_of_team(faceoff_winner_on_draw_team))
        else:
            opponent_team_id = "TeamB" if faceoff_winner_on_draw_team == "TeamA" else "TeamA"
            gb_winner = random.choice(get_players_of_team(opponent_team_id))

        log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, faceoff_winner_on_draw, 
                  details="Faceoff Groundball" + (" (Winner of Draw)" if gb_winner == faceoff_winner_on_draw else ""))
        
        current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
        if current_time >= GAME_DURATION: return False

        possession_player = gb_winner
        offensive_team_id = get_team_of_player(possession_player) # Offensive team set by faceoff winner
        if offensive_team_id is None and possession_player == GOALIE_ID: # Edge case: Goalie gets faceoff GB
             # If goalie gets it, their team becomes offensive. Assume goalie belongs to a default team or assign based on context.
             # For now, let's assign Goalie to TeamB if offensive_team_id is None. This needs robust handling if goalie plays for a team.
             # Simplified: if goalie wins GB, let's say Team B becomes offensive for example purposes.
             offensive_team_id = "TeamB" # Placeholder for robust goalie team assignment pre-offensive_team_id
             print(f"WARNING: Goalie {GOALIE_ID} won faceoff GB, setting offensive_team_id to {offensive_team_id} by default.")


        current_time += random.uniform(DELAY_POST_FACEOFF_WIN[0], DELAY_POST_FACEOFF_WIN[1])
        print(f"--- {faceoff_winner_on_draw} wins faceoff draw, {gb_winner} picks up groundball. Offensive Team: {offensive_team_id}. CT: {current_time:.2f}s ---")
        return current_time < GAME_DURATION

    # --- Start Game ---
    current_time = random.uniform(0.0, 5.0)
    print(f"--- Game starting at {current_time:.2f}s ---")
    
    if not initiate_faceoff():
        return format_output_v3(events, game_score)

    while current_time < GAME_DURATION:
        if possession_player is None:
            if not initiate_faceoff(): break
            if possession_player is None:
                print("CRITICAL ERROR: No possession after faceoff routine.")
                break 
        
        player_s_actual_team = get_team_of_player(possession_player)
        if player_s_actual_team is None:
            print(f"CRITICAL ERROR: Player {possession_player} has no discernible team. Offensive: {offensive_team_id}")
            break

        # --- Burst Events ---
        if should_have_burst_events():
            burst_count = random.randint(BURST_EVENT_COUNT[0], BURST_EVENT_COUNT[1])
            print(f"--- Burst of {burst_count} events starting. CT: {current_time:.2f}s ---")
            for _ in range(burst_count):
                if current_time >= GAME_DURATION: break
                passer = possession_player
                passer_team = get_team_of_player(passer)
                receiver = get_teammate(passer, get_players_of_team(passer_team))
                pass_time = random.uniform(0.1, 0.3)
                log_event(events, current_time, passer, ACTION_PASS, receiver)
                apply_fatigue(passer, ACTION_PASS)
                current_time += pass_time
                
                if random.random() < 0.8:
                    catch_time = random.uniform(0.1, 0.2)
                    log_event(events, current_time, receiver, ACTION_CATCH, passer)
                    current_time += catch_time
                    possession_player = receiver
                else: 
                    gb_winner = random.choice(ALL_FIELD_PLAYER_IDS + [GOALIE_ID]) 
                    log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, passer, details="Recovered Incomplete Burst Pass")
                    current_time += random.uniform(0.2, 0.4)
                    possession_player = gb_winner
                    # offensive_team_id does NOT change on turnover
            # End of for loop for burst events
            if current_time >= GAME_DURATION: break 
            if possession_player is not None: 
                continue 

        # --- REGULAR POSSESSION PHASE ---
        if possession_player != GOALIE_ID:
            current_time += random.uniform(DELAY_PLAYER_POSSESSION_MOVEMENT[0], DELAY_PLAYER_POSSESSION_MOVEMENT[1])
            apply_fatigue(possession_player, "movement")
        if current_time >= GAME_DURATION: break

        current_time = add_random_dead_time(current_time)
        if current_time >= GAME_DURATION: break

        # --- Turnover check during movement ---
        if possession_player != GOALIE_ID:
            player_skill = get_player_attributes(possession_player)["ball_handling_skill"]
            fatigue_mod = get_fatigue_modifier(possession_player)
            turnover_rate = BASE_TURNOVER_RATE_DURING_MOVEMENT / (player_skill * fatigue_mod)
            
            if random.random() < turnover_rate:
                turnover_by_player = possession_player
                turnover_by_player_team = get_team_of_player(turnover_by_player)
                # Opponent recovers
                recovering_opponent_team_id = "TeamB" if turnover_by_player_team == "TeamA" else "TeamA"
                recovering_opponent = get_random_player_from_team(get_players_of_team(recovering_opponent_team_id))
                
                log_event(events, current_time, turnover_by_player, ACTION_GROUNDBALL, recovering_opponent, details="Lost Ball Turnover")
                current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
                if current_time >= GAME_DURATION: break
                
                possession_player = recovering_opponent
                log_event(events, current_time, possession_player, ACTION_GROUNDBALL, turnover_by_player, details="Recovered Turnover")
                # offensive_team_id does NOT change
                recovering_team_id = get_team_of_player(possession_player)
                print(f"--- Turnover! {possession_player} (Team {recovering_team_id}) has possession. Offensive Team still {offensive_team_id}. CT: {current_time:.2f}s ---")
                if current_time >= GAME_DURATION: break
                continue # Re-evaluate state

        # --- Action Choice based on offensive/defensive possession ---
        action_choice_rand = random.random()
        player_s_actual_team = get_team_of_player(possession_player) # Re-fetch in case it changed

        if player_s_actual_team != offensive_team_id:
            # --- DEFENSIVE POSSESSION STATE ---
            print(f"--- {possession_player} (Team {player_s_actual_team}) in Defensive Possession (Offensive: {offensive_team_id}). Opting to pass/clear. ---")
            
            action_is_clear = False # Default: pass

            if possession_player == GOALIE_ID:
                # Goalie: High chance to pass to a defender, low chance to clear directly.
                CHANCE_GOALIE_ATTEMPTS_CLEAR_DIRECTLY = 0.2 # e.g., 20% chance goalie clears, 80% goalie passes.
                if random.random() < CHANCE_GOALIE_ATTEMPTS_CLEAR_DIRECTLY:
                    action_is_clear = True  # Goalie will attempt to clear
                else:
                    action_is_clear = False # Goalie will pass
            else:
                # Field player in defensive possession: Higher chance to clear themselves.
                CHANCE_FIELD_PLAYER_ATTEMPTS_CLEAR = random.uniform(0.6, 0.9) # 60-90% chance to clear
                if random.random() < CHANCE_FIELD_PLAYER_ATTEMPTS_CLEAR:
                    action_is_clear = True
                else:
                    action_is_clear = False

            if action_is_clear:
                # ATTEMPT CLEAR
                print(f"--- {possession_player} attempting to clear. CT: {current_time:.2f}s ---")
                clear_time = random.uniform(TIME_CLEAR_EXECUTION[0], TIME_CLEAR_EXECUTION[1])
                current_time += clear_time
                if current_time >= GAME_DURATION: break

                player_skill = get_player_attributes(possession_player)["ball_handling_skill"]
                fatigue_mod = get_fatigue_modifier(possession_player)
                clear_success_rate = (BASE_CLEAR_PASS_SUCCESS_RATE if possession_player == GOALIE_ID else BASE_CLEAR_SUCCESS_RATE) * player_skill * fatigue_mod

                if random.random() < clear_success_rate:
                    print(f"--- Successful clear by {possession_player}. CT: {current_time:.2f}s ---")
                    possession_player = None  # Reset for faceoff. offensive_team_id will be set by faceoff.
                else: # Failed Clear
                    current_time += random.uniform(0.2, 0.5) # GB situation
                    if current_time >= GAME_DURATION: break
                    # Opponent (current offensive_team_id) likely recovers
                    gb_winner = random.choice(get_offensive_team_players() + ([GOALIE_ID] if GOALIE_ID not in get_offensive_team_players() else [])) # Higher chance for offensive team
                    log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, possession_player, details="Recovered Failed Clear")
                    current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
                    possession_player = gb_winner
                    # offensive_team_id does NOT change
                    print(f"--- Failed clear by {turnover_by_player if 'turnover_by_player' in locals() else possession_player}, {possession_player} recovers. Offensive Team still {offensive_team_id}. CT: {current_time:.2f}s ---")
            else:
                # DEFENSIVE PASS (pass to own team)
                passer = possession_player
                team_to_pass_to = get_players_of_team(player_s_actual_team)
                receiver = get_teammate(passer, team_to_pass_to)
                if receiver is None: break # Should not happen with proper team lists

                print(f"--- {passer} (Defensive Possession) passing to {receiver}. CT: {current_time:.2f}s ---")
                pass_time = random.uniform(TIME_PASS_EXECUTION[0], TIME_PASS_EXECUTION[1])
                log_event(events, current_time, passer, ACTION_PASS, receiver)
                apply_fatigue(passer, ACTION_PASS)
                current_time += pass_time
                if current_time >= GAME_DURATION: break

                passer_skill = get_player_attributes(passer)["passing_skill"]
                fatigue_mod = get_fatigue_modifier(passer)
                pass_success_rate = BASE_PASS_COMPLETION_RATE * passer_skill * fatigue_mod

                if random.random() < pass_success_rate or passer == GOALIE_ID: # Goalie passes usually complete to own team
                    catch_time = random.uniform(TIME_CATCH_EXECUTION[0], TIME_CATCH_EXECUTION[1])
                    log_event(events, current_time, receiver, ACTION_CATCH, passer)
                    current_time += catch_time
                    possession_player = receiver # Still in defensive possession
                else: # Bad Defensive Pass
                    current_time += random.uniform(0.2, 0.5)
                    if current_time >= GAME_DURATION: break
                    # Groundball, could be anyone, but higher chance for current offensive_team_id
                    gb_roll = random.random()
                    if gb_roll < 0.6: # Offensive team gets it
                        gb_winner = random.choice(get_offensive_team_players())
                    elif gb_roll < 0.9: # Passer's (defensive) team keeps it
                        gb_winner = get_teammate(passer, team_to_pass_to) if random.random() < 0.5 else receiver
                    else: # Goalie
                        gb_winner = GOALIE_ID
                    
                    log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, passer, details="Recovered Bad Defensive Pass")
                    current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
                    possession_player = gb_winner
                    # offensive_team_id does NOT change
                    recovered_team = get_team_of_player(possession_player)
                    print(f"--- Bad defensive pass by {passer}, {gb_winner} (Team {recovered_team}) recovers. Offensive Team still {offensive_team_id}. CT: {current_time:.2f}s ---")
        else:
            # --- OFFENSIVE POSSESSION STATE ---
            # Player's team IS the offensive_team_id. They can pass or shoot.
            print(f"--- {possession_player} (Team {player_s_actual_team}) in Offensive Possession. Opting to pass/shoot. ---")

            # Goalie on offensive possession (e.g. won faceoff GB) should not shoot
            if possession_player == GOALIE_ID:
                action_is_pass = True # Force goalie to pass
            else:
                pass_threshold = random.uniform(0.4, 0.7) # Chance to pass vs shoot
                action_is_pass = action_choice_rand < pass_threshold
            
            if action_is_pass:
                # OFFENSIVE PASS
                passer = possession_player
                team_to_pass_to = get_players_of_team(player_s_actual_team) # Pass to own (offensive) team
                receiver = get_teammate(passer, team_to_pass_to)
                if receiver is None: break

                print(f"--- {passer} (Offensive Possession) passing to {receiver}. CT: {current_time:.2f}s ---")
                pass_time = random.uniform(TIME_PASS_EXECUTION[0], TIME_PASS_EXECUTION[1])
                log_event(events, current_time, passer, ACTION_PASS, receiver)
                apply_fatigue(passer, ACTION_PASS)
                current_time += pass_time
                if current_time >= GAME_DURATION: break

                passer_skill = get_player_attributes(passer)["passing_skill"]
                fatigue_mod = get_fatigue_modifier(passer)
                pass_success_rate = BASE_PASS_COMPLETION_RATE * passer_skill * fatigue_mod

                if random.random() < pass_success_rate:
                    catch_time = random.uniform(TIME_CATCH_EXECUTION[0], TIME_CATCH_EXECUTION[1])
                    log_event(events, current_time, receiver, ACTION_CATCH, passer)
                    current_time += catch_time
                    possession_player = receiver # Still in offensive possession
                else: # Bad Offensive Pass
                    current_time += random.uniform(0.2, 0.5)
                    if current_time >= GAME_DURATION: break
                    # Groundball, higher chance for non_offensive_team (defenders)
                    gb_roll = random.random()
                    if gb_roll < 0.6: # Defensive team gets it
                        gb_winner = random.choice(get_non_offensive_team_players())
                    elif gb_roll < 0.9: # Passer's (offensive) team keeps it
                        gb_winner = get_teammate(passer, team_to_pass_to) if random.random() < 0.5 else receiver
                    else: # Goalie
                        gb_winner = GOALIE_ID
                        
                    log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, passer, details="Recovered Bad Offensive Pass")
                    current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
                    possession_player = gb_winner
                    # offensive_team_id does NOT change
                    recovered_team = get_team_of_player(possession_player)
                    print(f"--- Bad offensive pass by {passer}, {gb_winner} (Team {recovered_team}) recovers. Offensive Team still {offensive_team_id}. CT: {current_time:.2f}s ---")
            else:
                # SHOT
                shooter = possession_player
                print(f"--- {shooter} (Offensive Possession) taking a SHOT. CT: {current_time:.2f}s ---")
                shot_time = random.uniform(TIME_SHOT_EXECUTION[0], TIME_SHOT_EXECUTION[1])
                # GOALIE_ID is always the defending goalie
                log_event(events, current_time, shooter, ACTION_SHOT, GOALIE_ID)
                apply_fatigue(shooter, ACTION_SHOT)
                current_time += shot_time
                if current_time >= GAME_DURATION: break

                save_attempt_time = random.uniform(TIME_SAVE_ATTEMPT_EXECUTION[0], TIME_SAVE_ATTEMPT_EXECUTION[1])
                
                shooter_skill = get_player_attributes(shooter)["shooting_skill"]
                fatigue_mod = get_fatigue_modifier(shooter)
                shot_on_goal_rate = BASE_SHOT_ON_GOAL_RATE * shooter_skill * fatigue_mod # Renamed from shot_success_rate

                if random.random() < shot_on_goal_rate: # Shot is on target
                    # Goalie save attempt (GOALIE_ID is on non_offensive_team)
                    goalie_skill = get_player_attributes(GOALIE_ID)["ball_handling_skill"] # Using ball_handling as proxy for save skill
                    goalie_fatigue_mod = get_fatigue_modifier(GOALIE_ID)
                    actual_goalie_save_rate = BASE_GOALIE_SAVE_RATE_ON_TARGET * (goalie_skill * goalie_fatigue_mod)


                    if random.random() < actual_goalie_save_rate:
                        # SAVE
                        log_event(events, current_time, GOALIE_ID, ACTION_SAVE_ATTEMPT, shooter, details="Save")
                        current_time += save_attempt_time
                        print(f"--- Save by {GOALIE_ID}! ---")
                        possession_player = GOALIE_ID
                        # offensive_team_id does NOT change. Goalie is now in defensive possession.
                        if current_time >= GAME_DURATION: break
                        # Loop will continue, goalie will be in defensive possession state
                    else:
                        # GOAL
                        log_event(events, current_time, GOALIE_ID, ACTION_SAVE_ATTEMPT, shooter, details="Goal")
                        current_time += save_attempt_time
                        if offensive_team_id == "TeamA": game_score["TeamA"] +=1
                        else: game_score["TeamB"] +=1
                        print(f"--- GOAL by {shooter} ({offensive_team_id})! Score A:{game_score['TeamA']}-B:{game_score['TeamB']} CT: {current_time:.2f}s ---")
                        possession_player = None # Reset for faceoff
                else: # Shot is OFF target (Missed Shot)
                    current_time += random.uniform(0.2, 0.5) # Quick delay for groundball
                    if current_time >= GAME_DURATION: break
                    
                    # Determine who gets groundball (can be anyone, but weighted)
                    gb_roll = random.random()
                    if gb_roll < 0.35: # Shooter's (offensive) team
                        gb_winner = get_teammate(shooter, get_players_of_team(offensive_team_id))
                    elif gb_roll < 0.60: # Shooter
                        gb_winner = shooter
                    elif gb_roll < 0.85: # Non-offensive team
                        gb_winner = random.choice(get_non_offensive_team_players())
                    else: # Goalie
                        gb_winner = GOALIE_ID

                    log_event(events, current_time, gb_winner, ACTION_GROUNDBALL, shooter, details="Recovered Missed Shot")
                    current_time += random.uniform(TIME_GROUNDBALL_PICKUP_EXECUTION[0], TIME_GROUNDBALL_PICKUP_EXECUTION[1])
                    possession_player = gb_winner
                    # offensive_team_id does NOT change
                    recovered_team = get_team_of_player(possession_player)
                    print(f"--- Missed shot by {shooter}, {possession_player} (Team {recovered_team}) recovers. Offensive Team still {offensive_team_id}. CT: {current_time:.2f}s ---")
            if current_time >= GAME_DURATION: break # End of action choice if game ended
            
    return format_output_v3(events, game_score)

def format_output_v3(events_list, final_score):
    game_data = {
        "game_setup": {
            "team_A": TEAM_A_PLAYER_IDS,
            "team_B": TEAM_B_PLAYER_IDS,
            "goalie": GOALIE_ID,
            "duration_seconds": GAME_DURATION,
            "sim_actions": [ACTION_FACEOFF, ACTION_PASS, ACTION_CATCH, ACTION_SHOT, ACTION_SAVE_ATTEMPT, ACTION_GROUNDBALL]
        },
        "final_score": final_score,
        "events": events_list
    }
    return game_data

if __name__ == "__main__":
    # Generate one set of ground truth data
    ground_truth_game_data = generate_lacrosse_game_data_v3()
    
    output_filename_gt = "2v2_groundtruth.json"
    with open(output_filename_gt, 'w') as f:
        json.dump(ground_truth_game_data, f, indent=4)
    print(f"\nGround truth data generated ({len(ground_truth_game_data['events'])} events) and saved to {output_filename_gt}")
    print(f"Final Score (Ground Truth): TeamA {ground_truth_game_data['final_score']['TeamA']} - TeamB {ground_truth_game_data['final_score']['TeamB']}")

    