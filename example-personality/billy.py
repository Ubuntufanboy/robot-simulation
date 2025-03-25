import math
import random

class Personality:
    """
    A personality is the *brain* of the robot that makes decisions. Here is an example personality Billy. Billy just moves toward the ball and nothing else.
    """

    def __init__(self):
        self.name = "Billy the bold!"
        self.target_x = None
        self.target_y = None
        self.state = "chase_ball"  # Possible states: chase_ball, attack_goal, defend
        self.decision_cooldown = 0

    def get_action(self, game_state):
        """
        This is the most important function. In this function, you can decide what you want to do given the state of the game (in a dict).
        You must return a vector in the form of a tuple (magnatude, direction).
        """
        # Find this robot's data in game state
        own_robot = None
        for robot in game_state['robots']:
            if robot['id'] == 0 and robot['team'] == 0:
                own_robot = robot
                break

        if not own_robot:
            return 0, 0  # No action if robot not found

        # cooldown (which might be helpful if you are on defense to prevent getting faked or spazzing out)
        if self.decision_cooldown > 0:
            self.decision_cooldown -= 1

        # Get ball pos
        ball_x = game_state['ball']['x']
        ball_y = game_state['ball']['y']

        # Find the goal you should be scoring on
        if own_robot['team'] == 0:  # Team A
            goal_x = game_state['field']['width'] / 2
            goal_y = game_state['field']['height']  # BOTTOM goal
        else:  # Team B
            goal_x = game_state['field']['width'] / 2
            goal_y = 0  # TOP goal

        # Do something... unless you are on cooldown
        if self.decision_cooldown == 0:
            # Check if robot has the ball. Then try to score into the net
            if own_robot['has_ball']:
                self.state = "attack_goal"
            else:
                teammate_has_ball = False
                for robot in game_state['robots']:
                    if robot['team'] == own_robot['team'] and robot['has_ball']:
                        teammate_has_ball = True
                        break

                if teammate_has_ball:
                    self.state = "support_attack"
                else:
                    self.state = "chase_ball"

            self.decision_cooldown = 10  # Wait 10 frames before changing decision. (To prevent loops or spazzing)

        # Act based on the robot's decision
        if self.state == "chase_ball":
            target_x = ball_x
            target_y = ball_y
        elif self.state == "attack_goal":
            target_x = goal_x
            target_y = goal_y
        elif self.state == "support_attack":
            # Maybe a pass or rebound
            if own_robot['team'] == 0:  # Team A
                target_x = ball_x + 30 if ball_x < game_state['field']['width'] / 2 else ball_x - 30
                target_y = ball_y + 30  # AHEAD of the ball
            else:  # Team B
                target_x = ball_x + 30 if ball_x < game_state['field']['width'] / 2 else ball_x - 30
                target_y = ball_y - 30  # AHEAD of the ball
        else:
            # Default to chasing the ball... cuz he's just billy afterall
            target_x = ball_x
            target_y = ball_y

        # Calculate angle
        dx = target_x - own_robot['x']
        dy = target_y - own_robot['y']
        angle_to_target = math.atan2(dy, dx)

        # Calculate angle difference to turn in
        angle_diff = angle_to_target - own_robot['angle']
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Calculate angle change proportional to the difference
        # Use a higher factor for quicker turning
        # The reason we have this is because we don't want to waste angular movement
        angle_change = angle_diff * 0.5
        distance = math.sqrt(dx**2 + dy**2)

        # Determine acceleration
        if self.state == "attack_goal" and own_robot['has_ball']:
            # PEDAL TO THE METAL if has ball
            acceleration = 10.0
        elif distance < 10:
            acceleration = 5.0
        else:
            acceleration = 8.0

        # Add some noise to keep things interesting
        if random.random() < 0.25:  # 5% chance
            angle_change += random.uniform(-0.2, 0.2)

        return acceleration, angle_change

# Create an instance to be used by the simulation
personality = Personality()

# This function will be called by the simulation
def get_action(game_state):
    return personality.get_action(game_state)
