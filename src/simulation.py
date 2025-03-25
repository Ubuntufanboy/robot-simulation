import pygame
import sys
import math
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any, Callable
import importlib.util
import os
import random

FIELD_WIDTH = 158
FIELD_HEIGHT = 219
OUTER_AREA_WIDTH = 12
TOTAL_WIDTH = 182
TOTAL_HEIGHT = 243
GOAL_WIDTH = 79
PENALTY_SEMICIRCLE_RADIUS = 4
ROBOT_RADIUS = 9 
BALL_RADIUS = 2.1
FRICTION_COEFFICIENT = 0.4
ROBOT_MASS = 2
MAX_VELOCITY = 18
MIN_ACCELERATION = -10
MAX_ACCELERATION = 15
PENALTY_TIME = 60
IDLE_TIME_LIMIT = 5
WEDGE_BASE = 10
WEDGE_RISE = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

SCALE_FACTOR = 3  # 3 pixels per cm

FIELD_WIDTH_PX = int(FIELD_WIDTH * SCALE_FACTOR)
FIELD_HEIGHT_PX = int(FIELD_HEIGHT * SCALE_FACTOR)
OUTER_AREA_WIDTH_PX = int(OUTER_AREA_WIDTH * SCALE_FACTOR)
TOTAL_WIDTH_PX = int(TOTAL_WIDTH * SCALE_FACTOR)
TOTAL_HEIGHT_PX = int(TOTAL_HEIGHT * SCALE_FACTOR)
GOAL_WIDTH_PX = int(GOAL_WIDTH * SCALE_FACTOR)
PENALTY_SEMICIRCLE_RADIUS_PX = int(PENALTY_SEMICIRCLE_RADIUS * SCALE_FACTOR)
ROBOT_RADIUS_PX = int(ROBOT_RADIUS * SCALE_FACTOR)
BALL_RADIUS_PX = int(BALL_RADIUS * SCALE_FACTOR)
WEDGE_BASE_PX = int(WEDGE_BASE * SCALE_FACTOR)
WEDGE_RISE_PX = int(WEDGE_RISE * SCALE_FACTOR)

# This makes things a lot easier
TEAM_A = 0
TEAM_B = 1

class Ball:
    def __init__(self):
        self.x = TOTAL_WIDTH_PX / 2
        self.y = TOTAL_HEIGHT_PX / 2
        self.vx = 0
        self.vy = 0
        self.radius = BALL_RADIUS_PX
        
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Apply friction
        friction_deceleration = FRICTION_COEFFICIENT * 9.8  # Force (In Newtons) = Friction Coefficent * mass * Accel 
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 0:
            deceleration = min(friction_deceleration * dt, speed)
            self.vx -= (self.vx / speed) * deceleration
            self.vy -= (self.vy / speed) * deceleration
            
        self._handle_boundary_collisions()
    
    def _handle_boundary_collisions(self):
        left_bound = OUTER_AREA_WIDTH_PX
        right_bound = TOTAL_WIDTH_PX - OUTER_AREA_WIDTH_PX
        top_bound = OUTER_AREA_WIDTH_PX
        bottom_bound = TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX
        
        goal_left = (TOTAL_WIDTH_PX - GOAL_WIDTH_PX) / 2
        goal_right = (TOTAL_WIDTH_PX + GOAL_WIDTH_PX) / 2
        
        if self.x - self.radius < left_bound:
            if self.y >= top_bound and self.y <= top_bound + ROBOT_RADIUS_PX and self.x >= goal_left and self.x <= goal_right:
                pass # We chillin
            else:
                # Roll the ball. This means the ball is on the wedge
                self.x = left_bound + self.radius
                self.vx = abs(self.vx) * 0.8  # Not smart enough to calculate it. Using an approximation of 0.8
        
        if self.x + self.radius > right_bound:
            if self.y >= bottom_bound - ROBOT_RADIUS_PX and self.y <= bottom_bound and self.x >= goal_left and self.x <= goal_right:
                pass
            else:
                self.x = right_bound - self.radius
                self.vx = -abs(self.vx) * 0.8
        
        if self.y - self.radius < top_bound:
            if self.x >= goal_left and self.x <= goal_right:
                pass
            else:
                self.y = top_bound + self.radius
                self.vy = abs(self.vy) * 0.8  # Same as before
        
        if self.y + self.radius > bottom_bound:
            if self.x >= goal_left and self.x <= goal_right:
                pass
            else:
                self.y = bottom_bound - self.radius
                self.vy = -abs(self.vy) * 0.8  # I loveeeeee programmingggg
                
    def reset(self):
        # We do this when somebody scores or at kickoff
        self.x = TOTAL_WIDTH_PX / 2
        self.y = TOTAL_HEIGHT_PX / 2
        self.vx = 0
        self.vy = 0

class Robot:
    def __init__(self, x, y, team, robot_id):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.angle = 0  # RADIANS not degrees
        self.radius = ROBOT_RADIUS_PX
        self.team = team
        self.id = robot_id
        self.penalty_time = 0
        self.has_ball = False
        self.personality = None
        
    def set_personality(self, personality_module):
        self.personality = personality_module # The personality is what the user provides
        
    def update(self, dt, game_state):
        if self.penalty_time > 0:
            # Keeping track of those pesky rule breakers
            self.penalty_time -= dt
            return
        
        if self.personality:
            # Get the action from the personality
            acceleration, angle_change = self.personality.get_action(game_state)
            
            # We have to clamp down the acceleration in case a robot tries to set an insane acceleration
            acceleration = max(MIN_ACCELERATION, min(MAX_ACCELERATION, acceleration))
            
            # This isnt completely accurate to real movement but I dont wanna do calc for this lol
            self.angle += angle_change
            
            # Move in the indivisual vector componets from the main vector
            ax = acceleration * math.cos(self.angle)
            ay = acceleration * math.sin(self.angle)
            
            self.vx += ax * dt
            self.vy += ay * dt
            
            friction_deceleration = FRICTION_COEFFICIENT * 9.8
            speed = math.sqrt(self.vx**2 + self.vy**2)
            if speed > 0:
                deceleration = min(friction_deceleration * dt, speed)
                self.vx -= (self.vx / speed) * deceleration
                self.vy -= (self.vy / speed) * deceleration
            
            # Limit velocity based on terminal velocity (Not actually based on drag tho)
            speed = math.sqrt(self.vx**2 + self.vy**2)
            if speed > MAX_VELOCITY:
                self.vx = (self.vx / speed) * MAX_VELOCITY
                self.vy = (self.vy / speed) * MAX_VELOCITY
            
            self.x += self.vx * dt
            self.y += self.vy * dt
            
            self._handle_boundary_collisions()
            
    def _handle_boundary_collisions(self):
        left_bound = OUTER_AREA_WIDTH_PX
        right_bound = TOTAL_WIDTH_PX - OUTER_AREA_WIDTH_PX
        top_bound = OUTER_AREA_WIDTH_PX
        bottom_bound = TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX
        
        self.x = max(left_bound + self.radius, min(right_bound - self.radius, self.x))
        self.y = max(top_bound + self.radius, min(bottom_bound - self.radius, self.y))
        
    def is_in_penalty_area(self, field):
        # Lock up those pests!
        if self.team == TEAM_A:
            other_goal_center_x = field.goal_centers[TEAM_B][0]
            other_goal_center_y = field.goal_centers[TEAM_B][1]
            
            dist = math.sqrt((self.x - other_goal_center_x)**2 + 
                             (self.y - other_goal_center_y)**2)
            
            return dist < PENALTY_SEMICIRCLE_RADIUS_PX
        
        else:
            other_goal_center_x = field.goal_centers[TEAM_A][0]
            other_goal_center_y = field.goal_centers[TEAM_A][1]
            
            dist = math.sqrt((self.x - other_goal_center_x)**2 + 
                             (self.y - other_goal_center_y)**2)
            
            return dist < PENALTY_SEMICIRCLE_RADIUS_PX
    
    def reset_to_kickoff(self, team, position_idx):
        center_x = TOTAL_WIDTH_PX / 2
        center_y = TOTAL_HEIGHT_PX / 2
        
        if team == TEAM_A:
            if position_idx == 0:  # Adjacent left position
                self.x = center_x - ROBOT_RADIUS_PX - BALL_RADIUS_PX - 2
                self.y = center_y
            elif position_idx == 1:  # Goalie kick position
                self.x = center_x
                self.y = OUTER_AREA_WIDTH_PX + PENALTY_SEMICIRCLE_RADIUS_PX + ROBOT_RADIUS_PX + 5
        else:  # TEAM_B
            if position_idx == 0:  # Adjacent right pos
                self.x = center_x + ROBOT_RADIUS_PX + BALL_RADIUS_PX + 2
                self.y = center_y
            elif position_idx == 1:  # Goalie kick pos
                self.x = center_x
                self.y = TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX - PENALTY_SEMICIRCLE_RADIUS_PX - ROBOT_RADIUS_PX - 5
                
        self.vx = 0
        self.vy = 0
        self.angle = 0 if team == TEAM_A else math.pi # Team A's ball
        self.has_ball = False
        
    def get_front_position(self):
        front_x = self.x + self.radius * math.cos(self.angle)
        front_y = self.y + self.radius * math.sin(self.angle)
        return (front_x, front_y)

class Field:
    def __init__(self):
        self.width = FIELD_WIDTH_PX
        self.height = FIELD_HEIGHT_PX
        self.total_width = TOTAL_WIDTH_PX
        self.total_height = TOTAL_HEIGHT_PX
        self.outer_width = OUTER_AREA_WIDTH_PX
        
        self.goal_width = GOAL_WIDTH_PX
        self.goal_centers = {
            TEAM_A: (TOTAL_WIDTH_PX / 2, OUTER_AREA_WIDTH_PX),
            TEAM_B: (TOTAL_WIDTH_PX / 2, TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX)
        }
        
        goal_half_width = GOAL_WIDTH_PX / 2
        self.goal_boundaries = {
            TEAM_A: (
                TOTAL_WIDTH_PX / 2 - goal_half_width,  # lefffft
                TOTAL_WIDTH_PX / 2 + goal_half_width,  # righhhht
                OUTER_AREA_WIDTH_PX,                  # topy top
                0                                      # bottttttom
            ),
            TEAM_B: (
                TOTAL_WIDTH_PX / 2 - goal_half_width,  # leffffft (im losing my mind)
                TOTAL_WIDTH_PX / 2 + goal_half_width,  # righhhhht
                TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX, # toppy top
                TOTAL_HEIGHT_PX                        # bottttttom
            )
        }
    
    def draw(self, screen):
        """Draw the field"""
        pygame.draw.rect(screen, (100, 200, 100), (0, 0, TOTAL_WIDTH_PX, TOTAL_HEIGHT_PX))
        
        # Baller green field
        pygame.draw.rect(screen, GREEN, (
            OUTER_AREA_WIDTH_PX, 
            OUTER_AREA_WIDTH_PX, 
            FIELD_WIDTH_PX, 
            FIELD_HEIGHT_PX
        ))
        
        pygame.draw.rect(screen, WHITE, (
            OUTER_AREA_WIDTH_PX, 
            OUTER_AREA_WIDTH_PX, 
            FIELD_WIDTH_PX, 
            FIELD_HEIGHT_PX
        ), 2)
        
        pygame.draw.line(screen, WHITE, 
                         (OUTER_AREA_WIDTH_PX, TOTAL_HEIGHT_PX / 2), 
                         (TOTAL_WIDTH_PX - OUTER_AREA_WIDTH_PX, TOTAL_HEIGHT_PX / 2),
                         2)
        
        pygame.draw.circle(screen, WHITE, 
                           (TOTAL_WIDTH_PX // 2, TOTAL_HEIGHT_PX // 2), 
                           20, 2)
        
        goal_left_A = TOTAL_WIDTH_PX / 2 - GOAL_WIDTH_PX / 2
        pygame.draw.rect(screen, (200, 200, 255), (
            goal_left_A,
            0,
            GOAL_WIDTH_PX,
            OUTER_AREA_WIDTH_PX
        ))
        
        goal_left_B = TOTAL_WIDTH_PX / 2 - GOAL_WIDTH_PX / 2
        pygame.draw.rect(screen, (255, 200, 200), (
            goal_left_B,
            TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX,
            GOAL_WIDTH_PX,
            OUTER_AREA_WIDTH_PX
        ))
        
        pygame.draw.arc(screen, WHITE,
                        (TOTAL_WIDTH_PX / 2 - PENALTY_SEMICIRCLE_RADIUS_PX,
                         OUTER_AREA_WIDTH_PX - PENALTY_SEMICIRCLE_RADIUS_PX,
                         PENALTY_SEMICIRCLE_RADIUS_PX * 2,
                         PENALTY_SEMICIRCLE_RADIUS_PX * 2),
                        math.pi, 2 * math.pi, 2)
        
        pygame.draw.arc(screen, WHITE,
                        (TOTAL_WIDTH_PX / 2 - PENALTY_SEMICIRCLE_RADIUS_PX,
                         TOTAL_HEIGHT_PX - OUTER_AREA_WIDTH_PX - PENALTY_SEMICIRCLE_RADIUS_PX,
                         PENALTY_SEMICIRCLE_RADIUS_PX * 2,
                         PENALTY_SEMICIRCLE_RADIUS_PX * 2),
                        0, math.pi, 2)

    def is_ball_in_goal(self, ball):
        goal_A = self.goal_boundaries[TEAM_A]
        if (goal_A[0] <= ball.x <= goal_A[1] and 
            goal_A[3] <= ball.y <= goal_A[2] + ball.radius):
            return TEAM_B  # Team B scored a GOALLLLLLL
        
        goal_B = self.goal_boundaries[TEAM_B]
        if (goal_B[0] <= ball.x <= goal_B[1] and 
            goal_B[2] - ball.radius <= ball.y <= goal_B[3]):
            return TEAM_A 
        
        return None

class Game:
    def __init__(self, personalities_dir=None):
        self.field = Field()
        self.ball = Ball()
        
        self.robots = []
        
        self.robots.append(Robot(TOTAL_WIDTH_PX * 0.25, TOTAL_HEIGHT_PX * 0.25, TEAM_A, 0))
        self.robots.append(Robot(TOTAL_WIDTH_PX * 0.75, TOTAL_HEIGHT_PX * 0.25, TEAM_A, 1))
        
        self.robots.append(Robot(TOTAL_WIDTH_PX * 0.25, TOTAL_HEIGHT_PX * 0.75, TEAM_B, 0))
        self.robots.append(Robot(TOTAL_WIDTH_PX * 0.75, TOTAL_HEIGHT_PX * 0.75, TEAM_B, 1))
        
        self.scores = {TEAM_A: 0, TEAM_B: 0}
        self.game_time = 0
        self.last_ball_touch_time = 0
        self.kickoff_team = TEAM_A
        
        if personalities_dir:
            self.load_personalities(personalities_dir)
        
        self.running = False
        self.step_mode = False
        self.game_over = False
        self.winner = None
        self.max_game_time = 300  # 5 min halfs (We only simulate one half though)
        
    def load_personalities(self, personalities_dir):
        if not os.path.exists(personalities_dir):
            print(f"Error: Aint no '{personalities_dir}' on dis block")
            return
        
        personality_files = [f for f in os.listdir(personalities_dir) 
                            if f.endswith('.py') and not f.startswith('__')]
        
        if not personality_files:
            print(f"Aint no personality modules found in the '{personalities_dir}' block")
            return
        
        for i, personality_file in enumerate(personality_files[:min(4, len(personality_files))]):
            module_path = os.path.join(personalities_dir, personality_file)
            module_name = os.path.splitext(personality_file)[0]
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                personality_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(personality_module)
                
                if i < len(self.robots):
                    self.robots[i].set_personality(personality_module)
                    print(f"Assigned personality '{module_name}' to Robot {i}")
            except Exception as e:
                print(f"Error somehow idk '{module_name}': {e}")
    
    def reset_for_kickoff(self, scoring_team=None):
        self.ball.reset()
        
        if scoring_team is not None:
            self.kickoff_team = TEAM_A if scoring_team == TEAM_B else TEAM_B
        
        for team in [TEAM_A, TEAM_B]:
            for i in range(2):
                robot_idx = team * 2 + i
                self.robots[robot_idx].reset_to_kickoff(team, i)
        
        self.last_ball_touch_time = self.game_time
    
    def check_ball_capture(self):
        for robot in self.robots:
            if robot.penalty_time > 0:
                continue
                
            front_x, front_y = robot.get_front_position()
            
            dist = math.sqrt((front_x - self.ball.x)**2 + (front_y - self.ball.y)**2)
            
            if dist < robot.radius * 0.5 + self.ball.radius:
                robot.has_ball = True
                
                self.ball.vx = robot.vx
                self.ball.vy = robot.vy
                
                angle = robot.angle
                self.ball.x = robot.x + (robot.radius + self.ball.radius) * math.cos(angle)
                self.ball.y = robot.y + (robot.radius + self.ball.radius) * math.sin(angle)
                
                self.last_ball_touch_time = self.game_time
                
                return True
            
        return False
    
    def check_robot_collisions(self):
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                robot1 = self.robots[i]
                robot2 = self.robots[j]
                
                if robot1.penalty_time > 0 or robot2.penalty_time > 0:
                    continue
                
                dist = math.sqrt((robot1.x - robot2.x)**2 + (robot1.y - robot2.y)**2)
                
                if dist < robot1.radius + robot2.radius:
                    nx = (robot2.x - robot1.x) / dist
                    ny = (robot2.y - robot1.y) / dist
                    
                    vx = robot2.vx - robot1.vx
                    vy = robot2.vy - robot1.vy
                    
                    vn = vx * nx + vy * ny
                    
                    if vn > 0:
                        continue
                    
                    # INPULSE(J) = F * t 
                    j = -(1 + 0.8) * vn  # When I measured it, 0.8 was roughly coefficient of restitution
                    j /= 1/ROBOT_MASS + 1/ROBOT_MASS
                    
                    # Apply our impulse
                    robot1.vx -= j * nx / ROBOT_MASS
                    robot1.vy -= j * ny / ROBOT_MASS
                    robot2.vx += j * nx / ROBOT_MASS
                    robot2.vy += j * ny / ROBOT_MASS
                    
                    # If a robot has the ball, it loses it... Cuz it bumped
                    if robot1.has_ball:
                        robot1.has_ball = False
                        # Put some arbitrary velocity on the ball
                        self.ball.vx = robot1.vx * 1.5
                        self.ball.vy = robot1.vy * 1.5
                    
                    if robot2.has_ball:
                        robot2.has_ball = False
                        # Same here
                        self.ball.vx = robot2.vx * 1.5
                        self.ball.vy = robot2.vy * 1.5
    
    def check_ball_robot_collisions(self):
        for robot in self.robots:
            if robot.penalty_time > 0 or robot.has_ball:
                continue
            
            dist = math.sqrt((robot.x - self.ball.x)**2 + (robot.y - self.ball.y)**2)
            
            if dist < robot.radius + self.ball.radius:
                nx = (self.ball.x - robot.x) / dist
                ny = (self.ball.y - robot.y) / dist
                
                vx = self.ball.vx - robot.vx
                vy = self.ball.vy - robot.vy
                
                vn = vx * nx + vy * ny
                
                if vn > 0:
                    continue
                
                mass_ratio = 0.1  # ball mass / robot ass
                
                j = -(1 + 0.9) * vn  # 0.9 is coefficient of restitution for ball probably. I honestly just guessed lol
                j /= 1 + mass_ratio
                
                # Apply impulse to ball only (robot is much thiccer)
                self.ball.vx += j * nx
                self.ball.vy += j * ny
                
                self.last_ball_touch_time = self.game_time
    
    def check_penalty_violations(self):
        for robot in self.robots:
            if robot.penalty_time > 0:
                continue
                
            if robot.is_in_penalty_area(self.field):
                robot.penalty_time = PENALTY_TIME
                
                if robot.has_ball:
                    robot.has_ball = False
                    if robot.team == TEAM_A:
                        self.ball.vy = 5 
                    else:
                        self.ball.vy = -5 
    
    def check_idle_time(self):
        if self.game_time - self.last_ball_touch_time > IDLE_TIME_LIMIT:
            print("Ball stuck for too long, resetting to kickoff")
            self.reset_for_kickoff()
    
    def update(self, dt):
        if not self.running or self.game_over:
            return
        
        self.game_time += dt
        
        if self.game_time >= self.max_game_time:
            self.game_over = True
            if self.scores[TEAM_A] > self.scores[TEAM_B]:
                self.winner = TEAM_A
            elif self.scores[TEAM_B] > self.scores[TEAM_A]:
                self.winner = TEAM_B
            else:
                self.winner = None
            return
        
        game_state = self.get_game_state()
        
        for robot in self.robots:
            old_has_ball = robot.has_ball
            robot.update(dt, game_state)
            
            if old_has_ball and robot.has_ball:
                angle = robot.angle
                self.ball.x = robot.x + (robot.radius + self.ball.radius) * math.cos(angle)
                self.ball.y = robot.y + (robot.radius + self.ball.radius) * math.sin(angle)
                self.ball.vx = robot.vx
                self.ball.vy = robot.vy
        
        self.check_penalty_violations()
        
        self.check_robot_collisions()
        
        if not any(robot.has_ball for robot in self.robots):
            self.check_ball_robot_collisions()
            
            self.check_ball_capture()
        
        if not any(robot.has_ball for robot in self.robots):
            self.ball.update(dt)
        
        scoring_team = self.field.is_ball_in_goal(self.ball)
        if scoring_team is not None:
            self.scores[scoring_team] += 1
            print(f"GOAL! Team {scoring_team + 1} scores! Score: {self.scores[TEAM_A]}-{self.scores[TEAM_B]}")
            
            self.reset_for_kickoff(scoring_team)
        
        self.check_idle_time()
    
    def get_game_state(self):
        state = {
            'field': {
                'width': FIELD_WIDTH,
                'height': FIELD_HEIGHT,
                'total_width': TOTAL_WIDTH,
                'total_height': TOTAL_HEIGHT,
                'outer_width': OUTER_AREA_WIDTH,
                'goal_width': GOAL_WIDTH,
                'penalty_radius': PENALTY_SEMICIRCLE_RADIUS
            },
            'ball': {
                'x': self.ball.x / SCALE_FACTOR,  # Pix to cm
                'y': self.ball.y / SCALE_FACTOR,
                'vx': self.ball.vx / SCALE_FACTOR,
                'vy': self.ball.vy / SCALE_FACTOR,
                'radius': BALL_RADIUS
            },
            'robots': [],
            'scores': self.scores.copy(),
            'game_time': self.game_time,
            'last_ball_touch_time': self.last_ball_touch_time,
            'time_since_last_touch': self.game_time - self.last_ball_touch_time
        }
        
        for robot in self.robots:
            robot_data = {
                'id': robot.id,
                'team': robot.team,
                'x': robot.x / SCALE_FACTOR,
                'y': robot.y / SCALE_FACTOR,
                'vx': robot.vx / SCALE_FACTOR,
                'vy': robot.vy / SCALE_FACTOR,
                'angle': robot.angle,
                'radius': ROBOT_RADIUS,
                'penalty_time': robot.penalty_time,
                'has_ball': robot.has_ball
            }
            state['robots'].append(robot_data)
        
        return state
    
    def draw(self, screen):
        self.field.draw(screen)
        
        pygame.draw.circle(screen, ORANGE, (int(self.ball.x), int(self.ball.y)), self.ball.radius)
        
        for robot in self.robots:
            if robot.penalty_time > 0:
                continue
                
            color = BLUE if robot.team == TEAM_A else RED
            pygame.draw.circle(screen, color, (int(robot.x), int(robot.y)), robot.radius)
            
            end_x = robot.x + robot.radius * math.cos(robot.angle)
            end_y = robot.y + robot.radius * math.sin(robot.angle)
            pygame.draw.line(screen, WHITE, (robot.x, robot.y), (end_x, end_y), 2)
            
            capture_x = robot.x + robot.radius * math.cos(robot.angle)
            capture_y = robot.y + robot.radius * math.sin(robot.angle)
            pygame.draw.circle(screen, YELLOW, (int(capture_x), int(capture_y)), 5)
            
            if robot.has_ball:
                pygame.draw.circle(screen, (255, 255, 255), 
                                  (int(robot.x), int(robot.y)), 
                                  int(robot.radius * 0.8), 2)
        
        font = pygame.font.Font(None, 36)
        score_text = f"Team A: {self.scores[TEAM_A]}  Team B: {self.scores[TEAM_B]}  Time: {int(self.game_time)}s"
        text_surface = font.render(score_text, True, WHITE)
        screen.blit(text_surface, (10, 10))
        
        if self.game_over:
            if self.winner is not None:
                message = f"Game Over! Team {'A' if self.winner == TEAM_A else 'B'} wins!"
            else:
                message = "Game Over! It's a draw!"
                
            game_over_text = font.render(message, True, WHITE)
            text_rect = game_over_text.get_rect(center=(TOTAL_WIDTH_PX/2, TOTAL_HEIGHT_PX/2))
            screen.blit(game_over_text, text_rect)

class SoccerSimulationAPI:
    
    def __init__(self, personalities_dir=None):
        pygame.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH_PX, TOTAL_HEIGHT_PX))
        pygame.display.set_caption("Robot Soccer Simulation")
        self.clock = pygame.time.Clock()
        self.game = Game(personalities_dir)
        self.rendering_enabled = True
        self.running = True
        
    def reset(self):
        self.game = Game(self.game.personalities_dir if hasattr(self.game, 'personalities_dir') else None)
        self.game.reset_for_kickoff()
        self.game.running = True
        return self.game.get_game_state()
    
    def step(self, actions):
        for i, robot in enumerate(self.game.robots):
            if robot.personality is None and i < len(actions):
                acceleration, angle_change = actions[i]
                
                acceleration = max(MIN_ACCELERATION, min(MAX_ACCELERATION, acceleration))
                
                robot.angle += angle_change
                
                ax = acceleration * math.cos(robot.angle)
                ay = acceleration * math.sin(robot.angle)
                
                robot.custom_action = (ax, ay)
        
        self.game.update(0.1)
        
        rewards = self._calculate_rewards()
        
        done = self.game.game_over
        
        state = self.game.get_game_state()
        
        info = {
            'scores': self.game.scores,
            'time': self.game.game_time
        }
        
        return state, rewards, done, info
    
    def _calculate_rewards(self):
        # Since I want t train a PPO robot, it's easy to make a func that automatically gets rewards
        rewards = [0.0] * len(self.game.robots)
        
        if self.game.scores[TEAM_A] > self._prev_scores[TEAM_A]:
            # Team A scored
            for i, robot in enumerate(self.game.robots):
                if robot.team == TEAM_A:
                    rewards[i] += 10.0
                else:
                    rewards[i] -= 5.0
        
        if self.game.scores[TEAM_B] > self._prev_scores[TEAM_B]:
            # Team B scored
            for i, robot in enumerate(self.game.robots):
                if robot.team == TEAM_B:
                    rewards[i] += 10.0
                else:
                    rewards[i] -= 5.0
        
        self._prev_scores = self.game.scores.copy()
        
        for i, robot in enumerate(self.game.robots):
            if robot.has_ball:
                rewards[i] += 0.2
        
        for i, robot in enumerate(self.game.robots):
            if robot.penalty_time > 0:
                rewards[i] -= 0.5
        
        ball_y_velocity = self.game.ball.vy / SCALE_FACTOR
        
        for i, robot in enumerate(self.game.robots):
            if robot.team == TEAM_A:
                rewards[i] += 0.01 * ball_y_velocity
            else:
                rewards[i] -= 0.01 * ball_y_velocity
        
        return rewards
    
    def render(self):
        if not self.rendering_enabled:
            return
            
        self.screen.fill(BLACK)
        self.game.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.rendering_enabled:
            pygame.quit()
    
    def toggle_rendering(self, enabled=True):
        if enabled and not self.rendering_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((TOTAL_WIDTH_PX, TOTAL_HEIGHT_PX))
            pygame.display.set_caption("Robot Soccer Simulation")
            self.rendering_enabled = True
        elif not enabled and self.rendering_enabled:
            pygame.display.quit()
            self.rendering_enabled = False
    
    def set_rendering_fps(self, fps):
        self.fps = fps
    
    def run_episode(self, max_steps=3000, custom_actions_fn=None):
        self.reset()
        self._prev_scores = self.game.scores.copy()
        
        total_rewards = [0.0] * len(self.game.robots)
        
        for step in range(max_steps):
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return {
                        'status': 'interrupted',
                        'steps': step,
                        'scores': self.game.scores,
                        'total_rewards': total_rewards
                    }
            
            if not self.running:
                break
                
            if custom_actions_fn:
                actions = custom_actions_fn(self.game.get_game_state())
            else:
                actions = []
                for robot in self.game.robots:
                    if robot.personality is None:
                        acceleration = random.uniform(MIN_ACCELERATION, MAX_ACCELERATION)
                        angle_change = random.uniform(-0.1, 0.1)
                        actions.append((acceleration, angle_change))
            
            _, rewards, done, _ = self.step(actions)
            
            for i in range(len(rewards)):
                total_rewards[i] += rewards[i]
            
            self.render()
            
            if done:
                break
        
        return {
            'status': 'completed' if self.game.game_over else 'max_steps_reached',
            'steps': min(step + 1, max_steps),
            'scores': self.game.scores,
            'winner': self.game.winner,
            'total_rewards': total_rewards,
            'game_time': self.game.game_time
        }

class DummyPersonality:
    
    def __init__(self):
        self.name = "Dummy"
    
    def get_action(self, game_state):
        acceleration = random.uniform(MIN_ACCELERATION, MAX_ACCELERATION)
        angle_change = random.uniform(-0.1, 0.1)
        return acceleration, angle_change

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Robot Soccer Simulation')
    parser.add_argument('--personalities', dest='personalities_dir', help='Directory containing personality modules')
    parser.add_argument('--headless', action='store_true', help='Run without rendering')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=3000, help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    api = SoccerSimulationAPI(args.personalities_dir)
    
    if args.headless:
        api.toggle_rendering(False)
    
    for episode in range(args.episodes):
        print(f"Running episode {episode+1}/{args.episodes}")
        result = api.run_episode(max_steps=args.max_steps)
        print(f"Episode complete: {result['status']}")
        print(f"Scores: Team A {result['scores'][TEAM_A]} - Team B {result['scores'][TEAM_B]}")
        print(f"Steps: {result['steps']}, Game time: {result['game_time']:.1f}s")
        
        if result['status'] == 'completed':
            if result['winner'] is not None:
                print(f"Winner: Team {'A' if result['winner'] == TEAM_A else 'B'}")
            else:
                print("Game ended in a draw")
    
    api.close()

if __name__ == "__main__":
    main()
