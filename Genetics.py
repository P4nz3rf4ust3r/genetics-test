import pygame
import random
import pygame.gfxdraw
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import os.path

pygame.init()

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (34, 139, 34)
grey = (79, 79, 69)

screen_width = 1400
screen_height = 800
screen = pygame.display.set_mode([screen_width, screen_height])
pygame.display.set_caption('Genetic Evolution')


def text_objects(text, color, size):
    font = pygame.font.SysFont("Britannica", size)
    text_surf = font.render(text, True, color)
    return text_surf, text_surf.get_rect()


def message_to_screen(text, color, y_displace=0, size=50):
    text_surf, text_rect = text_objects(text, color, size)
    text_rect.center = screen_width/2, screen_height/2 + y_displace
    screen.blit(text_surf, text_rect)


learning_rate = 2e-3
score_requirement = 400
generations = 10
score_goal = 2000
high_reward = 10
medium_reward = 5
low_reward = 1


class Creature(pygame.sprite.Sprite):
    def __init__(self, color, radius, health):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([radius*2, radius*2], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.center_x = self.rect.centerx
        self.center_y = self.rect.centery
        pygame.gfxdraw.filled_circle(self.image, self.center_x, self.center_y, radius, color)
        self.health = health
        self.birth_time = pygame.time.get_ticks()
        self.score = 0
        self.memory = []
        self.observation = []
        self.requirement_complete = False
        self.goal_complete = False
        self.is_trained = False
        self.prev_observation = []


class DeadCreature(pygame.sprite.Sprite):
    def __init__(self, color, radius):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([radius * 2, radius * 2], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.center_x = self.rect.centerx
        self.center_y = self.rect.centery
        pygame.gfxdraw.filled_circle(self.image, self.center_x, self.center_y, radius, color)


class Food(pygame.sprite.Sprite):
    def __init__(self, color, side, expiry):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([side, side])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.expiry = expiry
        self.spawn_time = pygame.time.get_ticks()


creature_list = pygame.sprite.Group()
food_list = pygame.sprite.Group()
sprite_list = pygame.sprite.Group()
corpse_list = pygame.sprite.Group()


def randomize_creatures(population, health, trained_status=False):
    for i in range(population):
        creature = Creature(black, 15, health)
        creature.rect.x = random.randrange(screen_width)
        creature.rect.y = random.randrange(screen_height)
        creature.time = pygame.time.get_ticks()
        if trained_status:
            creature.is_trained = True
        creature_list.add(creature)
        sprite_list.add(creature)


def dead_creature(x, y):
    corpse = DeadCreature(grey, 15)
    corpse.rect.x = x
    corpse.rect.y = y
    sprite_list.add(corpse)
    corpse_list.add(corpse)


def randomize_food(population, expiry):
    for i in range(population):
        food = Food(green, 5, expiry)
        free_location = False
        while not free_location:
            free_location = True
            food.rect.x = random.randrange(screen_width)
            food.rect.y = random.randrange(screen_height)
            for creature in creature_list:
                distance = np.sqrt(creature.rect.x**2+creature.rect.y**2)
                if distance < 22:
                    free_location = False
        food_list.add(food)
        sprite_list.add(food)


def spawn_food(expiry, spawn_chance):
    if random.random() < spawn_chance:
        randomize_food(1, expiry)


def clear_setup():
    creature_list.empty()
    food_list.empty()
    sprite_list.empty()
    corpse_list.empty()


update_clock = pygame.time.Clock()
FPS = 20

POPULATION = 20
INITIAL_FOOD = 20
MAX_HEALTH = 8000
SPEED = 3
HUNGER_RESTORE = 5000
NUMBER_OF_ACTIONS = 9


# def observations(creature, action):
#     observed_x = creature.rect.x
#     observed_y = creature.rect.y
#     distance_list = []
#     food_x_list = []
#     food_y_list = []
#     food_spawn_list = []
#     for food in food_list:
#         food_x = food.rect.x
#         food_y = food.rect.y
#         distance = np.sqrt((observed_x-food_x)**2+(observed_y-food_y)**2)
#         food_spawn_time = food.spawn_time
#         food_x_list.append(food_x)
#         food_y_list.append(food_y)
#         distance_list.append(distance)
#         food_spawn_list.append(food_spawn_time)
#
#     food_data = sorted(zip(distance_list, food_x_list, food_y_list, food_spawn_list))
#     sorted_distance = tuple(x[0] for x in food_data)
#     sorted_food_x = tuple(x[1] for x in food_data)
#     sorted_food_y = tuple(x[2] for x in food_data)
#     sorted_food_spawn = tuple(x[3] for x in food_data)
#     if len(sorted_distance) >= 3:
#         tuple_size = 3
#     else:
#         tuple_size = len(sorted_distance)  # TODO: MAKE SO IF TUPLE SIZE < 3, TRAINING DATA IS NOT REDUCED
#
#     sorted_distance = sorted_distance[0:tuple_size]
#     sorted_food_x = sorted_food_x[0:tuple_size]
#     sorted_food_y = sorted_food_y[0:tuple_size]
#     sorted_food_spawn = sorted_food_spawn[0:tuple_size]
#     sorted_food_lifetime = []
#     for spawn_time in sorted_food_spawn:
#         food_lifetime = pygame.time.get_ticks() - spawn_time
#         sorted_food_lifetime.append(food_lifetime)
#
#     reward = 0
#     for i in range(0, tuple_size):
#         if sorted_food_x[i] > observed_x and sorted_food_y[i] >= observed_y and action == 2:
#             reward += high_reward
#         elif sorted_food_x[i] >= observed_x and sorted_food_y[i] < observed_y and action == 4:
#             reward += high_reward
#         elif sorted_food_x[i] < observed_x and sorted_food_y[i] >= observed_y and action == 9:
#             reward += high_reward
#         elif sorted_food_x[i] <= observed_x and sorted_food_y[i] < observed_y and action == 6:
#             reward += high_reward
#
#     print(reward)
#     data_list = [[reward, observed_x, observed_y], sorted_distance, sorted_food_x, sorted_food_y, sorted_food_lifetime]
#     flat_list = [item for sublist in data_list for item in sublist]
#     return flat_list


def observations(creature, action):
    observed_x = creature.rect.x
    observed_y = creature.rect.y
    food_x_list = []
    food_y_list = []
    food_spawn_list = []
    distance_list = []
    for food in food_list:
        food_x = food.rect.x
        food_y = food.rect.y
        distance = np.sqrt((observed_x - food_x) ** 2 + (observed_y - food_y) ** 2)
        food_spawn_time = food.spawn_time
        food_x_list.append(food_x)
        food_y_list.append(food_y)
        distance_list.append(distance)
        food_spawn_list.append(food_spawn_time)

    food_data = sorted(zip(distance_list, food_x_list, food_y_list, food_spawn_list))
    sorted_food_x = tuple(x[1] for x in food_data)
    sorted_food_y = tuple(x[2] for x in food_data)
    sorted_distance = tuple(x[0] for x in food_data)
    sorted_food_spawn = tuple(x[3] for x in food_data)
    #if len(sorted_food_x) >= 3:
    #    tuple_size = 3
    #else:
    #    tuple_size = len(sorted_food_x)  # TODO: MAKE SO IF TUPLE SIZE < 3, TRAINING DATA IS NOT REDUCED

    #sorted_food_x = sorted_food_x[0:tuple_size]
    #sorted_food_y = sorted_food_y[0:tuple_size]
    #sorted_food_spawn = sorted_food_spawn[0:tuple_size]
    sorted_food_lifetime = []
    for spawn_time in sorted_food_spawn:
        food_lifetime = pygame.time.get_ticks() - spawn_time
        sorted_food_lifetime.append(food_lifetime)

    reward = 0
    for i in range(0, len(sorted_food_x)):
        if sorted_food_x[i] > observed_x and sorted_food_y[i] >= observed_y and action == 2:
            reward += high_reward
            if sorted_distance[i] < 100:
                reward += int(np.ceil(100-sorted_distance[i])*low_reward)
        elif sorted_food_x[i] >= observed_x and sorted_food_y[i] < observed_y and action == 4:
            reward += high_reward
            if sorted_distance[i] < 100:
                reward += int(np.ceil(100-sorted_distance[i])*low_reward)
        elif sorted_food_x[i] < observed_x and sorted_food_y[i] >= observed_y and action == 9:
            reward += high_reward
            if sorted_distance[i] < 100:
                reward += int(np.ceil(100-sorted_distance[i])*low_reward)
        elif sorted_food_x[i] <= observed_x and sorted_food_y[i] < observed_y and action == 6:
            reward += high_reward
            if sorted_distance[i] < 100:
                reward += int(np.ceil(100-sorted_distance[i])*low_reward)

    print(reward)
    data_list = [[reward, observed_x, observed_y], sorted_distance, sorted_food_x, sorted_food_y, sorted_food_lifetime]
    flat_list = [item for sublist in data_list for item in sublist]
    return flat_list


def movement(creature, action, speed=SPEED):
    def stay():
        pass

    def up():
        creature.rect.y += speed

    def up_right():
        creature.rect.x += speed
        creature.rect.y += speed

    def right():
        creature.rect.x += speed

    def down_right():
        creature.rect.x += speed
        creature.rect.y -= speed

    def down():
        creature.rect.y -= speed

    def down_left():
        creature.rect.x -= speed
        creature.rect.y -= speed

    def left():
        creature.rect.x -= speed

    def up_left():
        creature.rect.x -= speed
        creature.rect.y += speed

    movement_choices = {
        0: stay,
        1: up,
        2: up_right,
        3: right,
        4: down_right,
        5: down,
        6: down_left,
        7: left,
        8: up_left
    }

    movement_choices[action]()


def neural_network_model(input_size, output_size):
    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    learning_data = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    labels_list = np.asarray([i[1] for i in training_data])
    labels = np.zeros((len(labels_list), NUMBER_OF_ACTIONS))
    for i in range(len(labels_list)):
        temp = labels_list[i]
        labels[i][temp] = 1

    if not model:
        model = neural_network_model(input_size=len(learning_data[0]), output_size=9)

    model.fit(learning_data, labels, n_epoch=10, show_metric=True, batch_size=1)

    return model


def app_intro():
    intro = True
    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    intro = False
                    randomize_creatures(POPULATION, MAX_HEALTH)
                    randomize_food(INITIAL_FOOD, 5000)
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_t:  # TODO: make sure that training data exists
                    training_data = np.load('TrainingData.npy')
                    model = train_model(training_data)
                    model.save('TrainedCell')
                if event.key == pygame.K_p:
                    if os.path.isfile('TrainedCell.meta'):
                        intro = False
                        randomize_creatures(POPULATION, MAX_HEALTH, True)
                        randomize_food(INITIAL_FOOD, 5000)
                    else:
                        print('Model file does not exist.')  # TODO: say it on screen

        screen.fill(white)
        message_to_screen("Welcome to Genetic Evolution", black, -100)
        message_to_screen("ENTER: start     T: train     ESC: quit     P: predict", black, 50, 30)
        update_clock.tick(FPS)
        pygame.display.flip()


def app_breeding():
    breeding = True
    while breeding:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    breeding = False
                    randomize_creatures(POPULATION, MAX_HEALTH)
                    randomize_food(INITIAL_FOOD, 5000)
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_t:  # TODO: make sure that training data exists
                    training_data = np.load('TrainingData.npy')
                    model = train_model(training_data)
                    model.save('TrainedCell')
                if event.key == pygame.K_p:
                    if os.path.isfile('TrainedCell.meta'):
                        breeding = False
                        randomize_creatures(POPULATION, MAX_HEALTH, True)
                        randomize_food(INITIAL_FOOD, 5000)
                    else:
                        print('Model file does not exist.')  # TODO: say it on screen

        screen.fill(white)
        message_to_screen("Ready to train", black, -100)
        message_to_screen("ENTER: start     T: train     ESC: quit     P: predict", black, 50, 30)
        update_clock.tick(FPS)
        pygame.display.flip()


def app_loop():
    training_data = []
    scores = []
    accepted_scores = []
    if os.path.isfile('TrainedCell.meta'):
        model = neural_network_model(14, NUMBER_OF_ACTIONS)
        model.load('TrainedCell')

    app_exited = False
    while not app_exited:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app_exited = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    clear_setup()
                    training_data = []
                    scores = []
                    accepted_scores = []
                    randomize_creatures(POPULATION, MAX_HEALTH)
                    randomize_food(INITIAL_FOOD, 5000)

        screen.fill(white)
        message_to_screen("R: restart", black, 300, 20)

        spawn_food(5000, 0.4)

        for food in food_list:
            elapsed = pygame.time.get_ticks() - food.spawn_time
            if elapsed >= food.expiry:
                food.kill()

        for creature in creature_list:
            if creature.is_trained:
                if creature.prev_observation:
                    observed_data = np.array(creature.prev_observation)
                    action = np.argmax(model.predict(observed_data.reshape(-1, len(observed_data))))
                else:
                    action = random.randrange(0, NUMBER_OF_ACTIONS, 1)
            else:
                action = random.randrange(0, NUMBER_OF_ACTIONS, 1)
            movement(creature, action, SPEED)
            creature.observation = observations(creature, action)
            reward = creature.observation.pop(0)
            creature.prev_observation = creature.observation
            if creature.score >= score_goal:
                creature.lifetime_goal_complete = True
            creature.memory.append([creature.observation, action, reward])
            creature.score += reward

        for creature in creature_list:
            elapsed = pygame.time.get_ticks() - creature.birth_time
            scores.append(creature.score)
            if elapsed >= creature.health:
                dead_creature(creature.rect.x, creature.rect.y)
                if creature.score >= score_requirement:
                    accepted_scores.append(creature.score)
                    for data in creature.memory:
                        if data[2] >= 10:
                            training_data.append([data[0], data[1]])
                creature.kill()

            collision_list = pygame.sprite.groupcollide(creature_list, food_list, False, True)
            for listed_creature in collision_list:
                listed_creature.health += HUNGER_RESTORE

        sprite_list.draw(screen)
        update_clock.tick(FPS)
        pygame.display.flip()

        if not creature_list:
            training_data_save = np.array(training_data)
            np.save('TrainingData', training_data_save)
            if accepted_scores:
                print('Average accepted score:', mean(accepted_scores))
                print('Median accepted score:', median(accepted_scores))
            else:
                print('Average accepted score: 0')
                print('Median accepted score: 0')
            print(Counter(accepted_scores))

            clear_setup()
            app_breeding()
            training_data = []
            scores = []
            accepted_scores = []

    pygame.quit()
    quit()


app_intro()
app_loop()
