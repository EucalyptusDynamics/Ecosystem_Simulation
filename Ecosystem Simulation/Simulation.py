import pygame
import random
import numpy as np
import NNwithGA


#display background image function

def displayenv(fishery):
    screen.blit(fishery, (posx - int(fishery.get_width() / 2), posy - int(fishery.get_height() / 2)))

#the fish class

class Fish():

    def __init__(self, position):
        self.position = position
        self.player = pygame.image.load('Fish.png')
        self.fish = pygame.transform.scale(self.player, (50, 80))
        self.rotz = 0
        self.fishy = pygame.transform.rotate(self.fish, self.rotz)
        self.foodtargetdistance = 1000
        self.foodangle = 0
        self.angle = 0
        self.output_speed = 0

    def displayfish(self):
        self.blitted = screen.blit(self.fishy, (self.position[0] - int(self.fishy.get_width() / 2), self.position[1] - int(self.fishy.get_height() / 2)))
        self.rotate_fish()
        self.go_forward()

    def rotate_fish(self):
        self.fishy = pygame.transform.rotate(self.fish, self.rotz)

    def go_forward(self):
        self.position[0] += np.cos(np.deg2rad(self.rotz + 90)) * 0.7 * self.output_speed
        self.position[1] += - np.sin(np.deg2rad(self.rotz + 90)) * 0.7 * self.output_speed

    def set_angle(self,angle):
        self.rotz = angle

    def get_angle(self):
        return self.angle


#The Food class

class Food():

    def __init__(self, position):
        self.position = position
        self.food = pygame.image.load('Food.png')
        self.food2 = pygame.transform.scale(self.food, (50, 80))
        self.rotz = 0
        self.food3 = pygame.transform.rotate(self.food2, 0)

        #bounds
        self.offsetup = 50
        self.foodupperbound = self.position[1] + self.offsetup
        self.offsetdown = -50
        self.foodlowerbound = self.position[1] + self.offsetdown
        self.offsetleft = -50
        self.foodleftbound = self.position[0] + self.offsetleft
        self.offsetright = 50
        self.foodrightbound = self.position[0] + self.offsetright

    def displayfood(self):
        screen.blit(self.food3, (self.position[0], self.position[1]))

        
def Out_of_bounds():
    for i in range(len(fishes)):
        if fishes[i].position[1] <= 0:
            fishes[i].position[1] = 2
        if fishes[i].position[1] >= 900:
            fishes[i].position[1] = 898

        if fishes[i].position[0] <= 0:
            fishes[i].position[0] = 2
        if fishes[i].position[0] >= 1800:
            fishes[i].position[0] = 1798
            

#initializing pygame

pygame.init()

X = 1800
Y = 900
rotz = 0
posx = 500
posy = 250


screen = pygame.display.set_mode((X, Y))


pygame.display.set_caption('Ecosystem Simulation 1')

coralreef = pygame.image.load('coralreef.jpg')

pygame.display.flip()



#create instances of fish

fishes = []

for i in range(30):
    fishes.append(Fish([random.randint(0, 1800), random.randint(0, 900)]))



#create instances of food

food = []

for i in range(20):
    food.append(Food([random.randint(0, 1800), random.randint(0, 900)]))





trash = []

for i in fishes:
    i.set_angle(random.randint(-100,100))

frame = 0
frame2 = 0

#While loop to keep the simulation running

is_running = True
while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False


    #control fish for testing purposes

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]: 
        fishes[0].rotate_fishRIGHT()
    if keys[pygame.K_RIGHT]:
        fishes[0].rotate_fishLEFT()
    if keys[pygame.K_UP]:
        fishes[0].position[0] += np.cos(np.deg2rad(fishes[0].rotz + 90)) * 0.7
        fishes[0].position[1] += - np.sin(np.deg2rad(fishes[0].rotz + 90)) * 0.7


    #display the background image of the fishes environment
    
    displayenv(coralreef)  

    #display the fish on screen

    Out_of_bounds()

    for i in fishes:
        i.displayfish()

    #display food on screen

    for i in food:
        i.displayfood()


    #load more food

    if frame2 == 250:
        for i in range(5):
            food.append(Food([random.randint(0, 1800), random.randint(0, 900)]))
        frame2 = 0

    
    if frame == 5:

        #calculating the distance and angle of the food from each fish for inputs to the neural network

        for i in range(len(fishes)):
            for j in range(len(food)):
                fishes[i].foodtargetdistance = np.sqrt(np.square(fishes[i].position[0] - food[j].position[0]) + np.square(fishes[i].position[1] - food[j].position[1]))
                fishes[i].foodangle = np.arctan((fishes[i].position[0] - food[j].position[0]) / (fishes[i].position[1] - food[j].position[1])) - fishes[i].rotz
                
                NNwithGA.input_angle = fishes[i].foodangle
                NNwithGA.input_distance = fishes[i].foodtargetdistance


                #Neural Network Layers with input and output called each frame below
            
                NNwithGA.X = [[NNwithGA.input_angle, NNwithGA.input_distance]] #inputs

                NNwithGA.layer1.forward(NNwithGA.X)

                NNwithGA.activation1.forward(NNwithGA.layer1.output)

                NNwithGA.layer2.forward(NNwithGA.activation1.output)

                NNwithGA.activation2.forward(NNwithGA.layer2.output)

                NNwithGA.output_rotz = NNwithGA.activation2.output[0][0] - 0.5    #output 1
                NNwithGA.output_speed = NNwithGA.activation2.output[0][1]

                fishes[i].output_speed = NNwithGA.output_speed
                fishes[i].rotz += NNwithGA.output_rotz * 4



            #seeing if the fish is where the food is

        for i in fishes:
            for k in range(len(food)):
                if i.position[1] <= food[k].foodupperbound and i.position[1] >= food[k].foodlowerbound and i.position[0] >= food[k].foodleftbound and i.position[0] <= food[k].foodrightbound:
                        trash.append(k)

        frame = 0

    #remove food when eating

    for i in sorted(trash, reverse=False):
        try:
            food.pop(i)
        except:
            print('food not destroyed')
    
    trash = []               
           
    frame += 1
    frame2 += 1
    

    #Updates frames in the While loop
    
    pygame.display.update()

pygame.quit()
