import pygame
import button
import random
import time
pygame.init()






#faster -rcnn

# import the opencv library
import cv2
import os
import six
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
num_threads = 5
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
os.environ["TF_NUM_INTEROP_THREADS"] = "5"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
import importlib
importlib.reload(vis_util)

PATH_TO_CKPT = 'inference_graph' + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('inference_graph','faster_rcnn_inception_v2_custom_dataset.pbtxt')

NUM_CLASSES = 3
frcnn_graph = tf.Graph()
with frcnn_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def get_classes_name_and_scores(
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.7): # returns bigger than 90% precision
    display_str = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            if classes[i] in six.viewkeys(category_index):
                display_str['name'] = category_index[classes[i]]['name']
                display_str['score'] = '{}%'.format(int(100 * scores[i]))

    return display_str


def PIL_to_numpy(image):
  (w, h) = image.size

  return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess):
  
    # Get handles to input and output tensors
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)
        
    if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
                                    'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict




class START_GAME:
    menu = None
    #create game window
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("GAME KÉO BÚA BAO")

    #game variables
    game_paused = False
    menu_state = "main"
    select_icon = False

    computer_select = 0
    user_select = 0
    #ROCK, PAPER OR SCISSORS
    list_select = {
        "bao" : 0 ,
        "bua" : 1,
        "keo" : 2
    }
    user_cmd = None
    who_win = None
    #define fonts
    font = pygame.font.SysFont("arialblack", 30)

    #define colours
    TEXT_COL = (255, 255, 255)

    #load button resource/images
    resume_img = pygame.image.load("resource/images/button_resume.png").convert_alpha()
    quit_img = pygame.image.load("resource/images/button_quit.png").convert_alpha()
    bao_img = pygame.image.load("resource/images/button_bao.png").convert_alpha()
    bua_img = pygame.image.load("resource/images/button_bua.png").convert_alpha()
    keo_img = pygame.image.load("resource/images/button_keo.png").convert_alpha()
    home_img = pygame.image.load("resource/images/button_home.png").convert_alpha()


    #create button instances    
    resume_button = button.Button(304, 125, resume_img, 1)
    quit_button = button.Button(336, 250, quit_img, 1)
    bao_button = button.Button(30,200,bao_img,1)
    bua_button = button.Button(300,200,bua_img,1)
    keo_button = button.Button(570,200,keo_img,1)
    home_button = button.Button(750,20,home_img,1)
    #music
    run = True
    music = pygame.mixer.music.load('resource/music/nhacnen.mp3')
    pygame.mixer.music.play(-1)


    #f-rcnn
    IMAGE_PATHS = [ os.path.join('', 'image.jpg')]



    def __init__(self) -> None:
        self.name = 'START_GAME'  
        
    def draw_text(self,text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))


    

    def run_start_game_choose_camera(self,choose_round):
        
        print(choose_round)
        user_win = 0
        computer_win = 0
        total_round = 0
        resume_whowin = 0
        while self.run:
            self.screen.fill((52, 78, 91))
            
            if self.game_paused == False:
                if self.select_icon == False:
                    print('bat cam')
                    vid = cv2.VideoCapture(0)
                    x_ke = 0
                    x_ba = 0
                    x_bu = 0
                    with frcnn_graph.as_default():
                        with tf.compat.v1.Session() as sess:
                            while True:
                                ret, frame  = vid.read()
                                cv2.imwrite(filename='image.jpg',img=frame)
                                cv2.imshow('cap',frame)
                                for image_path in self.IMAGE_PATHS:
                                    image = Image.open(image_path)
                                    image_np = PIL_to_numpy(image)

                                # image_np_expanded = np.expand_dims(image_np, axis=0)
                                
                                    output_dict = run_inference_for_single_image(image_np, sess)
                                
                                    vis_util.visualize_boxes_and_labels_on_image_array(
                                        image_np,
                                        output_dict['detection_boxes'],
                                        output_dict['detection_classes'],
                                        output_dict['detection_scores'],
                                        category_index,
                                        instance_masks=output_dict.get('detection_masks'),
                                        use_normalized_coordinates=True,
                                        line_thickness=8)

                                    plt.imshow(image_np)
                                    plt.savefig("test.jpg")
                                    img_new = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
                                    cv2.imshow('frame',cv2.resize(img_new,(1000,800)))
                                    str_obj = get_classes_name_and_scores(
                                    output_dict['detection_boxes'],
                                    output_dict['detection_classes'],
                                    output_dict['detection_scores'],
                                    category_index)
                                    
                                    print(str_obj)
                        
                                    try:
                                        if str_obj["name"] == "bao":
                                            x_ba +=1
                                            print('go bao')
                                            if x_ba >=3 :
                                                self.user_select = self.list_select["bao"]
                                                self.user_cmd = "BAN DA CHON BAO"
                                                self.select_icon = True
                                                total_round +=1
                                                resume_whowin = 1
                                                break
                                        elif str_obj["name"] == "bua":
                                            x_bu +=1
                                            print('go bua')
                                            if x_bu >=3 :
                                                self.user_select = self.list_select["bua"]
                                                self.user_cmd = "BAN DA CHON BUA"
                                                self.select_icon = True
                                                total_round +=1
                                                resume_whowin = 1
                                                break
                                        else:
                                            x_ke +=1
                                            print('go keo ')
                                            print(x_ke)
                                            if x_ke >=3:
                                                print('x_ke >=3')
                                                self.user_select = self.list_select["keo"]
                                                self.user_cmd = "BAN DA CHON KEO"
                                                self.select_icon = True
                                                total_round +=1
                                                resume_whowin = 1
                                                break
                                    except Exception as e:
                                        pass
                                # cv2.imshow('image', cv2.resize(image_np,(1000,800)))
                                if self.select_icon == True:
                                    break
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                    # After the loop release the cap object
                    vid.release()
                    # Destroy all the windows
                    cv2.destroyAllWindows()
                   
                    
                else:
                    if self.who_win == None:
                        

                        
                        self.draw_text(self.user_cmd + ", VUI LONG CHO MAY CHON", self.font, self.TEXT_COL, 20, 200)
                        pygame.display.update()
                        time.sleep(1)
                        self.computer_select = random.randint(0,2)
                        if self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "computer"
                        elif self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "computer"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "computer"
                    else:
                        if self.computer_select == self.list_select["bao"]:
                            button.Button(550,350,self.bao_img,1).draw(self.screen)
                        elif self.computer_select == self.list_select["bua"]:
                            button.Button(550,350,self.bua_img,1).draw(self.screen)
                        elif self.computer_select == self.list_select["keo"]:
                            button.Button(550,350,self.keo_img,1).draw(self.screen)
                        
                        if self.user_select == self.list_select["bao"] :
                            button.Button(50,350,self.bao_img,1).draw(self.screen)
                        elif self.user_select == self.list_select["bua"]:
                            button.Button(50,350,self.bua_img,1).draw(self.screen)
                        elif self.user_select == self.list_select["keo"]:
                            button.Button(50,350,self.keo_img,1).draw(self.screen)

                        
                        if total_round <= choose_round and resume_whowin == 1:
                            if self.who_win == "hoa":
                                # self.user_cmd = "BAN VA MAY DA HOA"
                                print('hoa')
                            elif self.who_win == "computer":
                                # self.user_cmd = "MAY DA THANG BAN"
                                print('computer win')
                                computer_win +=1
                            elif self.who_win == "user" :
                                # self.user_cmd = "BAN DA THANG MAY"
                                print('user win')
                                user_win +=1
                            resume_whowin = 0
                            
                            if total_round == choose_round :
                                total_round = choose_round + 1
                        elif total_round > choose_round:
                            if computer_win > user_win :
                                self.user_cmd = "COMPUTER WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 270, 50)
                            elif user_win > computer_win :
                                self.user_cmd = "USER WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 300, 50)
                            else:
                                self.user_cmd = "NO WHO WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 300, 50)

                        if total_round < choose_round:
                            self.user_cmd = "ROUND - " +str(total_round)
                            self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 320, 50)
                        # self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 230, 50)
                        self.user_cmd = "VS"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 380, 400)
                        self.user_cmd = "USER"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 100, 200)
                        self.user_cmd = "COMPUTER"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 550, 200)
                        self.user_cmd = str(user_win)
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 130, 250)
                        self.user_cmd = str(computer_win)
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 640, 250)



                        if self.resume_button.draw(self.screen):
                            self.game_paused = False
                            self.select_icon = False
                            self.user_cmd = None
                            self.who_win = None
                            if total_round > choose_round :
                                total_round = 0
                                computer_win =0
                                user_win = 0
                        if self.quit_button.draw(self.screen):
                            self.run = False
                            self.game_paused = False
                            self.select_icon = False
                            self.user_cmd = None
                            self.who_win = None

            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    self.run = False
                    self.game_paused = False
                    self.select_icon = False
                    self.user_cmd = None
                    self.who_win = None

            pygame.display.update()


    def run_start_game_choose_hinhanh(self,choose_round):
        

        
        
        print(choose_round)
        user_win = 0
        computer_win = 0
        total_round = 0
        resume_whowin = 0
        while self.run:
            self.screen.fill((52, 78, 91))
            
            if self.game_paused == False:
                if self.select_icon == False:
                    self.draw_text(str(choose_round) + " ROUND", self.font, self.TEXT_COL, 330, 80)
                    self.draw_text("BAN HAY CHON KEO, BUA HOAC BAO", self.font, self.TEXT_COL, 80, 500)
                    if self.home_button.draw(self.screen):
                        self.run = False
                        self.game_paused = False
                        self.select_icon = False
                        self.user_cmd = None
                        self.who_win = None
                        
                    if self.bao_button.draw(self.screen):
                        self.user_select = self.list_select["bao"]
                        self.user_cmd = "BAN DA CHON BAO"
                        self.select_icon = True
                        total_round +=1
                        resume_whowin = 1

                    if self.bua_button.draw(self.screen):
                        self.user_select = self.list_select["bua"]
                        self.user_cmd = "BAN DA CHON BUA"
                        self.select_icon = True
                        total_round +=1
                        resume_whowin = 1

                    if self.keo_button.draw(self.screen):
                        self.user_select = self.list_select["keo"]
                        self.user_cmd = "BAN DA CHON KEO"
                        self.select_icon = True
                        total_round +=1
                        resume_whowin = 1
                    
                else:
                    if self.who_win == None:
                        

                        
                        self.draw_text(self.user_cmd + ", VUI LONG CHO MAY CHON", self.font, self.TEXT_COL, 20, 200)
                        pygame.display.update()
                        time.sleep(1)
                        self.computer_select = random.randint(0,2)
                        if self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "computer"
                        elif self.user_select == self.list_select["bao"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "computer"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["bua"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["bao"]:
                            self.who_win = "user"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["keo"]:
                            self.who_win = "hoa"
                        elif self.user_select == self.list_select["keo"] and self.computer_select == self.list_select["bua"]:
                            self.who_win = "computer"
                    else:
                        if self.computer_select == self.list_select["bao"]:
                            button.Button(550,350,self.bao_img,1).draw(self.screen)
                        elif self.computer_select == self.list_select["bua"]:
                            button.Button(550,350,self.bua_img,1).draw(self.screen)
                        elif self.computer_select == self.list_select["keo"]:
                            button.Button(550,350,self.keo_img,1).draw(self.screen)
                        
                        if self.user_select == self.list_select["bao"] :
                            button.Button(50,350,self.bao_img,1).draw(self.screen)
                        elif self.user_select == self.list_select["bua"]:
                            button.Button(50,350,self.bua_img,1).draw(self.screen)
                        elif self.user_select == self.list_select["keo"]:
                            button.Button(50,350,self.keo_img,1).draw(self.screen)

                        
                        if total_round <= choose_round and resume_whowin == 1:
                            if self.who_win == "hoa":
                                # self.user_cmd = "BAN VA MAY DA HOA"
                                print('hoa')
                            elif self.who_win == "computer":
                                # self.user_cmd = "MAY DA THANG BAN"
                                print('computer win')
                                computer_win +=1
                            elif self.who_win == "user" :
                                # self.user_cmd = "BAN DA THANG MAY"
                                print('user win')
                                user_win +=1
                            resume_whowin = 0
                            
                            if total_round == choose_round :
                                total_round = choose_round + 1
                        elif total_round > choose_round:
                            if computer_win > user_win :
                                self.user_cmd = "COMPUTER WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 270, 50)
                            elif user_win > computer_win :
                                self.user_cmd = "USER WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 300, 50)
                            else:
                                self.user_cmd = "NO WHO WIN"
                                self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 300, 50)

                        if total_round < choose_round:
                            self.user_cmd = "ROUND - " +str(total_round)
                            self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 320, 50)
                        # self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 230, 50)
                        self.user_cmd = "VS"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 380, 400)
                        self.user_cmd = "USER"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 100, 200)
                        self.user_cmd = "COMPUTER"
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 550, 200)
                        self.user_cmd = str(user_win)
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 130, 250)
                        self.user_cmd = str(computer_win)
                        self.draw_text(self.user_cmd, self.font, self.TEXT_COL, 640, 250)



                        if self.resume_button.draw(self.screen):
                            self.game_paused = False
                            self.select_icon = False
                            self.user_cmd = None
                            self.who_win = None
                            if total_round > choose_round :
                                total_round = 0
                                computer_win =0
                                user_win = 0
                        if self.quit_button.draw(self.screen):
                            self.run = False
                            self.game_paused = False
                            self.select_icon = False
                            self.user_cmd = None
                            self.who_win = None

            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    self.run = False
                    self.game_paused = False
                    self.select_icon = False
                    self.user_cmd = None
                    self.who_win = None

            pygame.display.update()

        # pygame.quit()




