############################
### AUXILLIARY FUNCTIONS ###
############################

# Import libraries
import numpy as np
import time
import cv2
import os
import platform
import psutil
import pandas as pd
import datetime
from statistics import mean
import tensorflow as tf

# Test to see if the GPU is being used
if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:
   print("Please install GPU version of TF")

# Load Models
def load_models(net_names) :
    print("[INFO]: Loading Models...")
    nets = {}
    for num, net in enumerate(net_names) :
        nets[net] = tf.lite.Interpreter(model_path='Models_TF/lite-model_{}.tflite'.format(net), num_threads=4)
        nets[net].allocate_tensors()
        # model = "/home/pi/Research/Models/" + net_names[i] + "/frozen_inference_graph.pb"
        # config = "/home/pi/Research/Models/" + net_names[i] + "/config.pbtxt"
        # nets['{}'.format(net_names[i])] = cv2.dnn.readNetFromTensorflow(model=model,config=config)
        print('[INFO]: {} loaded.'.format(net))

    return nets

# Set up GPIO pins
if platform.node() != 'michael-desktop' :
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(5, GPIO.OUT)

# Measure CPU temperature
def measure_temp_CPU():

    if platform.node() == 'michael-desktop' :
        # max_cpu_temp = 0
        # for i in range(6) :
        #     tempFile = open( "/sys/class/thermal/thermal_zone" + str(i) + "/temp")
        #     cpu_temp = tempFile.read()
        #     tempFile.close()
        #     cpu_temp = float(cpu_temp)/1000
        #     if cpu_temp > max_cpu_temp : max_cpu_temp = cpu_temp
        # cpu_temp = max_cpu_temp

        tempFile = open( "/sys/class/thermal/thermal_zone1/temp")
        cpu_temp = tempFile.read()
        tempFile.close()
        cpu_temp = float(cpu_temp)/1000

    else: 
        tempFile = open( "/sys/class/thermal/thermal_zone0/temp")
        cpu_temp = tempFile.read()
        tempFile.close()
        cpu_temp = float(cpu_temp)/1000

    return cpu_temp

# Measure GPU temperature
def measure_temp_GPU():

    if platform.node() == 'michael-desktop' :
        tempFile = open( "/sys/class/thermal/thermal_zone2/temp")
        cpu_temp = tempFile.read()
        tempFile.close()
        gpu_temp = float(cpu_temp)/1000

    else: 
        temp = os.popen("vcgencmd measure_temp").readline()
        numeric_filter = filter(str.isdigit, temp)
        numeric_string = "".join(numeric_filter)

        gpu_temp = float(numeric_string)/10
    return gpu_temp

# Measure CPU usage
def measure_cpu():
    cpu_use = psutil.cpu_percent(interval=None)
    return float(cpu_use)

# Calculate moving average
def moving_average(numbers, window_size):
#     return np.convolve(numbers, np.ones(window_size)/window_size, mode='valid')
    i = 0
    moving_averages = []
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

# Find mean temperature
def find_mean_temp(window) :
    print("[INFO]: Taking {} second mean temperature...".format(window))
    test_temp_record_CPU = []
    timer_start = time.time()
    while timer_start+window > time.time():
        test_temp_record_CPU.append(measure_temp_CPU())
        time.sleep(1)
    mean_test_temp = np.mean(test_temp_record_CPU)
    print("[INFO]: Mean temperature: {} degrees Celsius.".format(mean_test_temp))
    return mean_test_temp

# Turn fan on
def fan_on() :
    GPIO.output(5, True)
    print("[INFO]: Fan on. Temperature at: {}".format(measure_temp_CPU())) 

# Turn fan off
def fan_off() :
    GPIO.output(5, False)
    print("[INFO]: Fan off. Temperature at: {}".format(measure_temp_CPU()))

# Stabilize temperature
def stabilize_temp(des_temp, nets) :
    print("[INFO]: Stabilizing temperature.")

    if platform.node() == 'michael-desktop' :
        image_files = os.listdir('/home/michael/val2017/')
        # MobileNetSSDv2_model = "Models/" + 'MobileNetSSDv2' + ".pb"
        # MobileNetSSDv2_pbtxt = "Models/" + 'MobileNetSSDv2' + ".pbtxt"
        # MobileNetSSDv2_net = cv2.dnn.readNetFromTensorflow(MobileNetSSDv2_model,MobileNetSSDv2_pbtxt)
        if measure_temp_CPU() < des_temp :
            while measure_temp_CPU() < des_temp :
                print("[INFO]: Heating up...")

                # frame = cv2.imread('/home/michael/val2017/' + image_files[0])
                # (h, w) = frame.shape[:2]
                # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), size=(300, 300), swapRB=True, crop=False)
                # MobileNetSSDv2_net.setInput(blob)
                # detections = MobileNetSSDv2_net.forward()

                process_image_tf('nano', image_files, [0], ['efficientdet_lite4_detection_metadata_2'], nets)

        if measure_temp_CPU() > des_temp :
            while measure_temp_CPU() > des_temp :
                print("[INFO]: Cooling down...")
                time.sleep(1)

    else :
        print('[INFO]: Cooling down...')
        fan_on()
        while measure_temp_CPU() > des_temp :
            time.sleep(1)
        fan_off()

    print('[INFO]: Temperature stabilized at {}.'.format(measure_temp_CPU()))

# Record data first time
def record_data_1(device, iterator_record, time_record, 
                processing_duration, temp_record_CPU, temp_record_GPU, 
                processing_duration_record, loop_duration_record, cpu_record,
                pause_duration_record, avg_accuracy_record, action_record) : 

    CPU_temp = measure_temp_CPU()
    GPU_temp = measure_temp_GPU()
    CPU_use = measure_cpu()

    iterator_record.append(iterator_record[-1])
    time_record.append(time_record[-1]+processing_duration)
    temp_record_CPU.append(CPU_temp)
    temp_record_GPU.append(GPU_temp)
    cpu_record.append(CPU_use)
    processing_duration_record.append(processing_duration)
    loop_duration_record.append(loop_duration_record[-1])
    pause_duration_record.append(pause_duration_record[-1])
    avg_accuracy_record.append(avg_accuracy_record[-1])
    action_record.append(action_record[-1])

# Record data second time
def record_data_2(device, iterator_record, time_record, loop_duration, 
                processing_duration, temp_record_CPU, temp_record_GPU, 
                processing_duration_record, loop_duration_record, cpu_record,
                pause_duration_record, pause_duration, avg_accuracy_record,
                avg_accuracy, action_record, action) : 

    CPU_temp = measure_temp_CPU()
    GPU_temp = measure_temp_GPU()
    CPU_use = measure_cpu()

    # update iterator record
    iterator_record.append(iterator_record[-1] + 1)
    time_record.append(time_record[-1]+loop_duration-processing_duration)
    temp_record_CPU.append(CPU_temp)
    temp_record_GPU.append(GPU_temp)
    processing_duration_record.append(processing_duration_record[-1])
    loop_duration_record.append(loop_duration)
    cpu_record.append(CPU_use)
    pause_duration_record.append(pause_duration)
    avg_accuracy_record.append(avg_accuracy)
    action_record.append(action)

# Save data
def save_data(path, device, iterator_record, time_record, temp_record_CPU,
                temp_record_GPU, cpu_record, processing_duration_record,
                pause_duration_record, loop_duration_record, 
                net_record, avg_accuracy_record, action_record) :

    if device == 'nano' :
        dict = {'iterator_record':iterator_record,
                'time_record':time_record,
                'temp_record_CPU':temp_record_CPU,
                'temp_record_GPU':temp_record_GPU,
                'cpu_record':cpu_record,
                'processing_duration_record':processing_duration_record,
                'pause_duration_record':pause_duration_record,
                'loop_duration_record':loop_duration_record,
                'net_record':net_record,
                'avg_accuracy_record':avg_accuracy_record,
                'action_record':action_record
                }

    elif device == 'rpi' :
        dict = {'iterator_record':iterator_record,
                'time_record':time_record,
                'temp_record_CPU':temp_record_CPU,
                'temp_record_GPU':temp_record_GPU,
                'cpu_record':cpu_record,
                'processing_duration_record':processing_duration_record,
                'pause_duration_record':pause_duration_record,
                'loop_duration_record':loop_duration_record,
                'net_record':net_record,
                'avg_accuracy_record':avg_accuracy_record,
                'action_record':action_record
                }

    results = pd.DataFrame(dict)
    results.to_csv(os.path.join(path, "results" + ".csv"))
    
    return results

# def image_prep(device, image_files, iterator_record) :
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels

#     if device == 'nano' : frame = cv2.imread('/home/michael/val2017/' + image_files[iterator_record[-1]])
#     elif device == 'rpi' : frame = cv2.imread('/home/pi/val2017/' + image_files[iterator_record[-1]])
#     # grab the frame dimensions and convert it to a blob
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), size=(300, 300), swapRB=True, crop=False)
#     return blob, h, w, frame

# def process_image(net_record, blob, nets, net_names) :

#     for i in range(len(net_names)) :
#         if net_record[-1] == net_names[i] :
#             nets[net_names[i]].setInput(blob)
#             detections = nets[net_names[i]].forward()

#     return detections    

# Image processing function
def process_image_tf(device, image_files, iterator_record, net_record, nets) :
    if device == 'nano' : frame = cv2.imread('/home/michael/val2017/' + image_files[10])
    elif device == 'rpi' : frame = cv2.imread('/home/pi/val2017/' + image_files[10])

    input_details = nets[net_record[-1]].get_input_details()
    input_shape = input_details[0]['shape']

    working_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    working_frame = np.asarray(working_frame)
    working_frame = np.expand_dims(working_frame,axis=0)
    nets[net_record[-1]].set_tensor(input_details[0]['index'], working_frame)
    nets[net_record[-1]].invoke()

# Calculate average network duration
def get_net_info(device, nets, image_files, net_names, start_temp) :

    net_durations = {}
    # our_coco = COCO('/home/pi/annotations/instances_val2017.json')

    for net in net_names :
        print('[INFO]: Getting info for net: {}.'.format(net))
        stabilize_temp(start_temp, nets)
        pdr = []
        # accuracy_record = []
        # bbox_record = []
        iterator_record = [0]
        while iterator_record[-1] < 5 :
            loop_start_time = datetime.datetime.now()
            process_image_tf(device, image_files, iterator_record, [net], nets)
            processing_end_time = datetime.datetime.now()
            processing_duration = (processing_end_time - loop_start_time).total_seconds()
            pdr.append(processing_duration)

        # # Confidence Display    
        #     # loop over the detections
        #     for i in np.arange(0, detections.shape[2]):
        #         # extract the confidence (i.e., probability) associated with
        #         # the prediction
        #         confidence = detections[0, 0, i, 2]

        #         # filter out weak detections by ensuring the `confidence` is
        #         # greater than the minimum confidence
        #         if confidence > 0.7:
        #             # extract the index of the class label from the
        #             # `detections`, then compute the (x, y)-coordinates of
        #             # the bounding box for the object
        #             idx = int(detections[0, 0, i, 1])
        #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #             (startX, startY, endX, endY) = box.astype("int")


        #             # draw the prediction on the frame
        #             label = "{}: {:.2f}%".format(CLASSES[idx],
        #                 confidence * 100)
        #             cv2.rectangle(frame, (startX, startY), (endX, endY),
        #                 COLORS[idx], 2)
        #             y = startY - 15 if startY - 15 > 15 else startY + 15
        #             cv2.putText(frame, label, (startX, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)    
                    
        #             bbox_record.append(int(image_files[iterator_record[-1]].strip('.jpg')))
        #             bbox_record.append(startX)
        #             bbox_record.append(startY)
        #             bbox_record.append(endX)
        #             bbox_record.append(endY)            

            # cv2.imwrite('detections_image_{}.png'.format(iterator_record[-1]), frame)
            iterator_record.append(iterator_record[-1] + 1)

    # # ACCURACY

    #     counter = 1
    #     for i in range(0,int(len(bbox_record)/5)):
    #         img_id = bbox_record[i*5]

    #         ann_ids = our_coco.getAnnIds(imgIds=img_id)
    #         anns = our_coco.loadAnns(ann_ids)
    #         anns_len = len(anns)
    #         max_iou = 0
    #         num_annotations = 0
    #         iou = 0
    #         for ann in anns:

    #             x_topLeft   = ann['bbox'][0]
    #             y_topLeft   = ann['bbox'][1]
    #             bbox_width  = ann['bbox'][2]
    #             bbox_height = ann['bbox'][3] 
                
    #             x_bottomRight = x_topLeft + bbox_width
    #             y_bottomRight = y_topLeft + bbox_height

    #             # determine the (x, y)-coordinates of the intersection rectangle
    #             xA = max(bbox_record[counter], x_topLeft)
    #             yA = max(bbox_record[counter+1], y_topLeft)
    #             xB = min(bbox_record[counter+2], x_bottomRight)
    #             yB = min(bbox_record[counter+3], y_bottomRight)
    #             # compute the area of intersection rectangle
    #             interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #             # compute the area of both the prediction and ground-truth
    #             # rectangles
    #             boxAArea = (bbox_record[counter+2] - bbox_record[counter] + 1) * (bbox_record[counter+3] - bbox_record[counter+1] + 1)
    #             boxBArea = (x_bottomRight - x_topLeft + 1) * (y_bottomRight - y_topLeft + 1)
    #             # compute the intersection over union by taking the intersection
    #             # area and dividing it by the sum of prediction + ground-truth
    #             # areas - the interesection area
    #             iou = interArea / float(boxAArea + boxBArea - interArea)            

    #             num_annotations +=1
                
    #             if iou > max_iou:
    #                 max_iou = iou
    #                 max_iou_ground_x_topLeft = int(x_topLeft)
    #                 max_iou_ground_y_topLeft = int(y_topLeft)
    #                 max_iou_ground_x_bottomRight = int(x_bottomRight)
    #                 max_iou_ground_y_bottomRight = int(y_bottomRight)
                
    #             if num_annotations == len(anns):

    #                 # frame = cv2.imread('/home/pi/val2017/' + image_files[iterator_record[i]])

    #                 # cv2.rectangle(frame, (bbox_record[counter], bbox_record[counter+1]), (bbox_record[counter+2], bbox_record[counter+3]),(0,0,255), 2)
    #                 # cv2.putText(frame, 'detected', (bbox_record[counter], bbox_record[counter+1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)    
                    
    #                 # cv2.rectangle(frame, (max_iou_ground_x_topLeft,max_iou_ground_y_topLeft), (max_iou_ground_x_bottomRight,max_iou_ground_y_bottomRight),(255,0,0), 2)
    #                 # cv2.putText(frame, 'ground', (max_iou_ground_x_topLeft, max_iou_ground_y_topLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)    

    #                 # # show the output frame
    #                 # cv2.imshow("Frame", frame)
    #                 # key = cv2.waitKey(0) & 0xFF
    #                 # # if the `q` key was pressed, break from the loop
    #                 # if key == ord("q"):
    #                 #     break    
                                                
    #                 accuracy_record.append(max_iou)
    #                 counter += 5

        # net_accuracies[net] = mean(accuracy_record)
        net_durations[net] = mean(pdr)
        # print('[INFO]: Average {} accuracy: {}.'.format(net, net_accuracies[net]))
        print('[INFO]: Average {} duration: {}.'.format(net, net_durations[net]))

    # if device == 'nano' :
    #     net_info = {'net':['MobileNetV2SSD', 'InceptionV2SSD'],
    #                 'duration':[0.35, 0.675],
    #                 'accuracy':[0.7181, 0.739] 
    #                 }
    # elif device == 'rpi' :
    #     net_info = {'net':['MobileNetV2SSD', 'InceptionV2SSD'],
    #                 'duration':[0.8, 1.775],
    #                 'accuracy':[0.7181, 0.739] 
    #                 }

    # net_accuracies = {k: v for k, v in sorted(net_accuracies.items(), key=lambda item: item[1])}
    # net_names = list(net_accuracies.keys())

    # net_accuracies = pd.DataFrame(net_accuracies, index=[0])
    net_durations = pd.DataFrame(net_durations, index=[0])
    print('[INFO]: Net info gathered.')
    # print('[INFp]: {}'.format(net_durations))
    return net_durations

