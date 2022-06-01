
##############################
### TESTING PROGRAM (MAIN) ###
##############################

from auxiliary_functions import *
from program import program
import pandas as pd
import os
import numpy as np

print('[INFO]: Packages imported.')

alg1 = False
alg2 = False
alg3 = True
alg4 = False

# Check which device is being used
if platform.node() == 'michael-desktop' : device = 'nano'
else : device = 'rpi'

print('[INFO]: Device: {}'.format(device))

# Create parent directory for this batch
if device == 'nano' :
    parent_directory = '/home/michael/Research/TestBatches'
    threshold_temp = 55
    start_temp = 50
    stop = 600

elif device == 'rpi' :
    parent_directory = '/home/pi/Research/TestBatches'
    threshold_temp = 70
    start_temp = 50
    stop = 600

current_date_and_time = time.strftime("%Y-%m-%d--%H-%M-%S")
parent_directory = os.path.join(parent_directory,current_date_and_time)
os.mkdir(parent_directory)


if device == 'nano' :
    image_files = os.listdir('/home/michael/val2017/')
elif device == 'rpi' :
    image_files = os.listdir('/home/pi/val2017/')

# Define networks to be used in order of decreasing speed
net_names = ['efficientdet_lite4_detection_metadata_2',
            'efficientdet_lite3_detection_metadata_1',
            'efficientdet_lite2_detection_metadata_1',
            'efficientdet_lite1_detection_metadata_1',
            'efficientdet_lite0_detection_metadata_1']

# Use names to load models from tensorflow hubD
nets = load_models(net_names)

# Define network accuracies
accuracies = [.4196, .3770, .3397, .3055, .2569]
net_accuracies = {}
for num, net in enumerate(net_names) : net_accuracies[net] = accuracies[num]
net_accuracies = pd.DataFrame(net_accuracies, index=[0])

# Get network durations
net_durations = get_net_info(device=device, net_names=net_names, nets=nets, image_files=image_files, start_temp=start_temp)

if alg1 == True :
    algorithm_name = 'alg1'
    TAC = 0
    strat = 0

    program_path = os.path.join(parent_directory,algorithm_name)
    os.mkdir(program_path)



    # for lr in range (0,1) :
    #     lr = 1
    #     start_net = 'efficientdet_lite4_detection_metadata_2' 
    for start_net in net_names :
        net_path = os.path.join(program_path,start_net)
        os.mkdir(net_path)

        initial_pause_duration_records = []
        pause_adjustment_coef_records = []
        alg1_results = {}
        test_iterator = 0

        # for initial_pause_duration in [0] :
        for initial_pause_duration in np.linspace(0,2,num=5,endpoint=True) :
            # for pause_adjustment_coef in [0] :
            for pause_adjustment_coef in np.linspace(0,0.2,num=5,endpoint=True) :
                test_iterator += 1
        
                test_name = 'IPD_{}__PAC_{}'.format(initial_pause_duration,pause_adjustment_coef)
                test_path = os.path.join(net_path,test_name)
                os.mkdir(test_path)

                results = program(test_path, algorithm_name, start_net, 
                                    device, stop, start_temp, threshold_temp, TAC, 
                                    strat, initial_pause_duration, pause_adjustment_coef,
                                    nets, net_names, net_accuracies, net_durations, image_files)

                alg1_results[test_iterator] = results

                initial_pause_duration_records.append(initial_pause_duration)
                pause_adjustment_coef_records.append(pause_adjustment_coef)

        with pd.ExcelWriter(os.path.join(net_path, 'data.xlsx')) as writer:
            for i in range(1, len(alg1_results)+1):
                alg1_results[i].to_excel(writer, sheet_name='Test{}'.format(i))

        test_params = {'stop':stop,
            'threshold_temp':threshold_temp,
            'start_temp':start_temp,
            'start_net':start_net
            }

        data_frame = pd.DataFrame(test_params,index=[0])
        data_frame.to_csv(os.path.join(net_path, "test_params" + ".csv"))   

        metadata = {'pause_adjustment_coef_records':pause_adjustment_coef_records,
            'initial_pause_duration_records':initial_pause_duration_records
            }

        data_frame = pd.DataFrame(metadata)
        data_frame.to_csv(os.path.join(net_path, "metadata" + ".csv"))   


if alg2 == True :
    algorithm_name = 'alg2'
    TAC = 0
    start_net = net_names[0]

    initial_pause_duration = 0
    pause_adjustment_coef = 0.2

    program_path = os.path.join(parent_directory,algorithm_name)
    os.mkdir(program_path)
    alg2_results = {}
    
    test_iterator = 0
    strat_record = []

    for strat in range(1,4) :
        test_iterator += 1
        test_name = 'Strat_{}'.format(strat)
        test_path = os.path.join(program_path,test_name)
        os.mkdir(test_path)

        results = program(test_path, algorithm_name, start_net, 
            device, stop, start_temp, threshold_temp, TAC, 
            strat, initial_pause_duration, pause_adjustment_coef,
            nets, net_names, net_accuracies, net_durations, image_files)

                
        alg2_results[test_iterator] = results
        strat_record.append(strat)
        
    with pd.ExcelWriter(os.path.join(program_path, 'data.xlsx')) as writer:
        for i in range(1, len(alg2_results)+1):
            alg2_results[i].to_excel(writer, sheet_name='Test{}'.format(i))

    test_params = {'stop':stop,
    'threshold_temp':threshold_temp,
    'start_temp':start_temp,
    'start_net':start_net
    }

    data_frame = pd.DataFrame(test_params,index=[0])
    data_frame.to_csv(os.path.join(program_path, "test_params" + ".csv"))   

    metadata = {'strat_record':strat_record}
    data_frame = pd.DataFrame(metadata)
    data_frame.to_csv(os.path.join(program_path, "metadata" + ".csv"))   

if alg3 == True :
    algorithm_name = 'alg3'
    start_net = net_names[0]
    initial_pause_duration = 0
    pause_adjustment_coef = 0.2

    test_iterator = 0

    program_path = os.path.join(parent_directory,algorithm_name)
    os.mkdir(program_path)
    alg3_results = {}
    strat_record = []
    TAC_record = []

    for TAC in [0, 1, 2] :
        for strat in range (1,4) :
            test_iterator += 1
            test_name = 'TAC_{}__Strat_{}'.format(TAC,strat)
            test_path = os.path.join(program_path,test_name)
            os.mkdir(test_path)

            results = program(test_path, algorithm_name, start_net, 
                                device, stop, start_temp, threshold_temp, TAC, 
                                strat, initial_pause_duration, pause_adjustment_coef,
                                nets, net_names, net_accuracies, net_durations, image_files)

            alg3_results[test_iterator] = results
            strat_record.append(strat)
            TAC_record.append(TAC)

    with pd.ExcelWriter(os.path.join(program_path, 'data.xlsx')) as writer:
        for i in range(1, len(alg3_results)+1):
            alg3_results[i].to_excel(writer, sheet_name='Test{}'.format(i))


    test_params = {'stop':stop,
                    'threshold_temp':threshold_temp,
                    'start_temp':start_temp,
                    'start_net':start_net
                    }

    data_frame = pd.DataFrame(test_params,index=[0])
    data_frame.to_csv(os.path.join(program_path, "test_params" + ".csv"))   


    metadata = {'strat_record':strat_record,
                'TAC_record':TAC_record}
    data_frame = pd.DataFrame(metadata)
    data_frame.to_csv(os.path.join(program_path, "metadata" + ".csv"))  


if alg4 == True :
    algorithm_name = 'alg4'
    TAC = 0
    strat = 0
    start_net = net_names[0]

    initial_pause_duration = 0
    pause_adjustment_coef = 0.2

    program_path = os.path.join(parent_directory,algorithm_name)
    os.mkdir(program_path)
    alg4_results = {}
    
    test_iterator = 0
    strat_record = []

    test_iterator += 1
    test_name = 'Strat_{}'.format(strat)
    test_path = os.path.join(program_path,test_name)
    os.mkdir(test_path)

    results = program(test_path, algorithm_name, start_net, 
        device, stop, start_temp, threshold_temp, TAC, 
        strat, initial_pause_duration, pause_adjustment_coef,
        nets, net_names, net_accuracies, net_durations, image_files)

    alg4_results[test_iterator] = results
    strat_record.append(strat)
        
    with pd.ExcelWriter(os.path.join(program_path, 'data.xlsx')) as writer:
        for i in range(1, len(alg4_results)+1):
            alg4_results[i].to_excel(writer, sheet_name='Test{}'.format(i))

    test_params = {'stop':stop,
    'threshold_temp':threshold_temp,
    'start_temp':start_temp,
    'start_net':start_net
    }

    data_frame = pd.DataFrame(test_params,index=[0])
    data_frame.to_csv(os.path.join(program_path, "test_params" + ".csv"))   

    metadata = {'strat_record':strat_record}
    data_frame = pd.DataFrame(metadata)
    data_frame.to_csv(os.path.join(program_path, "metadata" + ".csv"))   


# Save batch data
batch_data = {'device' : device,
        'alg1' : alg1,
        'alg2' : alg2,
        'alg3' : alg3,
        'alg4' : alg4
        }

data_frame = pd.DataFrame(batch_data, index=[0])
data_frame.to_csv(os.path.join(parent_directory, "batch_data" + ".csv"))   



