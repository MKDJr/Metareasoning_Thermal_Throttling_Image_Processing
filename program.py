###############
### PROGRAM ###
###############

# Import packages
import datetime
from auxiliary_functions import *
from metareasoning import *
import pandas as pd
from q_learning import *

# Program
def program(path, algorithm_name, start_net, 
            device, stop, start_temp, threshold_temp, TAC, 
            strat, initial_pause_duration, pause_adjustment_coef,
            nets, net_names, net_accuracies, net_durations, image_files) :

    # Print test info (for debugging)
    print('[INFO]: Running test with: Stop = {}, strat = {}, TAC = {}, IPD = {}, PAC = {}, start_net = {}'.format(
        stop, strat, TAC, initial_pause_duration, pause_adjustment_coef, start_net
    ))
    
    # Calculate maximum loop length, throughput, and accuracy
    max_loop_length = net_durations.max(axis=1).tolist()[0]
    max_throughput = 1/max_loop_length
    max_accuracy = net_accuracies.max(axis=1).tolist()[0]

    # If running q-learning...
    if algorithm_name == 'alg4' :
        # If training, initialize q-table (uncomment below)
        # q_table_df = initialize_q_table()

        # If testing, load q-table (uncomment below)
        if device == 'rpi' :
            q_table_df = pd.read_csv(os.path.join('/home/pi/Research/TestBatches/2022-03-30--17-24-29--Alg4RPi2/alg4/Strat_0/q_table.csv'))
        elif device == 'nano' :
            q_table_df = pd.read_csv('/home/michael/Research/TestBatches/2022-03-31--11-21-05--Alg4Nano2/alg4/Strat_0/q_table.csv', index_col=False)
        
        # Cleans up loaded q-table
        q_table_df = q_table_df.loc[:,~q_table_df.columns.str.contains('^Unnamed')]

    # print(list(q_table_df.columns))
    
    # Define initial values for data records
    cpu_record = [0]
    loop_duration_record = [0]
    net_record = [start_net]
    temp_record_CPU = [measure_temp_CPU()]
    temp_record_GPU = [measure_temp_GPU()]
    processing_duration_record = [0]
    time_record = [0]
    iterator_record = [0]
    avg_accuracy_record = [max_accuracy]
    pause_duration_record = [initial_pause_duration]
    action = 'NA'
    action_record = ['NA']

    # Run function to stabilize temperature at desired start temperature
    stabilize_temp(start_temp, nets)

# Main Loop
    # loop over the frames from the video stream
    while time_record[-1] < stop :
        
    # Record loop start time
        loop_start_time = datetime.datetime.now()

    # Process Image
        # blob, h, w = image_prep(device, image_files, iterator_record)
        # detections = process_image(net_record, blob, nets, net_names)
        process_image_tf(device, image_files, iterator_record, net_record, nets)
  
    # Record Processing end time
        processing_end_time = datetime.datetime.now()
        processing_duration = (processing_end_time - loop_start_time).total_seconds()

    # Record Temperature and CPU
        record_data_1(device, iterator_record, time_record, 
                        processing_duration, temp_record_CPU, temp_record_GPU, 
                        processing_duration_record, loop_duration_record, cpu_record,
                        pause_duration_record, avg_accuracy_record, action_record)


    # Metareasoning    
        if algorithm_name == 'alg1' :
            pause_duration, avg_accuracy = alg1(net_record, pause_duration_record, pause_adjustment_coef, 
        temp_record_CPU, threshold_temp, net_accuracies, max_accuracy)

        if algorithm_name == 'alg2' :
            pause_duration, avg_accuracy = alg2(pause_duration_record, start_temp, temp_record_CPU, threshold_temp,
         net_record, net_accuracies, net_durations, max_accuracy, strat, net_names, max_loop_length,
         pause_adjustment_coef)
            
        if algorithm_name == 'alg3' :
            pause_duration, avg_accuracy = alg3(pause_duration_record, temp_record_CPU, start_temp, threshold_temp, 
        net_record, max_accuracy, net_accuracies, net_durations, max_throughput, 
        TAC, strat, net_names, pause_adjustment_coef)

        if algorithm_name == 'alg4' :
            action, q_row_index, avg_accuracy = classify_state(net_names, net_record, temp_record_CPU, pause_duration_record, q_table_df, max_accuracy, net_accuracies) 
            pause_duration = pick_action(action, net_record, pause_duration_record, net_names, 
                max_loop_length, net_durations, pause_adjustment_coef,
                temp_record_CPU, threshold_temp)

    # record loop end time
        loop_end_time = datetime.datetime.now()
        loop_duration = (loop_end_time - loop_start_time).total_seconds()

    # List Updates
        record_data_2(device, iterator_record, time_record, loop_duration, 
                        processing_duration, temp_record_CPU, temp_record_GPU, 
                        processing_duration_record, loop_duration_record, cpu_record,
                        pause_duration_record, pause_duration, avg_accuracy_record, 
                        avg_accuracy, action_record, action)

        # Uncomment to train q_table   
        # if algorithm_name == 'alg4' :
        #     reward =  give_reward(temp_record_CPU, threshold_temp)
        #     update_state_value(net_names, net_record, temp_record_CPU, pause_duration_record, q_table_df, q_row_index, action, reward)

    results = save_data(path, device, iterator_record, time_record, temp_record_CPU,
                temp_record_GPU, cpu_record, processing_duration_record,
                pause_duration_record, loop_duration_record, 
                net_record, avg_accuracy_record, action_record)

    # Save final q-table if training
    # if algorithm_name == 'alg4' : 
    #     q_table_df.to_csv(os.path.join(path, "q_table" + ".csv"))       


    print('[INFO]: Test Complete.')

    return results