#############################
### METAREASONING PROGRAM ###
#############################

# Import packages
import time

# Policy 1
def alg1(net_record, pause_duration_record, pause_adjustment_coef, 
        temp_record_CPU, threshold_temp, net_accuracies, max_accuracy) :

    # Record neural network 
    net_record.append(net_record[-1])
    net_record.append(net_record[-1])

    # Assign pause duration value
    pause_duration = pause_duration_record[-1] + pause_adjustment_coef*(temp_record_CPU[-1] - threshold_temp)
    if pause_duration < 0 :
        pause_duration = 0

    # Pause
    time.sleep(pause_duration)

    # "Accuracy" here is actually precision, but I don't want to change every variable name"
    # Calculate average expected precision across the last five frames
    avg_accuracy = max_accuracy
    sum_accuracy = 0
    accuracy_window = 5
    if len(net_record) > accuracy_window*2+1 :
        for i in range(accuracy_window) :
            working_net = net_record[-(i*2+1)]
            sum_accuracy = sum_accuracy + net_accuracies[working_net].tolist()[0]
        avg_accuracy = sum_accuracy/accuracy_window

    return pause_duration, avg_accuracy

# Policy 2
def alg2(pause_duration_record, start_temp, temp_record_CPU, threshold_temp,
         net_record, net_accuracies, net_durations, max_accuracy, strat, net_names, max_loop_length,
         pause_adjustment_coef) :

    # Pause
    time.sleep(pause_duration_record[-1])
    
    # Define quintile size
    quintile = (threshold_temp - start_temp)/5
    log_quintile = (threshold_temp - start_temp)/2

    # Strategy 1 (gearshifting)
    if strat == 1 :

        k = net_names.index(net_record[-1])

        if temp_record_CPU[-1] > threshold_temp : 

            if k != 4 :        
                net_record.append(net_names[k+1])
                net_record.append(net_names[k+1])
            
            else :
                net_record.append(net_record[-1])
                net_record.append(net_record[-1])

        elif k != 0 :  
            net_record.append(net_names[k-1])
            net_record.append(net_names[k-1])

        else :
            net_record.append(net_record[-1])
            net_record.append(net_record[-1])

    # Strategy 2 (linear)
    if strat == 2 :
        i = 0
        while  temp_record_CPU[-1] > (start_temp + quintile) :
            i += 1
            quintile = ((threshold_temp - start_temp) / 5) * i
            if i == len(net_names)-1 : break

        net_record.append(net_names[i])
        net_record.append(net_names[i])

    # Strategy 3 (logarithmic)
    if strat == 3 :
        i = 0
        while  temp_record_CPU[-1] > (start_temp + log_quintile) :
            i += 1
            log_quintile = log_quintile + (threshold_temp - (log_quintile + start_temp)) / 2 
            if i == len(net_names)-1 : break

        net_record.append(net_names[i])
        net_record.append(net_names[i])

    # Assign next pause duration value
    pause_duration = max_loop_length - net_durations[net_record[-1]].tolist()[0]

    # Calculate average expected precision across the last five frames
    avg_accuracy = max_accuracy
    sum_accuracy = 0
    accuracy_window = 5
    if len(net_record) > accuracy_window*2+1 :
        for i in range(accuracy_window) :
            working_net = net_record[-(i*2+1)]
            sum_accuracy = sum_accuracy + net_accuracies[working_net].tolist()[0]
        avg_accuracy = sum_accuracy/accuracy_window

        if (avg_accuracy == net_accuracies.min(axis=1).tolist()[0]) and (temp_record_CPU[-1] > threshold_temp) :
            pause_duration = pause_duration + pause_duration_record[-1] + pause_adjustment_coef*(temp_record_CPU[-1] - threshold_temp)
            print('contingency, pausing for {}'.format(pause_duration))

    if pause_duration < 0 :
        pause_duration = 0

    return pause_duration, avg_accuracy

# Policy 3 
def alg3(pause_duration_record, temp_record_CPU, start_temp, threshold_temp, 
        net_record, max_accuracy, net_accuracies, net_durations, max_throughput, 
        TAC, strat, net_names, pause_adjustment_coef) :
        
    # Pause
    time.sleep(pause_duration_record[-1])

    quintile = (threshold_temp - start_temp)/5
    log_quintile = (threshold_temp - start_temp)/2

    if strat == 1 :

        k = net_names.index(net_record[-1])

        if temp_record_CPU[-1] > threshold_temp : 

            if k != 4 :        
                net_record.append(net_names[k+1])
                net_record.append(net_names[k+1])
            
            else :
                net_record.append(net_record[-1])
                net_record.append(net_record[-1])

        elif k != 0 :  
            net_record.append(net_names[k-1])
            net_record.append(net_names[k-1])

        else :
            net_record.append(net_record[-1])
            net_record.append(net_record[-1])

    if strat == 2 :
        i = 0
        while  temp_record_CPU[-1] > (start_temp + quintile) :
            i += 1
            quintile = ((threshold_temp - start_temp) / 5) * i
            if i == len(net_names)-1 : break

        net_record.append(net_names[i])
        net_record.append(net_names[i])

    if strat == 3 :
        i = 0
        while  temp_record_CPU[-1] > (start_temp + log_quintile) :
            i += 1
            log_quintile = log_quintile + (threshold_temp - (log_quintile + start_temp)) / 2 
            if i == len(net_names)-1 : break

        net_record.append(net_names[i])
        net_record.append(net_names[i])

    # print('using network {}'.format(net_record[-1]))

    avg_accuracy = max_accuracy
    sum_accuracy = 0
    accuracy_window_max = 5
    
    # Calculate average expected precision across the last five frames
    accuracy_window = int(len(net_record)/2)
    if accuracy_window > accuracy_window_max : accuracy_window = accuracy_window_max
    for i in range(accuracy_window) :
        working_net = net_record[-(i*2+1)]
        sum_accuracy = sum_accuracy + net_accuracies[working_net].tolist()[0]
    avg_accuracy = sum_accuracy/accuracy_window
    # print('Avg acc: {}'.format(avg_accuracy))

    # Calculate desired throughput
    desired_throughput = max_throughput - (max_accuracy - avg_accuracy)*TAC
    if desired_throughput < 0 : desired_throughput = 0.01
    # print('max throughput: {}'.format(max_throughput))
    # print('max_accuracy: {}'.format(max_accuracy))
    # print('desired throughput: {}'.format(desired_throughput))
    desired_loop_length = 1/desired_throughput
    # print('desired loop length: {}'.format(desired_loop_length))
    # print('using net duration: {}'.format(net_durations[net_record[-1]].tolist()[0]))
    pause_duration = desired_loop_length - net_durations[net_record[-1]].tolist()[0]
    # print('using tac to adjust, pause for: {}'.format(pause_duration))

    if (avg_accuracy == net_accuracies.min(axis=1).tolist()[0]) :
        pause_duration = pause_duration + pause_duration_record[-1] + pause_adjustment_coef*(temp_record_CPU[-1] - threshold_temp)
        print('contingency, pausing for {}'.format(pause_duration))

    if pause_duration < 0 : 
        pause_duration = 0
        # print('pause to low, setting to 0')

    # else :  
    #     pause_duration = 0
    #     # print('waiting for enough loops')

    # ADD PAUSE
    
    return pause_duration, avg_accuracy