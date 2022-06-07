##########################
### Q-Learning Program ###
##########################

# Import packages
import numpy as np
import pandas as pd
import time

# Set q-learning parameters
gamma = 0.9
alpha = 0.2
epsilon = 0.9

# Initialize q-table
def initialize_q_table() :
    possible_temperatures = np.around(np.arange(35,95+0.1,0.1),1)
    possible_pauses = np.around(np.arange(0,10,0.1),1)
    possible_networks = range(0,5)
    states = np.array(np.meshgrid(possible_temperatures, possible_pauses, possible_networks)).T.reshape(-1,3)
    actions = np.zeros((4,1)).repeat(len(states),1).T
    q_table = np.hstack((states,actions))

    q_table_df = pd.DataFrame(q_table, columns = ['Temperature','Pause Duration', 'Network','Nothing','Downgrade','Upgrade','Adjust'])

    return q_table_df

# CLASSIFY STATE
def classify_state(net_names, net_record, temp_record_CPU, pause_duration_record, q_table_df, max_accuracy, net_accuracies) :
    k = net_names.index(net_record[-1])

    stateID = [round(temp_record_CPU[-1],1),round(pause_duration_record[-1],1), k]
    print('current state: {}'.format(stateID))
    # CHOOSE ACTION
    # check epsilon
    # lookup values to find the action to take
    q_row_index = q_table_df.index[(q_table_df['Temperature'] == stateID[0]) & (q_table_df['Pause Duration'] == stateID[1]) & (q_table_df['Network'] == stateID[2])].tolist()[0]
    print('q_row_index: {}'.format(q_row_index))
    q_row = q_table_df.iloc[[q_row_index]].to_numpy()[0]
    print('state-action pair values: {}'.format(q_row[3:])) 
    rand = np.random.uniform()
    if rand < epsilon :       
        print('finding best action')
        action = np.argwhere(q_row[3:] == max(q_row[3:])) # 3, 4, 5, or 6 (because 0, 1, 2 are state id)
        print('action: {}'.format(action))
        action = action[0][0]+3
        if ((max(q_row[3:]) == 0) & (min(q_row[3:]) == 0)) :
            print('state unexplored, taking random action')
            action = round(np.random.uniform(3.51,6.49))
    elif (rand > epsilon) :
        # or take a random action
        print('exploring actions')
        action = round(np.random.uniform(3.51,6.49))
    else :
        print('something wrong happened')
        action = round(np.random.uniform(3.51,6.49))

    print('supposed action: {}'.format(action))

    avg_accuracy = max_accuracy
    sum_accuracy = 0
    accuracy_window = 5
    if len(net_record) > accuracy_window*2+1 :
        for i in range(accuracy_window) :
            working_net = net_record[-(i*2+1)]
            sum_accuracy = sum_accuracy + net_accuracies[working_net].tolist()[0]
        avg_accuracy = sum_accuracy/accuracy_window

    return action, q_row_index, avg_accuracy

# Insert Pause 
def pick_action(action, net_record, pause_duration_record, net_names, 
                max_loop_length, net_durations, pause_adjustment_coef,
                temp_record_CPU, threshold_temp) :

    time.sleep(pause_duration_record[-1])

    k = net_names.index(net_record[-1])

    if action == 3 : 
        net_record.append(net_record[-1])
        net_record.append(net_record[-1])
        pause_duration = pause_duration_record[-1]

    elif action == 4 : 
        if k != 4 : 
            net_record.append(net_names[k+1])
            net_record.append(net_names[k+1])
            pause_duration = max_loop_length - net_durations[net_record[-1]].tolist()[0]

        else : 
            net_record.append(net_record[-1])
            net_record.append(net_record[-1])
            pause_duration = pause_duration_record[-1]

    elif action == 5 : 
        if k != 0 : 
            net_record.append(net_names[k-1])
            net_record.append(net_names[k-1])
            pause_duration = max_loop_length - net_durations[net_record[-1]].tolist()[0]

        else : 
            net_record.append(net_record[-1])
            net_record.append(net_record[-1])
            pause_duration = pause_duration_record[-1]

    elif action == 6 : 
        net_record.append(net_record[-1])
        net_record.append(net_record[-1])

        pause_duration = pause_duration_record[-1] + pause_adjustment_coef * (temp_record_CPU[-1] - threshold_temp)

    else :
        print('oops')
        net_record.append(net_record[-1])
        net_record.append(net_record[-1])
        pause_duration = pause_duration_record[-1]


    if pause_duration < 0 : pause_duration = 0

    return pause_duration


# GIVE REWARD
def give_reward(temp_record_CPU, threshold_temp) :
    if abs(temp_record_CPU[-1] - threshold_temp) < 1 :
        reward = 1/((temp_record_CPU[-1] - threshold_temp)**2) - 1
    elif abs(temp_record_CPU[-1] - threshold_temp) > 1 :
        reward = -1*(temp_record_CPU[-1] - threshold_temp)**2 + 1
    else :
        reward = 0

    return reward

# Update state value
def update_state_value(net_names, net_record, temp_record_CPU, pause_duration_record, q_table_df, q_row_index, action, reward) :

    # CALCULATE NEW STATE

    k = net_names.index(net_record[-1])

    new_stateID = [round(temp_record_CPU[-1],1), round(pause_duration_record[-1],1), k]
    print('new state: {}'.format(new_stateID))
    new_q_row_index = q_table_df.index[(q_table_df['Temperature'] == new_stateID[0]) & (q_table_df['Pause Duration'] == new_stateID[1]) & (q_table_df['Network'] == new_stateID[2])].tolist()[0]
    print('new_q_row_index: {}'.format(new_q_row_index))

    new_q_row = q_table_df.iloc[[new_q_row_index]].to_numpy()[0]
    new_q_max = max(new_q_row[3:])

    # UPDATE STATE VALUE
    q_table_df.iat[q_row_index,action] = q_table_df.iat[q_row_index,action] + alpha*(reward + gamma*new_q_max - q_table_df.iat[q_row_index,action])
            