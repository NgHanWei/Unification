# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:59:39 2022

@author: NEETHU

"""

import numpy as np
def remove_elements_between_markers(arr):
    hotkey_start_marker, hotkey_end_marker = 190, 192
    # Flag to track whether we are inside the specified range
    in_range = False
    # List to hold the filtered elements
    filtered_arr = []

    for item in arr:
        current_marker = item[2]
        if current_marker == hotkey_start_marker:
            # Start skipping elements
            in_range = True
        elif current_marker == hotkey_end_marker and in_range:
            # Stop skipping elements after encountering the end marker
            in_range = False
            continue  # Skip adding the end marker and reset for next potential range

        if not in_range:
            filtered_arr.append(item)

    return np.array(filtered_arr)


def remove_interrupted_trial(rr2):
    rr2 = remove_elements_between_markers(rr2)
    good_trials = []
    i = 0
    last_start = None

    start_markers = {103, 106}  # Set of markers indicating 'start'
    end_markers = 120
    while i < len(rr2):
        current = rr2[i][2]
        # Check if current marker is an starting marker
        if current in start_markers:
            if last_start is not None:
                # Check if there's a 'end_markers' between the last starting and this one
                if not any(sub[2] == end_markers for sub in rr2[last_start:i]):
                    # If no 'end_markers', skip adding these elements
                    pass
            last_start = i
        elif current == end_markers and last_start is not None:
            # Add the range from last starting marker to this 'end_markers' to good_trials
            good_trials.extend(rr2[last_start:i + 1])
            last_start = None  # Reset last_start after closing a valid block
        else:
            # This element is outside of any '199'/'200' to 'end_markers' block
            if last_start is None:
                # Only add elements that are outside of any abnormal block
                good_trials.append(rr2[i])
        i += 1

    return np.array(good_trials)

def count_interruption(arr):
    interruption_start_marker = 190
    end_marker = 120
    # Count the occurrences of each start marker and the end marker
    count_end_marker = sum(1 for item in arr if item[2] == end_marker)
    # Count the occurrences of each start marker and the end marker
    count_interruption = sum(1 for item in arr if item[2] == interruption_start_marker)

    # Verification: The sum of start markers should be equal to the count of the end marker
    if count_interruption != 0:
        print("Interruption detected!")
        print(f"The number of interruption is {count_interruption}")
    else:
        print("No interruption detected!")
        print(f"The number of trial is {count_end_marker}")

def eventlist_grazmi(event, fsamp):
    """
    Event list:   (UI --> VMRK --> MNE --> description)     
    100 --> S  4 --> 4 --> Experiment start/START_EXP_LABEL
    101 --> S  5 --> 5 --> ??/INTRUCTIONS_LABEL
    102 --> S  6 --> 6 --> Trial start/START_TRIAL_LABEL
    103 --> S  7 --> 7 --> Presentation of Left Cue/LEFT_PROMPT_LABEL
    105 --> S  9 --> 9 --> Presentation of Right Cue/RIGHT_PROMPT_LABEL
    104 --> S  8 --> 8 --> Start Left MI/LEFT_MI_LABEL
    106 --> S 10 --> 10 --> Start Right MI/RIGHT_MI_LABEL
    107 --> S 11 --> 11 --> Stop MI/RIGHT_MI_LABEL
    108 --> S 12 --> 12 --> Experiment stop/END_EXP_LABEL
    """
    codeL = 8
    codeR = 10
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):     
        if t[2]!=1006: #exclude all the duplicate markers
            if (t[2] == codeL):
                ev['label'].append('left_mi')
                ev['code'].append(0)
            elif (t[2] == codeR):
                ev['label'].append('right_mi')
                ev['code'].append(1)
            else:
                ev['label'].append('non')
                ev['code'].append(-1)
            ev['sampstart'].append(t[0])    
    return ev

def eventlist_cirgmi(event, fsamp): #old version
    """
    Event list: (UI --> VMRK --> MNE --> description)     
    102 --> R  4 --> Start of trial
    103 --> R  6 --> Presentation of Left Cue 
    105 --> R 10 --> Presentation of Right Cue
    104 --> R  8 --> Left MI starts  
    106 --> R 12 --> Right MI starts
    107 --> R 14 --> MI ends    
    """
    codeL = 1004
    codeR = 1006
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):        
        if (t[2] == codeL):
            ev['label'].append('left_mi')
            ev['code'].append(0)
        elif (t[2] == codeR):
            ev['label'].append('right_mi')
            ev['code'].append(1)
        else:
            ev['label'].append('non')
            ev['code'].append(-1)
        ev['sampstart'].append(t[0])    
    return ev

def eventlist_HRmi(event, fsamp):
    """
    Event list: (UI --> VMRK --> MNE --> description)     
    100 --> S  4 --> 4 --> Experiment start/START EXP_LABEL
    101 --> S  5 --> 5 --> FIXATION_LABEL 
    102 --> S  6 --> 6 --> PREP_CLOSE_LABEL
    103 --> S  7 --> 7 --> PREP_OPEN_LABEL
    104 --> S  8 --> 8 --> TRIAL_CLOSE_LABEL
    105 --> S  9 --> 9 --> TRIAL_OPEN_LABEL
    106 --> S 10 --> 10 --> REST_LABEL
    107 --> S 11 --> 11 --> END_EXP_LABEL
    """
    codeO = 8
    codeC = 9
    codeR = 10
    
    ev = dict({'code': [], 'sampstart': [],'label': []})
    for i,t in enumerate(event[0]):   
        if t[2]!=1006: #exclude all the duplicate markers
            if (t[2] == codeO):
                ev['label'].append('hand_open')
                ev['code'].append(0)
            elif (t[2] == codeC):
                ev['label'].append('hand_close')
                ev['code'].append(1)
            # if (t[2] == codeR):
            #     ev['label'].append('hand_rest')
            #     ev['code'].append(1)
            else:
                ev['label'].append('non')
                ev['code'].append(-1)
            ev['sampstart'].append(t[0])    
    return ev

def eventlist_ROC(event, fsamp, paradigm):
    """
    Event list: (UI --> VMRK --> MNE --> description)
   codeO = 107, beforeaudioO=106
   codeC = 104, beforeaudioC=103
   codeR=110,BESTILLAUDIO=109


    """
    codeO = 107
    codeC = 104
    codeR = 110
    import numpy as np
    rr = event[0]
    rr2 = np.delete(rr, rr[:, 2] == 1006, axis=0)  # deleting the 1006
    
    count_interruption(rr2)
    rr2 = remove_interrupted_trial(rr2)
    count_interruption(rr2)

    ev = dict({'code': [], 'sampstart': [], 'label': []})
    trialorder = 0

    if paradigm == 'Open':
        print("REST VS OPEN")

        for t in range(0, rr2.shape[0]):
            # print(rr2[t][2])
            if (rr2[t][2] == 109 and rr2[t + 1][2] == 110):  # bestill audio 109 and bestill task is 110
                ev['label'].append('rest')  #
                ev['code'].append(0)
                ev['sampstart'].append(rr2[t + 1][0])
                trialorder = trialorder + 1
            if (rr2[t][2] == 106 and rr2[t + 1][2] == 107):  # prep open audio 106 and task open 107
                ev['label'].append('hand_open')
                ev['code'].append(1)
                ev['sampstart'].append(rr2[t + 1][0])
                trialorder = trialorder + 1

    else:
        print("REST VS CLOSE")
        for t in range(0, rr2.shape[0]):
            # print(rr2[t][2])
            if (rr2[t][2] == 109 and rr2[t + 1][2] == 110):  # bestill audio 109 and bestill task is 110
                ev['label'].append('rest')  #
                ev['code'].append(0)
                ev['sampstart'].append(rr2[t + 1][0])
                trialorder = trialorder + 1
            if (rr2[t][2] == 103 and rr2[t + 1][2] == 104):  # prep close audio 103 and task close 104
                ev['label'].append('hand_close')
                ev['code'].append(2)
                ev['sampstart'].append(rr2[t + 1][0])
                trialorder = trialorder + 1

    # print("TRIALNUMS",trialorder)

    return ev