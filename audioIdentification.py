''' 
Code to match query songs to songs in the database, using hash matching
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import json
from scipy import ndimage

def audioIdentification(query_folder_path, fingerprints_path, output_txt_path):
    ''' 
    Function to match hashes from query songs with database songs, to predict which song each query is from

    Inputs:
    - query_folder_path (str): file path of folder containing query songs
    - fingerprints_path (str): file path of file containing database fingerprints
    - output_txt_path (str): file path of folder to save predictions to

    Returns:
    - predictions_dict (dict): dictionary containing each query song and the predicted top 3 matches

    Outputs:
    - A text file with tab seperated cols: (file, 1st prediction, 2nd prediction, 3rd prediction)
    '''

    ### Parameters:

    # Spectral peak window:
    dist_freq = 16 # in bins
    dist_time = 16 # in samples
    thresh = 0.01 # threshold to consider it a peak

    # Target-window:
    tz_dist = 1 # distance from anchor point (secs)
    tz_time = 11 # time length (secs)
    tz_freq = 700 # frequency length (bins)

    plot = False # change to true if you want to plot spectrogram and constellation map



    ### Load dataset hashes:

    with open(fingerprints_path) as f: 
        raw_hash_data = f.read() 

    raw_dict = json.loads(raw_hash_data) 

    hash_dict = {}
    for i in raw_dict.keys():
        hash_dict[str(i)] = raw_dict[i]

    ### Iterate through each file in the database:
    all_files = os.listdir(query_folder_path)
    predictions_dict = {}
    count = 0
    for file in all_files:
        if file[-4:] == '.wav': # check that it is an audio-file
            print('Current File:', count+1, 'out of:', len(all_files))
            audio_file_path = query_folder_path + '/' + file
            count+=1


            ### Load audio file and plot spectrogram for visualisation: ###

            audio_file, sr = librosa.load(audio_file_path)
            audio_file_spec = np.abs(librosa.stft(audio_file))
                # --> returns: matrix: [freq. bins, frames]
            print('Loaded audio file:', audio_file_path)
            
            result = ndimage.maximum_filter(audio_file_spec, size=[2*dist_freq+1, 2*dist_time+1], mode='constant') # find peaks
            Cmap = np.logical_and(audio_file_spec == result, result > thresh)

            if plot == True:
                # Plot spectrogram:
                fig, ax = plt.subplots()
                img = librosa.display.specshow(librosa.amplitude_to_db(audio_file_spec,
                                                                    ref=np.max),
                                            y_axis='log', x_axis='time', ax=ax)
                ax.set_title('Power spectrogram')
                fig.colorbar(img, ax=ax, format="%+2.0f dB")

                ### Plot Spectral Peaks: ###
                
                ax2, = plt.subplots()
                peaks = np.where(Cmap)
                # -> produces tuple (rows, cols) where true vals exist

                frames = []
                freq_peaks = []
                for i, freq_bin in enumerate(peaks[0]):
                    frame = peaks[1][i]
                    frame_time = librosa.frames_to_time(frame, sr=sr)
                    frames.append(frame_time)
                    freq_peaks.append(freq_bin)

                ax2.scatter(frames, freq_peaks, s=5, color = 'r')
                ax2.set_title('Spectral Peaks')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Freq. Bin')
                plt.show()

        

            ### Extract pairs of peaks: ###
            print('Computing fingerprints...')

            # Find frames with peaks and freq-bin of the peaks:
            peaks = np.where(Cmap.T) # find indices of peaks
                # returns tuple: --> [frames, peak_freq_bin]
            num_peaks = len(peaks[0])
            peak_frames = peaks[0]
            peak_bins = peaks[1]


            file_fingerprints = [] # initialise
            for i in range(num_peaks): # iterate through each peak

                # get anchor point:
                t1 = peak_frames[i]
                f1 = peak_bins[i]

                # get point if it is in the target zone:
                for j in range(num_peaks):
                    if j != i:
                        if (t1 + tz_dist) <= peak_frames[j] <= (t1 + (tz_dist+tz_time)):
                            if (f1 - tz_freq) <= peak_bins[j] <= (f1 + tz_freq):
                                t2 = peak_frames[j]
                                f2 = peak_bins[j]
                                t_diff = int(np.abs(t1-t2))
                                f_diff = int(np.abs(f1-f2))
                                # Ensure that these are the indices of peaks:
                                if not (Cmap[f1,t1] == True and Cmap[f2,t2] == True): 
                                    print('ERROR! Peaks and Indices do not match.')
                                # Construct fingerprint:
                                fingerprint = ((f_diff, t_diff), int(t1), str(file))
                                file_fingerprints.append(fingerprint) # append to list of fingerprints


            ### Store query fingerprints:
            query_fingerprints = []
            for i, fingerprint in enumerate(file_fingerprints):
                hash_val = ((file_fingerprints[i][0]))
                hash_str = "{},{}".format(hash_val[0], hash_val[1])
                time_offset = file_fingerprints[i][1] 
                query_fingerprint = (hash_str, time_offset) 
                song_name = str(file_fingerprints[i][2]) 
                if query_fingerprint in query_fingerprints:
                        pass
                else: # if not, add to list
                    query_fingerprints.append(query_fingerprint)
            print('Number of unique hashes for file:', file, 'is:', len(query_fingerprints))


            ### For each hash in query hashes, match with dataset songs containing the hash:

            print('Matching hashes from database')
            dataset_hashes = list(hash_dict.keys())

            hash_match_dict = {}
            for fingerprint in query_fingerprints: # iterate through all unique query hashes
                hash_val = fingerprint[0]
                query_offset = fingerprint[1]
                if hash_val in dataset_hashes:
                    matched_vals = hash_dict[hash_val]
                    for match in matched_vals:
                        dataset_offset = match[0]
                        song_name = match[1]
                        t_shift = int(np.abs(query_offset-dataset_offset))
                        fingerprint = (hash_val, (t_shift, song_name))
                        # add to dict of matches:
                        if song_name in hash_match_dict:
                            hash_match_dict[song_name].append(fingerprint)
                        else: 
                            hash_match_dict[song_name] = [fingerprint]

        
            ### Extract time shifts for each hash match, for each song and find peak from histograms:

            print('Matching time-shifts')
            time_matches = [] 
            for song_name, matches in hash_match_dict.items(): # iterate through all matching hashes and get all matching songs
                t_shifts = []
                for match in matches: # for all matches for a given song
                    t_shift = match[1][0]
                    t_shifts.append(t_shift)
                song_hist, _ = np.histogram(t_shifts, 2500)
                hist_peak = np.max(song_hist)
                time_matches.append((song_name, hist_peak))
                

            ### Sort time_matches to find top 3 songs with most time matches:
            time_matches = sorted(time_matches, key = lambda x: x[1], reverse=True)
            top_3_matches = time_matches[:3]



            print('Song:', file, 'Top 3 matches from database:', top_3_matches)

            ### Add to dict to store:
            # predictions_dict[file] = top_3_matches
            predictions_dict[file] = top_3_matches

    
    ### Save predictions to file:
    with open(output_txt_path, 'w') as convert_file: 
        for file_name, matches in predictions_dict.items():
            convert_file.write(file_name)
            convert_file.write('\t') 

            song_names = [str(match[0]) for match in matches[:3]]  # Extract song names from each tuple: (name, score)
            convert_file.write('\t'.join(song_names))
            convert_file.write('\n')

    return predictions_dict


    



    
