''' 
Code to extract fingerprints from database songs and save as a dictionary.
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import json
from scipy import ndimage


def fingerprintBuilder(database_audio_folder, fingerprints_path):
    '''
    Function which will compute the spectral peaks of a given audio file, then use these to extract audio fingerprints

    Inputs:
    - database_audio_folder (str): path to the folder containing all database files
    - fingerprints_path (str): path to save the database fingerprints dict to

    Returns:
    - hashes (dict): dictionary with each unique hash as a key and values of [(time_offset, song_name)], containing all database songs with this hash

    Outputs:
    - Text file of hashes dict

    '''

    ### Parameters:

    # Spectral peak window:
    dist_freq = 16 # in bins
    dist_time = 16 # in samples
    thresh = 0.01 # threshold to consider it a peak
    sr = 22050 # sample-rate (Hz)

    # Target-window:
    tz_dist = 1 # distance from anchor point (secs)
    tz_time = 11 # time length (secs)
    tz_freq = 700 # frequency length (bins)

    plot = False # change to true if you want to plot spectrogram and constellation map


    ### Iterate through each file in the database:
    all_files = os.listdir(database_audio_folder)
    all_fingerprints = [] 
    fingerprints_dict = {}
    count = 0
    for file in all_files:
        if file[-4:] == '.wav': # check that it is an audio-file
            print('Current File:', count+1, 'out of:', len(all_files))
            audio_file_path = database_audio_folder + '/' + file
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
                
                fig2, ax2, = plt.subplots()
                peaks = np.where(Cmap)
                # -> produces tuple (rows, cols) where true vals exist

                frames = []
                freq_peaks = []
                for i, freq_bin in enumerate(peaks[0]): # iterate through every frame
                    frame = peaks[1][i] # find the freq bin peaks of the frame
                    frame_time = librosa.frames_to_time(frame, sr=sr) # convert frame to time in secs
                    frames.append(frame_time)
                    freq_peaks.append(freq_bin)

                ax2.scatter(frames, freq_peaks, s=5, color = 'r')
                ax2.set_title('Spectral Peaks')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Freq. Bin')
                plt.show()

            print('Computing fingerprints...')


            ### Extract pairs of peaks: ###

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
                                    print('ERROR!')
                                # Construct fingerprint:
                                fingerprint = ((f_diff, t_diff), int(t1), str(file))
                                file_fingerprints.append(fingerprint)

                fingerprints_dict[file] = file_fingerprints # add file fingerprints to dict
        
            for fingerprint in file_fingerprints:
                all_fingerprints.append(fingerprint) # add to list of all fingerprints


    # Iterate through all fingerprints and store as a dict of hashes:
    hashes = {}
    for i, fingerprint in enumerate(all_fingerprints):
        hash_val = ((all_fingerprints[i][0]))
        hash_str = "{},{}".format(hash_val[0], hash_val[1])
        time_offset = all_fingerprints[i][1]
        song_name = all_fingerprints[i][2]
        if hash_str in hashes:
            hashes[hash_str].append((time_offset, song_name))
        else:
            hashes[hash_str] = [(time_offset, song_name)]

    # Write to file:
    with open(fingerprints_path, 'w') as convert_file: 
        convert_file.write(json.dumps(hashes))

    
    print('Finished all fingerprinting')

    return hashes