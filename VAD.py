import scipy.io.wavfile as wav
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import python_speech_features as psf

# Load audio file
rate, sig = wav.read("./Upscale Audio/input.wav")

print("Rate: ", rate)
print("Signal: ", sig)
print("Signal Length: ", len(sig))
# Extract MFCCs
mfcc_feat = psf.mfcc(sig,rate,numcep=13)
print("MFCCs: ", mfcc_feat)
print("MFCCs Length: ", len(mfcc_feat))
# Normalize MFCCs
scaler = StandardScaler()
mfccs_norm = scaler.fit_transform(mfcc_feat)

# Perform k-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(mfccs_norm)
clusters = kmeans.predict(mfccs_norm)
print("Clusters: ", clusters)
print("Clusters Length: ", len(clusters))

# Get the time-steps of the MFCCs for each cluster
speaker_1_indices = np.where(clusters == 0)[0]
speaker_2_indices = np.where(clusters == 1)[0]

# Get the corresponding audio segments for each speaker
speaker_1_audio = sig[speaker_1_indices*len(sig)/len(mfcc_feat)]
speaker_2_audio = sig[speaker_2_indices*len(sig)/len(mfcc_feat)]

# Write audio segments for speaker 1 to file
wav.write("speaker_1.wav", rate, speaker_1_audio)

# Write audio segments for speaker 2 to file
wav.write("speaker_2.wav", rate, speaker_2_audio)
