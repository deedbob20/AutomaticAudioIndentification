# AutomaticAudioIndentification


This repository implements the Shazam-style audio fingerprinting algorithm proposed by **Wang (2003)**, designed for robust and efficient audio identification from short audio clips. The implementation is organized into two main Python scripts and a Jupyter notebook for testing and demonstration.

---

## Project Structure

- `fingerprintBuilder.py`  
  Extracts robust audio fingerprints (constellation maps → hashes) from audio files. This script builds the searchable fingerprint database.

- `audioIdentification.py`  
  Matches query audio clips against the database of fingerprints to identify the best match using peak-pair hashing and offset consistency analysis.

- `testing.ipynb`  
  Jupyter notebook to run the full pipeline: loading reference tracks, generating fingerprints, and identifying a short query audio clip.

---

## Background

This project implements the algorithm introduced in:

> Wang, A. (2003). *An industrial-strength audio search algorithm*. In Proceedings of the 4th International Conference on Music Information Retrieval (ISMIR).  
> [Link to paper (PDF)](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf)

Key features of the algorithm:
- Converts audio into spectrogram peak "constellations"
- Pairs peaks into hashes using relative timing and frequency offsets
- Uses hash collisions and consistent offset histograms to find the best match

---


