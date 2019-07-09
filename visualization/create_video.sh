#!/usr/bin/env bash

# Set params
start_time=120
duration=10
input_audio='data/audio_segment.wav'
output_video='ground_truth_1.mp4'

end_time=$(( $start_time + $duration))

# Crop the segment of audio
ffmpeg -ss $start_time -t $duration -i $input_audio audio_segment.wav

# Generate video for gestures from the 3d position
python model_animator.py --start=$start_time --end=$end_time --out='data/temp_gesture_video.mp4'

# Add audio to the video
ffmpeg -i 'data/temp_gesture_video.mp4' -i audio_segment.wav -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k gesture_w_audo.mp4

