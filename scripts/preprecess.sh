#!/bin/bash

wav_file_list=($(ls *.wav))

for (( i = 0; i < ${#wav_file_list[@]}   ; i++ )); do
  echo ${wav_file_list[i]}
  ffmpeg -i ${wav_file_list[i]} -acodec pcm_s16le -ar 16000 -ac 1 p_${wav_file_list[i]}
  rm ${wav_file_list[i]}
done