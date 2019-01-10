# This script will add noise to the audio file in order to augment the dataset
# It is used in the script "prepare_data.py"

data=$4
for i in `seq ${2} ${3}`;
do
  echo "${i}"
  if [ -e ${data}/${1}/inputs/audio${i}.wav ]
  then 
    sox ${data}/${1}/inputs/audio${i}.wav -p synth whitenoise vol 0.01 | sox -m ${data}/${1}/inputs/audio${i}.wav - ${data}/${1}/inputs/naudio${i}.wav
    echo "naudio generated in ${1} for id ${i}"
  else
    echo "could not generate noisy audio, because original audio at ${data} was not found"
  fi
done
