echo "`root` required, please run outside of the container"

cd src/prep/SimECG-M/linux

chmod +x ecgsyn

python gen_sample.py

cd ..

python convert_to_pickle.py