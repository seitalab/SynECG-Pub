
cd src/ssl_pt/resouces/

python gen_yamls_r04.py

cd ..

python pretrain.py --pt 1 --device cuda:0
python pretrain.py --pt 2 --device cuda:0