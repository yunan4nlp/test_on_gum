export CUDA_VISIBLE_DEVICES=7

touch log
nohup python -u driver/TestOnGum.py --config_file experiment/rst_model/config.cfg  > log 2>&1 &
tail -f log
