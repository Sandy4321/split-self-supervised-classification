python3 /apdcephfs/private_gaoyilei/ft_local/AET_torch/train_unsupervised.py --cuda --outf  /apdcephfs/private_gaoyilei/ft_local/AET_torch/model_result/ --dataroot  /apdcephfs/private_gaoyilei/ft_local/AET_torch/dataset/



python3 /apdcephfs/private_gaoyilei/ft_local/AET_torch/classifier.py --dataroot




python classification.py --dataroot $YOUR_CIFAR10_PATH$ --epochs 200 --schedule 100 150 --gamma 0.1 -c ./output_cls --net ./output/net_epoch_1499.pth --gpu-id 0

python3 train_unsupervised.py --cuda --outf  ./model_result/ --dataroot  ./dataset/

python3 classifier.py --dataroot ./dataset --epochs 200 --schedule 100 150 --gamma 0.1 -c ./output_cls --net ./model_result/net_epoch_1350.pth --gpu-id 2
