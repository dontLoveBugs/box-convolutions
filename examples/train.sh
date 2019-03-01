python3 -u train.py \
	--arch ERFNet \
	--epochs 600 \
	--workers 6 \
	--batch-size 12 \
	--optimizer Adam \
	--lr 1e-3 \
	--wd 2e-4 \
	--run-name "erfnet-wd0.0002-decay0.35" \
       	/media/hpc4_Raid/e_burkov/Datasets/CS_small \
> train-log.txt 2>&1
 #--resume runs/enet/model.pth \

