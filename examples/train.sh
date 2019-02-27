python3 -u train.py \
	--arch ENet \
	--epochs 1 \
	--workers 1 \
	--batch-size 1 \
	--optimizer Adam \
	--lr 1e-3 \
	--wd 1e-4 \
	--run-name "test-run" \
	/home/shrubb/Datasets/Cityscapes \
# > train-log.txt 2>&1
	# --resume checkpoint.pth.tar \