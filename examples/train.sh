python3 -u train.py \
	--arch ERFNet \
	--epochs 800 \
	--workers 4 \
	--batch-size 12 \
	--optimizer Adam \
	--lr 1e-3 \
        --lr-decay 0.4 \
        --decay-steps "300,390,480,550,620,670,700" \
	--wd 2e-4 \
	--run-name "erfnet-wd0.0002-decay0.4-first300" \
       	/media/hpc4_Raid/e_burkov/Datasets/CS_small \
> train-log.txt 2>&1

