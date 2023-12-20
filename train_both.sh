echo "training lung model"

python -m entrypoint --cancer lung --train --data_dir ./datasets/train_lung/

echo "training prostate model"

python -m entrypoint --cancer prostate --train --data_dir ./datasets/train_prostate/train/

