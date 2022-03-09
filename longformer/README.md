# Long text token classification using LongFormer

The data comes from: https://www.kaggle.com/c/feedback-prize-2021/

To train the model for 5 folds, you can run:

    python train_sep.py --fold 0 --model allenai/longformer-base-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 3 --valid_batch_size 3

    python train_cls.py --fold 0 --model allenai/longformer-base-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 3 --valid_batch_size 3

    python infer_sep.py --fold 0 --model allenai/longformer-base-4096 --max_len 1536 --valid_batch_size 3

    python infer_cls.py --fold 0 --model allenai/longformer-base-4096 --max_len 1536 --valid_batch_size 3

Note that you need have `kfold` column in training data.
