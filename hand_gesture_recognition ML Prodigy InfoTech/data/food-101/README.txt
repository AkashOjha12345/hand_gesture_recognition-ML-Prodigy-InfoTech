Place the Kaggle Food-101 dataset in this structure:

food-101/
  images/
    apple_pie/
    baby_back_ribs/
    ...
  meta/
    train.txt
    test.txt

Example commands:
python main.py init-calories
python main.py train --epochs 10 --batch-size 32
python main.py predict path\to\food.jpg
python main.py webcam
