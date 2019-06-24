from logreg import main as lr_model
from gda import main as gda_model


lr_model(train_path='../data/ds1_train.csv',
         valid_path='../data/ds1_valid.csv',
         save_path='output/lr_pred_1.txt')

lr_model(train_path='../data/ds2_train.csv',
         valid_path='../data/ds2_valid.csv',
         save_path='output/lr_pred_2.txt')

gda_model(train_path='../data/ds1_train.csv',
         valid_path='../data/ds1_valid.csv',
         save_path='output/gda_pred_1.txt')

gda_model(train_path='../data/ds2_train.csv',
         valid_path='../data/ds2_valid.csv',
         save_path='output/gda_pred_2.txt')
