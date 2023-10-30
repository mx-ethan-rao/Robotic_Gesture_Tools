#!/bin/bash


set -e
source /home/ethanrao/anaconda3/bin/activate TCAN

bash train.sh JIGSAWS .Suturing.LOUO.B /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_B"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.B 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.C /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_C"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.C 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.D /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_D"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.D 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.E /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_E"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.E 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.F /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_F"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.F 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.G /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_G"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.G 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.H /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_H"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.H 100 /data/mingxing/MS_TCN_data/

bash train.sh JIGSAWS .Suturing.LOUO.I /data/mingxing/MS_TCN_data/
echo "JIGSAWS_LOUO_I"
bash test_epoch.sh JIGSAWS .Suturing.LOUO.I 100 /data/mingxing/MS_TCN_data/


# bash train.sh JIGSAWS .Suturing.LOSO.1 100
# echo "JIGSAWS_LOSO_1"
# bash test_epoch.sh JIGSAWS .Suturing.LOSO.1 100

# bash train.sh JIGSAWS .Suturing.LOSO.2 100
# echo "JIGSAWS_LOSO_2"
# bash test_epoch.sh JIGSAWS .Suturing.LOSO.2 100

# bash train.sh JIGSAWS .Suturing.LOSO.3 100
# echo "JIGSAWS_LOSO_3"
# bash test_epoch.sh JIGSAWS .Suturing.LOSO.3 100

# bash train.sh JIGSAWS .Suturing.LOSO.4 100
# echo "JIGSAWS_LOSO_4"
# bash test_epoch.sh JIGSAWS .Suturing.LOSO.4 100

# bash train.sh JIGSAWS .Suturing.LOSO.5 100
# echo "JIGSAWS_LOSO_5"
# bash test_epoch.sh JIGSAWS .Suturing.LOSO.5 100