#!/bin/bash


set -e
source /home/ethanrao/anaconda3/bin/activate TCAN

bash train.sh JIGSAWS .Knot_Tying.LOUO.B 100
echo "JIGSAWS_LOUO_B"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.B 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.C 100
echo "JIGSAWS_LOUO_C"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.C 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.D 100
echo "JIGSAWS_LOUO_D"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.D 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.E 100
echo "JIGSAWS_LOUO_E"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.E 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.F 100
echo "JIGSAWS_LOUO_F"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.F 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.G 100
echo "JIGSAWS_LOUO_G"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.G 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.H 100
echo "JIGSAWS_LOUO_H"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.H 100

bash train.sh JIGSAWS .Knot_Tying.LOUO.I 100
echo "JIGSAWS_LOUO_I"
bash test_epoch.sh JIGSAWS .Knot_Tying.LOUO.I 100
