set -vx
model=$1
epoch=$2
gcloud compute scp bajrang-10:~/data-science-bowl-2018/models/model-$1-$2.h5 /Users/vivekpandey/kaggle/data-science-bowl-2018/code/models/model-$1.h5
gcloud compute scp bajrang-10:~/data-science-bowl-2018/models/model-$1.json  /Users/vivekpandey/kaggle/data-science-bowl-2018/code/models/

