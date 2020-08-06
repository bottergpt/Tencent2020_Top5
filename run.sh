!/bin/bash
bash preprocess/run.sh &&
bash get_emb/run.sh &&
bash model/run.sh &&
bash oof/run.sh &&
bash ensemble/run.sh &&
echo "all done!"
