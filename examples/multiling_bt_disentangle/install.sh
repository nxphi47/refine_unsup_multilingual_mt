

# conda install -y -c pytorch faiss-gpu

# pip install sentencepiece
# pip install fairscale
# pip install indic-nlp-library

#  pip install python-Levenshtein
# pip install editdistance
# conda install numba

# editdistance.eval(x, y)
# from Levenshtein import distance as ldistance
# ldistance("012634", "67234")

# cd ..
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# NOTE apex is available in nvcr.io

# export root=/projects/nmt
# export cur=$PWD
# cd $root
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check  --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#   --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#   --global-option="--fast_multihead_attn" ./
# cd $cur


# install other stuff

pip install --editable ./
