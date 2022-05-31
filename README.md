## Refining Low-Resource Unsupervised Translation by Language Disentanglement of Multilingual Translation Model

This readme provide instruction to run the main experiments in fully unsupervised translation of low-resource languages

Pre-trained models will be release upon the paper is accepted and/or made public through arxiv.

## Running experiments


### Step 0: preparation

Assume root directory is `~/`, repo folder is saved in `~/fairseq-py`.

```bash
# download pre-trained models

mkdir -p ~/pret_models
cd ~/pret_models
wget https://dl.fbaipublicfiles.com/criss/criss_3rd_checkpoints.tar.gz
tar -xf criss_checkpoints.tar.gz

# download raw data
mkdir -p ~/raw_data/cc25
cd ~/raw_data/cc25
bash ~/fairseq-py/examples/multiling_bt_disentangle/scripts/download_cc100.sh ~/fairseq-py/examples/multiling_bt_disentangle/scripts/cc100_download_paths_cc25.txt 
for l in en ne si hi gu; do
    unxz ${l}.txt.xz
done
# filter each upto 100 lines
mkdir -p ~/raw_data/cc25_c100m
for l in ne si gu; do
    cp -r ~/raw_data/cc25/${l}.txt ~/raw_data/cc25_c100m/${l}.txt
done
for l in en hi; do
    shuf -n 100000000 ~/raw_data/cc25/${l}.txt > ~/raw_data/cc25_c100m/${l}.txt
done

```

### Step 1: pre-processing

```bash

# pre-processing
dict=~/pret_models/criss/criss_checkpoints/dict.txt
spm=~/pret_models/criss/criss_checkpoints/sentence.bpe.model
langs=en_XX,ne_NP,si_LK,hi_IN,gu_IN
rawdir=~/raw_data/cc25_c100m
outdir=~/data_fairseq/cc25_c100m_mono_en_indic
mkdir -p ~/data_fairseq/cc25_c100m_mono_en_indic
cd /projects/nmt/fairseq-py/examples/multiling_bt_disentangle/bash
bash preprocess_mb25.sh --langs ${langs} --rawdir ${rawdir} --outdir ${outdir} --dict ${dict} --spm ${spm}

# filtering raw data
export rawdir=~/data_fairseq/cc25_c100m_mono_en_indic
export outdir=${rawdir}/filu
mkdir -p ${outdir}
cp -r ${rawdir}/dict* ${outdir}/
cd ~/fairseq-py

bash examples/multiling_bt_disentangle/scripts/agg_filter.sh $rawdir $outdir

# binarizing
IFS=', ' read -r -a xlangs <<< "$langs"
export outdir=~/data_fairseq/cc25_c100m_mono_en_indic
export rawdir=${outdir}/filu
export destdir=${outdir}/bin_filu
echo "Start fairseq-process at ${destdir}"
mkdir -p ${destdir}
cp -r ${outdir}/dict* ${destdir}/
for xlang in "${xlangs[@]}"; do
    echo "start fairseq preprocess for ${xlang}, ${rawdir}/train --> ${destdir}"
    echo "$(wc -l ${rawdir}/train.${xlang})"
    fairseq-preprocess \
    --source-lang ${xlang} \
    --target-lang ${xlang} \
    --trainpref ${rawdir}/train \
    --destdir ${destdir} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${dict} \
    --only-source \
    --workers 8
done


# finally we have monolingual data store in ~/data_fairseq/cc25_c100m_mono_en_indic/bin_filu
# we can now start traiing

```


### Step 2: Finetune CRISS with language disentanglement refinement


#### Step 2.1 - Stage 1 - Multilingual back-translation

```bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROOT_DIR=~
cd ${ROOT_DIR}/fairseq-py/examples/multiling_bt_disentangle/sweeps
python swp_mling_bt_disentangle_indic.py --stage 1

# stop training by:
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done
```

#### Step 2.2 - Stage 1 - English disentanglement

```bash

# kill all previous processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROOT_DIR=~
cd ${ROOT_DIR}/fairseq-py/examples/multiling_bt_disentangle/sweeps
# to indic
python swp_mling_bt_disentangle_indic.py --stage 2
# after finishing, kill processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

# to english
python swp_mling_bt_disentangle_indic.py --stage 2 --to_english
# after finishing, kill processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

```


#### Step 2.3 - Stage 3 - Reinforcement

```bash

# kill all previous processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROOT_DIR=~
cd ${ROOT_DIR}/fairseq-py/examples/multiling_bt_disentangle/sweeps
# to indic
python swp_mling_bt_disentangle_indic.py --stage 3
# after finishing, kill processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

# to english
python swp_mling_bt_disentangle_indic.py --stage 3 --to_english
# after finishing, kill processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

```


#### Step 2.4 - Stage 4 - Target low-resource disentanglement

```bash

# kill all previous processes
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ROOT_DIR=~
cd ${ROOT_DIR}/fairseq-py/examples/multiling_bt_disentangle/sweeps

python swp_mling_bt_disentangle_indic.py --stage 4 --target_lang ne_NP
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

python swp_mling_bt_disentangle_indic.py --stage 4 --target_lang si_LK
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

python swp_mling_bt_disentangle_indic.py --stage 4 --target_lang gu_IN
for i in $(pgrep -f python); do echo $i && kill -9 $i ; done

```

### Step 3: inference

Prepare before inference

```bash

cd ~/
git clone https://github.com/facebookresearch/flores.git
mkdir -p ~/tools
cd ~/tools
git clone https://github.com/moses-smt/mosesdecoder.git
export root=~

# NOTE: preprocess test set similar to step 0-1 and store them into ~//data_fairseq/cc25_c100m_mono_en_indic/bin_filu as test set

```

Inference en->ne

```bash

# suppose en-ne model is store at ~/train_fairseq/c25c100indic-u/mb25.gft3toIn.cac.gft3toIn.tmuse.en_ne_tone.mtok12h.ufre2.4gpu
# infer_criss.sh run both forward (target_lang -> english) and backward (english - >target_lang)
# to run only english->target_lang, need to pass --no_fwd 1 (no forward)

export CUDA_VISIBLE_DEVICES=1
export root=~
export data=${root}/data_fairseq/cc25_c100m_mono_en_indic/bin_filu
export exp_dir=${root}/train_fairseq
export ckptdir=${exp_dir}/train_fairseq/c25c100indic-u/mb25.gft3toIn.cac.gft3toIn.tmuse.en_ne_tone.mtok12h.ufre2.4gpu
export maxtoks=15000
export lp=0.1
export src=ne_NP
export ovrw_ckpt=
bash infer_criss.sh --src ${src} --lst ${lp} --lts ${lp} --no_fwd 1
