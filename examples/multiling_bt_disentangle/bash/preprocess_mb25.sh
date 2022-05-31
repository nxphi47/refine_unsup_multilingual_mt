
set -e

# standard mbart cc25 preprocess


CODES=60000     # number of BPE codes
N_THREADS=30    # number of threads in data preprocessing

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --langs)
    langs="$2"; shift 2;;
  --rawdir)
    rawdir="$2"; shift 2;;
  --outdir)
    outdir="$2"; shift 2;;
  --dict)
    dict="$2"; shift 2;;
  --spm)
    spm="$2"; shift 2;;
  --trainpref)
    trainpref="$2"; shift 2;;
  --validpref)
    validpref="$2"; shift 2;;
  --testpref)
    testpref="$2"; shift 2;;
  --nobinarize)
    nobinarize="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ ! -f "$dict" ]; then echo "cannot locate dict: ${dict}"; exit; fi
if [ ! -f "$spm" ]; then echo "cannot locate spm: ${spm}"; exit; fi
if [ ! -d "$rawdir" ]; then echo "cannot locate rawdir: ${rawdir}"; exit; fi

if [ "$trainpref" == "" ]; then trainpref=train; fi
if [ "$testpref" == "" ]; then testpref=test; fi
if [ "$validpref" == "" ]; then validpref=valid; fi

# declare files
export CUR=$PWD

export SPM=${CUR}/../scripts/spm_encode.py
if [ ! -f "$SPM" ]; then echo "cannot locate SPM file: ${SPM}"; exit; fi


CC25_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

# expect langs: en_XX,fr_XX.....
# cc25_c100m: en_xx,fr_XX,es_XX,de_DE,hi_IN,ne_NP,si_LK,gu_IN

echo "=========================================="
echo "langs=${langs}"
echo "rawdir=${rawdir}"
echo "outdir=${outdir}"
echo "dict=${dict}"
echo "spm=${spm}"
echo "=========================================="

IFS=', ' read -r -a xlangs <<< "$langs"
# for element in "${array[@]}"; do echo "$element" ;done
# for index in "${!array[@]}"; do echo "$index ${array[index]}" ;done

mkdir -p ${outdir}
cp -r ${dict} ${outdir}

for xlang in "${xlangs[@]}"; do
    lang=${xlang:0:2}
    echo "=== process lang ${lang}, xlang=${xlang}"
    rawtxt=${rawdir}/${lang}.txt
    outtxt=${outdir}/${trainpref}.${xlang}
    if [ -f ${rawtxt} ]; then
        cp -r ${dict} ${outdir}/dict.${xlang}.txt
        if [ ! -f ${outtxt} ]; then
            echo "spm process ${rawtxt} -> ${outtxt}"
            python ${SPM} --model=${spm} --input ${rawtxt} --out ${outtxt}
        else
            echo "file ${outdir}/${trainpref}.${xlang} exists, not spm it again"
        fi
    else
        echo "file raw txt not found: ${rawtxt}"
    fi
done

# exit 0
if [ ${nobinarize} -eq 1 ]; then
  echo "skip binarization"
else
  export destdir=${outdir}/bin
  echo "Start fairseq-process at ${destdir}"
  mkdir -p ${destdir}
  cp -r ${outdir}/dict* ${destdir}/

  for xlang in "${xlangs[@]}"; do
      echo "start fairseq preprocess for ${xlang}"
      fairseq-preprocess \
      --source-lang ${xlang} \
      --target-lang ${xlang} \
      --trainpref ${outdir}/$trainpref \
      --destdir ${destdir} \
      --thresholdtgt 0 \
      --thresholdsrc 0 \
      --srcdict ${dict} \
      --only-source \
      --workers 8
  done
fi


# ########################################################
export check=0

if [ ${check} -eq 1 ]; then

# testing with some small data
root=/projects/nmt
rawdir=${root}/raw_data/cc25_c100m
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model
langs=gu_IN,ne_NP

langs=my_MM,gu_IN
outdir=${root}/data_fairseq/test_mono_cc25_mygu
cd /projects/nmt/fairseq-py/examples/multiling_bt/bash
bash preprocess_mb25.sh --langs ${langs} --rawdir ${rawdir} --outdir ${outdir} --dict ${dict} --spm ${spm}


# euro_indic monolingual data
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model

langs=en_XX,fr_XX,es_XX,de_DE,hi_IN,ne_NP,si_LK,gu_IN
rawdir=${root}/raw_data/cc25_c100m
outdir=${root}/data_fairseq/cc25_c100m_mono_enfresde_hinesigu

rawdir=${root}/raw_data/cc25_c100m
outdir=${root}/data_fairseq/dev-cc25_c100m_mono_enfresde_hinesigu

cd /projects/nmt/fairseq-py/examples/multiling_bt/bash
bash preprocess_mb25.sh --langs ${langs} --rawdir ${rawdir} --outdir ${outdir} --dict ${dict} --spm ${spm}

for v in valid test ; do cp -r /projects/nmt/data_fairseq/dev-ne_NP-en_XX-flores/${v}.ne_NP-en_XX.* ../cc25_c100m_mono_enfresde_hinesigu/bin/ ; done
for v in valid test ; do cp -r /projects/nmt/data_fairseq/dev-si_LK-en_XX-flores/${v}.si_LK-en_XX.* ../cc25_c100m_mono_enfresde_hinesigu/bin/ ; done


# TODO: uralic --- preprocessing and binarizing ------------------------------------
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model

langs=fi_FI,et_EE,lv_LV
rawdir=${root}/raw_data/cc25_c100m
outdir=${root}/data_fairseq/cc25_c100m_mono_enfietlv

cd /projects/nmt/fairseq-py/examples/multiling_bt/bash
bash preprocess_mb25.sh --langs ${langs} --rawdir ${rawdir} --outdir ${outdir} --dict ${dict} --spm ${spm}

# uralic - filtering first
export root=/projects/nmt
export rawdir=${root}/data_fairseq/cc25_c100m_mono_enfietlv
export outdir=${rawdir}/filu
mkdir -p ${outdir}
cp -r ${rawdir}/dict* ${outdir}/
cd ${root}/fairseq-py
for lang in fi_FI et_EE lv_LV; do
python examples/multiling_bt/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,alphaunicoderatio_70 
done

# uralic - binarizing
langs=fi_FI,et_EE,lv_LV

IFS=', ' read -r -a xlangs <<< "$langs"
export outdir=/projects/nmt/data_fairseq/cc25_c100m_mono_enfietlv
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


export test_dir=/projects/nmt/raw_data/wmt17_test
export src=en
export tgt=fi
export tgt=lv
export tgt=et
export srcf=${test_dir}/newstest2017-${src}${tgt}-src.${src}.sgm
export tgtf=${test_dir}/newstest2017-${src}${tgt}-ref.${tgt}.sgm
export srcfx=${test_dir}/test-${src}${tgt}.${src}
export tgtfx=${test_dir}/test-${src}${tgt}.${tgt}
grep '<seg id' ${srcf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${srcfx}
grep '<seg id' ${tgtf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${tgtfx}

# 
export src=en
export tgt=fi
export tgt=lv
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model
export test_out_dir=${test_dir}/fromen
export SPM=${root}/fairseq-py/examples/multiling_bt/scripts/spm_encode.py
CC25_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
for l in ${src} ${tgt}; do
  export rawtxt=${test_out_dir}/test-${src}${tgt}.${l}
  export outtxt=${test_out_dir}/test-${src}${tgt}.spm.${l}
  python ${SPM} --model=${spm} --input ${rawtxt} --out ${outtxt}
done
export destdir=${test_out_dir}/bin-${src}${tgt}
fairseq-preprocess \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --testpref ${test_out_dir}/test-${src}${tgt}.spm \
    --validpref ${test_out_dir}/test-${src}${tgt}.spm \
    --destdir ${destdir} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${dict} \
    --tgtdict ${dict} \
    --workers 8

# fromen -> cc
# wmt18
export test_dir=/projects/nmt/raw_data/wmt18_test
export src=en
export tgt=et
export srcf=${test_dir}/newstest2018-${src}${tgt}-src.${src}.sgm
export tgtf=${test_dir}/newstest2018-${src}${tgt}-ref.${tgt}.sgm
export srcfx=${test_dir}/test-${src}${tgt}.${src}
export tgtfx=${test_dir}/test-${src}${tgt}.${tgt}
grep '<seg id' ${srcf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${srcfx}
grep '<seg id' ${tgtf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${tgtfx}

# 
export src=en
export tgt=et
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model
export test_out_dir=${test_dir}/fromen
export SPM=${root}/fairseq-py/examples/multiling_bt/scripts/spm_encode.py
CC25_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
for l in ${src} ${tgt}; do
  export rawtxt=${test_out_dir}/test-${src}${tgt}.${l}
  export outtxt=${test_out_dir}/test-${src}${tgt}.spm.${l}
  python ${SPM} --model=${spm} --input ${rawtxt} --out ${outtxt}
done
export destdir=${test_out_dir}/bin-${src}${tgt}
fairseq-preprocess \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --testpref ${test_out_dir}/test-${src}${tgt}.spm \
    --validpref ${test_out_dir}/test-${src}${tgt}.spm \
    --destdir ${destdir} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${dict} \
    --tgtdict ${dict} \
    --workers 8


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# TODO: turkic languages -----------------------------------------------------------------
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model

# langs=tr_TR,kk_KZ
langs=kk_KZ
rawdir=${root}/raw_data/cc25_c100m
outdir=${root}/data_fairseq/cc25_c100m_mono_entrkk

cd /projects/nmt/fairseq-py/examples/multiling_bt/bash
bash preprocess_mb25.sh --langs ${langs} --rawdir ${rawdir} --outdir ${outdir} --dict ${dict} --spm ${spm} --nobinarize 1

# uralic - filtering first
export root=/projects/nmt
export rawdir=${root}/data_fairseq/cc25_c100m_mono_entrkk
export outdir=${rawdir}/filu
mkdir -p ${outdir}
cp -r ${rawdir}/dict* ${outdir}/
cd ${root}/fairseq-py
for lang in kk_KZ; do
python /projects/nmt/fairseq-py/examples/multiling_bt/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,alphaunicoderatio_70 
done

langs=kk_KZ
IFS=', ' read -r -a xlangs <<< "$langs"
export outdir=/projects/nmt/data_fairseq/cc25_c100m_mono_entrkk
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



# test set wmt19
export test_dir=/projects/nmt/raw_data/wmt19_test
export src=en
export tgt=kk
mkdir -p ${test_dir}/fromen
export srcf=${test_dir}/newstest2019-${src}${tgt}-src.${src}.sgm
export tgtf=${test_dir}/newstest2019-${src}${tgt}-ref.${tgt}.sgm
export srcfx=${test_dir}/fromen/test-${src}${tgt}.${src}
export tgtfx=${test_dir}/fromen/test-${src}${tgt}.${tgt}
grep '<seg id' ${srcf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${srcfx}
grep '<seg id' ${tgtf} | sed -e 's/<seg id="[0-9]*">\s*//g' | sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > ${tgtfx}
root=/projects/nmt
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
spm=${root}/pret_models/mbart.cc25.v2/sentence.bpe.model
export test_out_dir=${test_dir}/fromen
export SPM=${root}/fairseq-py/examples/multiling_bt/scripts/spm_encode.py
CC25_LANGS=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
for l in ${src} ${tgt}; do
  export rawtxt=${test_out_dir}/test-${src}${tgt}.${l}
  export outtxt=${test_out_dir}/test-${src}${tgt}.spm.${l}
  python ${SPM} --model=${spm} --input ${rawtxt} --out ${outtxt}
done
export destdir=${test_out_dir}/bin-${src}${tgt}
fairseq-preprocess \
    --source-lang ${src} \
    --target-lang ${tgt} \
    --testpref ${test_out_dir}/test-${src}${tgt}.spm \
    --validpref ${test_out_dir}/test-${src}${tgt}.spm \
    --destdir ${destdir} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${dict} \
    --tgtdict ${dict} \
    --workers 8


# ----------------------------------------------------------------------------------------




# filtered
root=/projects/nmt
rawdir=${root}/raw_data/cc25_c100m
dict=${root}/pret_models/mbart.cc25.v2/dict.txt


# missing gu hi
# export langs=en_XX,fr_XX,es_XX,de_DE,ne_NP,si_LK,
export langs=en_XX,fr_XX,es_XX,de_DE,hi_IN,ne_NP,si_LK,gu_IN
export langs=ne_NP
export langs=si_LK
export langs=gu_IN
export langs=en_XX
export langs=hi_IN
IFS=', ' read -r -a xlangs <<< "$langs"
export outdir=/projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu
# export rawdir=${outdir}/fil
# export destdir=${outdir}/bin_filfix
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

# copy
cp -r /projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu/bin_filfix/test* /projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu/bin_filu/
cp -r /projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu/bin_filfix/valid* /projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu/bin_filu/




root=/projects/nmt
rawdir=${root}/raw_data/cc25_c100m
dict=${root}/pret_models/mbart.cc25.v2/dict.txt
export outdir=/projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu

export src=ne_NP
export tgt=en_XX
export dname=euroindia_all.index.out.ne_NP.en_XX.12271.3
export dname=euroindia_all.index.out.ne_NP.en_XX.2014.2
export dname=euroindia_all.index.out.ne_NP.en_XX.467364.10
export dname=euroindia_all.index.out.ne_NP.en_XX.88754.5
export dname=euroindia_all.index.out.ne_NP.en_XX.2949.0.2.True.uplTrue
export dname=euroindia_all.index.out.ne_NP.en_XX.331.0.1.True.uplTrue
export dname=euroindia_all.index.out.ne_NP.en_XX.85965.0.5.True.uplTrue
export src=si_LK
export tgt=en_XX
# export dname=euroindia_all.index.out.si_LK.en_XX.219253.5
# export dname=euroindia_all.index.out.si_LK.en_XX.29407.3
# export dname=euroindia_all.index.out.si_LK.en_XX.4160.2
# export dname=euroindia_all.index.out.si_LK.en_XX.923494.10
export dname=euroindia_all.index.out.si_LK.en_XX.224157.0.5.True.uplTrue
export dname=euroindia_all.index.out.si_LK.en_XX.481.0.1.True.uplTrue
export dname=euroindia_all.index.out.si_LK.en_XX.6187.0.2.True.uplTrue
export src=gu_IN
export tgt=en_XX
export dname=euroindia_all.index.out.gu_IN.en_XX.4551.0.2.True.uplTrue
export src=hi_IN
export tgt=en_XX
export dname=euroindia_all.index.out.hi_IN.en_XX.23365.0.2.True.uplTrue
export dname=euroindia_all.index.out.hi_IN.en_XX.39841.0.2.True.uplTrue

export outdir=/projects/nmt/data_fairseq/cc25_c100m_mono_enfresde_hinesigu/filu/${dname}
export rawdir=${outdir}
export destdir=${outdir}/bin

echo "Start fairseq-process at ${destdir}"
mkdir -p ${destdir}
cp -r ${outdir}/dict* ${destdir}/

fairseq-preprocess \
--source-lang ${src} \
--target-lang ${tgt} \
--trainpref ${rawdir}/train \
--destdir ${destdir} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${dict} \
--tgtdict ${dict} \
--workers 8


export sname=0.5.True.uplTrue
export dname=euroindia_all.index.com.ne_NPsi_LK.en_XX.${sname}
mkdir -p ${dname}
cp -r euroindia_all.index.out*${sname}/train* ${dname}




# preprocess uralic
export n=100000000
export rawdir=/projects/nmt/raw_data/cc25/
export datadir=/projects/nmt/raw_data/cc25_c100m/
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# mkdir -p ${datadir}

# need to shuf hi as well
for lang in tr; do
echo "shuf ${lang}.... ${n}"
shuf --random-source=<(get_seeded_random 42) -n ${n} ${rawdir}/${lang}.txt > ${datadir}/${lang}.txt
done


# need to shuf tr


fi
