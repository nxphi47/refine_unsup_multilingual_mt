
set -e

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --no_fwd)
    no_fwd="$2"; shift 2;;
  --no_bwd)
    no_bwd="$2"; shift 2;;
  --src)
    src="$2"; shift 2;;
  --tgt)
    tgt="$2"; shift 2;;
  --lst)
    lst="$2"; shift 2;;
  --lts)
    lts="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$no_fwd" == "" ]; then no_fwd=0; else no_fwd=1 ; fi
if [ "$no_bwd" == "" ]; then no_bwd=0; else no_bwd=1 ; fi
# 
if [ "$src" != "" ]; then srcx=${src}; fi
if [ "$tgt" != "" ]; then tgtx=${tgt}; fi
if [ "$lst" != "" ]; then lpst=${lst}; fi
if [ "$lts" != "" ]; then lpts=${lts}; fi


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export srcx="${srcx:-ne_NP}"
export tgtx="${tgtx:-en_XX}"
export src=${srcx:0:2} && export tgt=${tgtx:0:2}

export root="${root:-/projects/nmt}"
export data="${data:-${root}/data_fairseq/dev-${srcx}-${tgtx}-flores}"
export spmmodel=${root}/pret_models/criss/criss_checkpoints/sentence.bpe.model
export lang_dict=${root}/pret_models/criss/criss_checkpoints/lang_dict.txt

export task="${task:-translation_multi_simple_epoch}"
export ckpt="${ckpt:-}"
export ovrw_ckpt="${ovrw_ckpt:-}"

export lpst=${lpst:-1}
export lpts=${lpts:-1}
export beam=${beam:-5}
export tfreq=${tfreq:--1}
export prepend_bos=${prepend_bos:-0}

export seed=${seed:-1}

export maxtoks=${maxtoks:-4000}

echo "========================================================================================="
echo "CUDA: ${CUDA_VISIBLE_DEVICES} | src: ${src} - ${tgt}; srcx: ${srcx}-${tgtx} | task: ${task} | lpst: ${lpst}, lpts: ${lpts}, beam: ${beam}, maxtoks: ${maxtoks}"
echo "ckpt: ${ckpt}"
# echo "========================================================================================="
echo "--------------------------------------------------------------------------------"

ROOTDIR=${root}/fairseq-py/examples/multiling_bt/scripts
FLORES_SCRIPTS=${root}/flores/floresv1/scripts

TOOLS_DIR=${root}/tools
MULTIBLEU=${TOOLS_DIR}/mosesdecoder/scripts/generic/multi-bleu.perl
MOSES=${TOOLS_DIR}/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
SCRIPTS=${FLORES_SCRIPTS}

WMT16_SCRIPTS=$TOOLS_DIR/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

INDIC_TOKENIZER_OLD="bash ${FLORES_SCRIPTS}/indic_norm_tok.sh ${src}"
INDIC_TOKENIZER="python $ROOTDIR/indic_tokenize.py  --language ${src} "


function criss_bleu () {
  cgen=$1
  lang=$2
  tail -n 1 ${cgen}
  # echo "lang: ${lang}"
  if [[ "$lang" == "ro" ]]; then
    cat $cgen | grep -P "^T-" | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape ${lang} > $cgen.ref 2> /dev/null
    cat $cgen | grep -P "^H-" | cut -f3 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape ${lang} > $cgen.hyp 2> /dev/null
    echo -ne "Bleu 1 [${lang}] "
    ${MULTIBLEU} $cgen.ref < $cgen.hyp 2> /dev/null
  elif [[ "$lang" == "en" || "$lang" == "fr" || "$lang" == "de" ]]; then
    cat $cgen | grep -P "^T-" | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${lang} > $cgen.ref 2> /dev/null
    cat $cgen | grep -P "^H-" | cut -f3 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${lang} > $cgen.hyp 2> /dev/null
    echo -ne "Bleu 2 [${lang}] "
    ${MULTIBLEU} $cgen.ref < $cgen.hyp 2> /dev/null
    
  elif [[ "$lang" == "hi" || "$lang" == "ne" || "$lang" == "si" || "$lang" == "gu" ]]; then
    INDIC_TOKENIZER_OLD="bash ${FLORES_SCRIPTS}/indic_norm_tok_py3.sh ${lang}"
    INDIC_TOKENIZER="python $ROOTDIR/indic_tokenize.py  --language ${lang} "
    cat $cgen | grep -P "^T-" | cut -f2  > $cgen.ref
    cat $cgen | grep -P "^H-" | cut -f3  > $cgen.hyp
    echo -ne "Bleu 3 [${lang}] "
    # echo "OLD indic tokenzier"
    $INDIC_TOKENIZER_OLD $cgen.ref > $cgen.tok.old.ref
    $INDIC_TOKENIZER_OLD $cgen.hyp > $cgen.tok.old.hyp
    ${MULTIBLEU} $cgen.tok.old.ref < $cgen.tok.old.hyp 2> /dev/null
  else
    cat $cgen | grep -P "^T-" | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${lang} > $cgen.ref 2> /dev/null
    cat $cgen | grep -P "^H-" | cut -f3 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${lang} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${lang} > $cgen.hyp 2> /dev/null
    echo -ne "Bleu 4(2) [${lang}] "
    ${MULTIBLEU} $cgen.ref < $cgen.hyp 2> /dev/null
  fi
}



if [ ${no_fwd} -eq 0 ]; then
    export fgen=$(mktemp /tmp/infer-fwd-script.XXXXXX)
    echo "===== ${srcx} --> ${tgtx} , lpst=${lpst}, beam=${beam} =====, Save in ${fgen}"
    # overwrite_states_path
    if [ "$ovrw_ckpt" != "" ]; then
      echo "Parse overwrite states path ${ovrw_ckpt}"
      fairseq-generate ${data} \
          --path ${ckpt} \
          --overwrite-states-path ${ovrw_ckpt} \
          --task ${task} \
          --gen-subset test \
          --remove-bpe=sentencepiece \
          --source-lang ${srcx} --target-lang ${tgtx} \
          --decoder-langtok \
          --lang-pairs "${srcx}-${tgtx},${tgtx}-${srcx}" \
          --lang-dict ${lang_dict} --lang-tok-style 'mbart' --sampling-method 'temperature' --sampling-temperature '1.0' \
          --seed ${seed} \
          --beam ${beam} \
          --lenpen ${lpst} \
          --sacrebleu --scoring sacrebleu \
          --max-tokens ${maxtoks} > ${fgen}
    else
      fairseq-generate ${data} \
          --path ${ckpt} \
          --task ${task} \
          --gen-subset test \
          --remove-bpe=sentencepiece \
          --source-lang ${srcx} --target-lang ${tgtx} \
          --decoder-langtok \
          --lang-pairs "${srcx}-${tgtx},${tgtx}-${srcx}" \
          --lang-dict ${lang_dict} --lang-tok-style 'mbart' --sampling-method 'temperature' --sampling-temperature '1.0' \
          --seed ${seed} \
          --beam ${beam} \
          --lenpen ${lpst} \
          --sacrebleu --scoring sacrebleu \
          --max-tokens ${maxtoks} > ${fgen}
      fi

        # --skip-invalid-size-inputs-valid-test \
    criss_bleu $fgen ${tgt}
    rm -rf ${fgen}*
fi

# ========================================

if [ ${no_bwd} -eq 0 ]; then
    export bgen=$(mktemp /tmp/infer-bwd-script.XXXXXX)
    echo "===== ${tgtx} --> ${srcx} , lpts=${lpts}, beam=${beam} =====, Save in ${bgen}"
    if [ "$ovrw_ckpt" != "" ]; then
      echo "Parse overwrite states path ${ovrw_ckpt}"
      fairseq-generate ${data} \
          --path ${ckpt} \
          --overwrite-states-path ${ovrw_ckpt} \
          --task ${task}  \
          --gen-subset test \
          --remove-bpe=sentencepiece \
          --source-lang ${tgtx} --target-lang ${srcx} \
          --decoder-langtok \
          --lang-pairs "${srcx}-${tgtx},${tgtx}-${srcx}" \
          --lang-dict ${lang_dict} --lang-tok-style 'mbart' --sampling-method 'temperature' --sampling-temperature '1.0' \
          --seed ${seed} \
          --beam ${beam} \
          --lenpen ${lpts} \
          --sacrebleu --scoring sacrebleu \
          --max-tokens ${maxtoks} > ${bgen}
    else
      fairseq-generate ${data} \
          --path ${ckpt} \
          --task ${task}  \
          --gen-subset test \
          --remove-bpe=sentencepiece \
          --source-lang ${tgtx} --target-lang ${srcx} \
          --decoder-langtok \
          --lang-pairs "${srcx}-${tgtx},${tgtx}-${srcx}" \
          --lang-dict ${lang_dict} --lang-tok-style 'mbart' --sampling-method 'temperature' --sampling-temperature '1.0' \
          --seed ${seed} \
          --beam ${beam} \
          --lenpen ${lpts} \
          --sacrebleu --scoring sacrebleu \
          --max-tokens ${maxtoks} > ${bgen}
    fi
        # --skip-invalid-size-inputs-valid-test \
    criss_bleu $bgen ${src}
    rm -rf ${bgen}*
fi


