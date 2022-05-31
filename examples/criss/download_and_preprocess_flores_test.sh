#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

CUR=$PWD
ROOT=$PWD/../../..
FLORES=${ROOT}/flores
SPM_ENCODE=${FLORES}/floresv1/scripts/spm_encode.py
# DATA=${FLORES}/data_tmp
DATA=${ROOT}/flores/floresv1/data
SPM_MODEL=${ROOT}/pret_models/criss/criss_checkpoints/sentence.bpe.model
DICT=${ROOT}/pret_models/criss/criss_checkpoints/dict.txt

download_data() {
  CORPORA=$1
  URL=$2

  if [ -f $CORPORA ]; then
    echo "$CORPORA already exists, skipping download"
  else
    echo "Downloading $URL"
    wget $URL -O $CORPORA --no-check-certificate || rm -f $CORPORA
    if [ -f $CORPORA ]; then
      echo "$URL successfully downloaded."
    else
      echo "$URL not successfully downloaded."
      rm -f $CORPORA
    fi
  fi
}

# if [[ -f ${FLORES} ]]; then
#   echo "flores already cloned"
# else
#   cd $ROOT
#   git clone https://github.com/facebookresearch/flores
#   cd $CUR
# fi

# mkdir -p $DATA
# download_data $DATA/wikipedia_en_ne_si_test_sets.tgz "https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz"
# pushd $DATA
# pwd
# tar -vxf wikipedia_en_ne_si_test_sets.tgz
# popd


for lang in ne_NP si_LK; do
  datadir=$DATA/${lang}-en_XX-flores
  rm -rf $datadir
  mkdir -p $datadir
  VALID_PREFIX=$DATA/wikipedia_en_ne_si_test_sets/wikipedia.dev
  TEST_PREFIX=$DATA/wikipedia_en_ne_si_test_sets/wikipedia.test
  echo "encode ${TEST_PREFIX}"
  python $SPM_ENCODE \
    --model ${SPM_MODEL} \
    --output_format=piece \
    --inputs ${TEST_PREFIX}.${lang:0:2}-en.${lang:0:2} ${TEST_PREFIX}.${lang:0:2}-en.en \ 
    --outputs $datadir/test.bpe.${lang}-en_XX.${lang} $datadir/test.bpe.${lang}-en_XX.en_XX
  
  echo "encode ${VALID_PREFIX}"
  python $SPM_ENCODE \
    --model ${SPM_MODEL} \
    --output_format=piece \
    --inputs ${VALID_PREFIX}.${lang:0:2}-en.${lang:0:2} ${VALID_PREFIX}.${lang:0:2}-en.en \
    --outputs $datadir/valid.bpe.${lang}-en_XX.${lang} $datadir/valid.bpe.${lang}-en_XX.en_XX

  # binarize data
  echo "preprocess ${TEST_PREFIX}"
  fairseq-preprocess \
    --source-lang ${lang} --target-lang en_XX \
    --testpref $datadir/test.bpe.${lang}-en_XX \
    --validpref $datadir/valid.bpe.${lang}-en_XX \
    --destdir $datadir \
    --srcdict ${DICT} \
    --joined-dictionary \
    --workers 4
done
