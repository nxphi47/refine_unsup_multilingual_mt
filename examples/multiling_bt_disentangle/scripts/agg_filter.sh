
rawdir=$1
outdir=$2


lang=en_XX && python examples/multiling_bt_disentangle/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,alphaunicoderatio_70 

lang=ne_NP && python examples/multiling_bt_disentangle/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,vocabtop_80,vocabfreq_2,devanagariratio_80 

lang=si_LK && python examples/multiling_bt_disentangle/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,vocabtop_80,vocabfreq_2,sinhalaratio_80 

lang=hi_IN && python examples/multiling_bt_disentangle/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,vocabtop_80,vocabfreq_2,devanagariratio_80 

lang=gu_IN && python examples/multiling_bt_disentangle/scripts/flores_agg_filter.py \
    --input $rawdir/train.${lang} \
    --output $outdir/train.${lang} \
    --filter unique_sent,vocabtop_80,vocabfreq_2,gujaratiratio_80 
