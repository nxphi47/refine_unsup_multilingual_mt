from multiprocessing.sharedctypes import Value
import sweep
from sweep import HyperParam
import os
import json
import itertools
from pretraineds import get_pretrained_mbart_models, pretrain_mbart_models, aug_datasets
from bt_directions import get_bt_direction, get_rank_to_langs
from sweep_libs import lambda_to_name
import hashlib

N_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))

ROOT_DIR = os.environ.get("ROOT_DIR", "~")
print(f"{ROOT_DIR=}")
print(f"{N_GPUS=}")


def hash_path_to_int(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8

TARGET_LANGS = ['ne_NP', 'si_LK', 'hi_IN', 'gu_IN']
STAGE = 0
TO_ENGLISH = False
TARGET_LANG = None
SHARD_ID = 0

PRET_MODEL_NAME = ""
PRET_CACHED_NAME = ""

def get_stage_direction(args):
    global STAGE, TO_ENGLISH, BT_DIRECTION_NAME, TARGET_LANG, TARGET_LANGS, SHARD_ID
    global PRET_MODEL_NAME, PRET_CACHED_NAME

    STAGE = args.stage
    TO_ENGLISH = args.to_english
    TARGET_LANG = args.target_lang

    if STAGE == 1:
        BT_DIRECTION_NAME = "ennehisigu"
        PRET_MODEL_NAME = "cri"
        PRET_CACHED_NAME = ""
    elif STAGE == 2 or STAGE == 3:
        if TO_ENGLISH:
            BT_DIRECTION_NAME = "en_indi_toen"
        else:
            BT_DIRECTION_NAME = "en_indi_toindi"
        if STAGE == 2:
            PRET_MODEL_NAME = "gft1"
            PRET_CACHED_NAME = "gft1"
        else:
            if TO_ENGLISH:
                PRET_MODEL_NAME = "gft2toEn"
                PRET_CACHED_NAME = "gft2toIn"
            else:
                PRET_MODEL_NAME = "gft2toIn"
                PRET_CACHED_NAME = "gft2toEn"
    elif STAGE == 4:
        assert TARGET_LANG in TARGET_LANGS
        SHARD_ID = TARGET_LANGS.index(TARGET_LANG)
        assert not TO_ENGLISH
        BT_DIRECTION_NAME = f'en_{TARGET_LANG[:2]}_to{TARGET_LANG[:2]}'
        PRET_MODEL_NAME = "gft3toIn"
        PRET_CACHED_NAME = "gft3toEn"
    else:
        raise ValueError



def get_mbart_params():
    # pre-trained models
    if STAGE == 1:
        pret_model = get_pretrained_mbart_models(ROOT_DIR, PRET_MODEL_NAME, "criss")
        cached_params = []
    elif STAGE in [2, 3]:
        pret_model = get_pretrained_mbart_models(ROOT_DIR, PRET_MODEL_NAME, "criss")
        cached_model = get_pretrained_mbart_models(ROOT_DIR, PRET_CACHED_NAME, "criss")
        cached_shard_model = cached_model.replace(".pt", "") + f".sharded_states_r$.pt"
        cached_params = [
            HyperParam("--cache-model-path", cached_model, save_key_fn=lambda x: f"cac.{PRET_CACHED_NAME}"),
            HyperParam("--cache-overwrite-sharded-path", cached_shard_model)
        ]
        assert os.path.exists(cached_model), f"{cached_model} not exists."
    elif STAGE == 4:
        pret_model = get_pretrained_mbart_models(ROOT_DIR, PRET_MODEL_NAME, "criss")
        cached_model = get_pretrained_mbart_models(ROOT_DIR, PRET_CACHED_NAME, "criss")
        cached_shard_model = cached_model.replace(".pt", "") + f".sharded_states_r{SHARD_ID}.pt"
        cached_params = [
            HyperParam("--cache-model-path", cached_model, save_key_fn=lambda x: f"cac.{PRET_CACHED_NAME}"),
            HyperParam("--cache-overwrite-sharded-path", cached_shard_model)
        ]
        assert os.path.exists(cached_model), f"{cached_model} not exists."
    else:
        raise ValueError
    
    # model names
    if STAGE == 4:
        model_params = [
            HyperParam("--arch", "mbart_large", save_key_fn=lambda x: "mb25"),
        ]
    else:
        model_params = [
            HyperParam("--arch", "gpuffn_thor_dec_mbart_large", save_key_fn=lambda x: "gftmb25"),
            HyperParam("--num-experts", N_GPUS),
            HyperParam("--inference-level", 1),
            HyperParam("--expert-increment", 3),
        ]

    assert os.path.exists(pret_model), f"{pret_model} not exists."
    # print(f"NO CACHED MODEL=========================")

    return model_params + [
        HyperParam("--encoder-normalize-before", True, binary_flag=True),
        HyperParam("--decoder-normalize-before", True, binary_flag=True),
        HyperParam("--layernorm-embedding", True, binary_flag=True),
        HyperParam("--finetune-from-model", pret_model, save_key_fn=lambda x: PRET_MODEL_NAME),
    ] + cached_params


def get_task_params(bt_data_name=None, bt_bin_dir=None):
    bt_data_name = bt_data_name or "cc25_c100m_mono_en_indic"
    bt_bin_dir = bt_bin_dir or "bin_filu"

    lang_dict = os.path.join(ROOT_DIR, "pret_models/criss/criss_checkpoints/lang_dict.txt")
    no_main_data = True
    lambda_main = "0"
    lambda_bt = "1"
    if STAGE == 0:
        lang_pairs = ",".join([f"en_XX-{x},{x}-en_XX" for x in ["ne_NP", "si_LK"]])
    else:
        if BT_DIRECTION_NAME == "en_indi_toindi":
            lang_pairs = ",".join([f"en_XX-{x}" for x in ["ne_NP", "si_LK"]])
        elif BT_DIRECTION_NAME == "en_indi_toen":
            lang_pairs = ",".join([f"{x}-en_XX" for x in ["ne_NP", "si_LK"]])
        else:
            if TO_ENGLISH:
                lang_pairs = f'{TARGET_LANG}-en_XX'
            else:
                lang_pairs = f'en_XX-{TARGET_LANG}'

    bt_data = os.path.join(ROOT_DIR, f"data_fairseq/{bt_data_name}/{bt_bin_dir}")
    rank_to_langs = get_rank_to_langs("enindi_4g")
    bt_direct_name = BT_DIRECTION_NAME
    bt_langs, bt_lang_pairs, bt_directions = get_bt_direction(bt_direct_name)
    print(f"{bt_direct_name=}")
    print(f"{bt_langs=}")
    print(f"{bt_lang_pairs=}")
    print(f"{bt_directions=}")
    print(f"{lang_pairs=}")


    if STAGE == 4:
        task_name = "translation_multi_umt_simple_epoch"
        task_params = [
            HyperParam("--task", task_name, lambda x: "tmuse"),
            HyperParam("--skip-invalid-size-inputs-valid-test", True, binary_flag=True),
            HyperParam("--gen-subset", "train"),
            HyperParam("--bpe", "sentencepiece"),
            HyperParam("--decoder-langtok", True, binary_flag=True),
            HyperParam("--lang-pairs", lang_pairs),
            HyperParam("--lang-dict", lang_dict),
            HyperParam("--lang-tok-style", "mbart"),
            HyperParam("--lambda-main", lambda_main, save_key_fn=lambda x: f"M{lambda_to_name(x)}"),
            HyperParam("--lambda-bt", lambda_bt, save_key_fn=lambda x: f"B{lambda_to_name(x)}"),
            HyperParam("--bt-directions", bt_directions, save_key_fn=lambda x: bt_direct_name),
        ]
    else:
        task_name = "gpusep_translation_multi_umt_simple_epoch"
        task_params = [
            # HyperParam("--task", task_name, lambda x: "".join([z[0] for z in x.split("_")])),
            HyperParam("--task", task_name, lambda x: "gtmuse"),
            HyperParam("--skip-invalid-size-inputs-valid-test", True, binary_flag=True),
            HyperParam("--gen-subset", "train"),
            HyperParam("--bpe", "sentencepiece"),
            HyperParam("--decoder-langtok", True, binary_flag=True),
            HyperParam("--lang-pairs", lang_pairs),
            HyperParam("--lang-dict", lang_dict),
            HyperParam("--lang-tok-style", "mbart"),
            HyperParam("--lambda-main", lambda_main),
            HyperParam("--lambda-bt", lambda_bt),
            HyperParam("--bt-directions", bt_directions, save_key_fn=lambda x: bt_direct_name),
            HyperParam("--rank-to-langs", rank_to_langs),
        ]

    return task_params + [
        HyperParam("--bt-data", bt_data),
        HyperParam("--bt-lang-pairs", bt_lang_pairs),
        HyperParam("--show-interval", 5000),
        HyperParam("--sampling-method", "uniform"),
        HyperParam("--sampling-temperature", 1.0),
        HyperParam("--no-main-data", no_main_data, binary_flag=True),

    ]


def get_training_params():
    return [
        HyperParam("--criterion", "label_smoothed_cross_entropy"),
        HyperParam("--label-smoothing", 0.2),
        HyperParam("--optimizer", "adam"),
        HyperParam("--adam-eps", 1e-06),
        HyperParam("--adam-betas", '(0.9, 0.98)'),
        HyperParam("--lr-scheduler", 'inverse_sqrt'),
        HyperParam("--lr", 3e-05),
        # HyperParam("--lr", 3e-05, save_key_fn=lambda x: f'lr3e5'),
        HyperParam("--warmup-updates", 2500),
        HyperParam("--dropout", 0.3),
        HyperParam("--attention-dropout", 0.1),
        HyperParam("--weight-decay", 0.0),

        HyperParam("--max-update", 25000),
        HyperParam("--max-tokens", 1280, save_key_fn=lambda x: f"mtok{x // 100}h"), # mtok12h
        HyperParam("--update-freq", 2, save_key_fn=lambda x: f"ufre{x}"),

        HyperParam("--save-interval", 1),
        HyperParam("--save-interval-updates", 5000),
        HyperParam("--keep-interval-updates", 10),

        # ---
        HyperParam("--no-epoch-checkpoints", True, binary_flag=True),
        HyperParam("--virtual-epoch-size", 10000000),
        # HyperParam("--seed", hash_path_to_int(f'{BT_DIRECTION_NAME}{STAGE}'), save_key_fn=lambda x: f's{str(x)[-2:]}'),
        HyperParam("--seed", hash_path_to_int(f'{BT_DIRECTION_NAME}{STAGE}')),
        HyperParam("--log-format", "simple"),
        HyperParam("--log-interval", 100),
        HyperParam("--fp16", True, binary_flag=True),

        # use_sharded_state
        HyperParam("--save-sharded-state", True, binary_flag=True),
    ] + ([HyperParam("--load-sharded-state-id", SHARD_ID)] if STAGE == 4 else [])


def get_grid(args):
    # training ndata
    bt_data_name = "cc25_c100m_mono_en_indic"
    bt_data_name_short = "c25c100indic"
    bt_bin_dir = "bin_filu"

    args.data = os.path.join(ROOT_DIR, f"data_fairseq/{bt_data_name}/{bt_bin_dir}")
    args.save_prefix = os.path.join(ROOT_DIR, f"train_fairseq/{bt_data_name_short}-{bt_bin_dir[-1]}")

    get_stage_direction(args)

    grid = []
    grid += get_mbart_params()
    grid += get_task_params(bt_data_name=bt_data_name, bt_bin_dir=bt_bin_dir)
    grid += get_training_params()

    return grid



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)



