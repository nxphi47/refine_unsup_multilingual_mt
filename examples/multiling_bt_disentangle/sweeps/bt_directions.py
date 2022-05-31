import os
import argparse
import json
import itertools

# def bt_direction_pack(bt_direct_name="euroindia"):

RANK_TO_LANGS = {}
BT_DIRECTIONS = {}

def register_directions(fn):
    # BT_DIRECTIONS[fn.__name__]
    assert fn.__name__ not in BT_DIRECTIONS, f'{fn.__name__=} exists'
    BT_DIRECTIONS[fn.__name__] = fn
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def register_rank_to_langs(fn):
    RANK_TO_LANGS[fn.__name__] = fn
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


@register_directions
def euroindia():
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})

    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enin():
    eu_langs = "en_XX"
    in_langs = "hi_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})

    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enhi():
    return enin()


@register_directions
def euroindia_all():
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: all
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enindia_all():
    eu_langs = "en_XX"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: all
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enindia_all_test():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: all
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enndia_all_tri():
    """
    en_XX -> [hi,ne,si,gu]
    ne_NP -> [en_XX] and [hi,si,gu]
    """
    eu_langs = "en_XX"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: all
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def euroindi_toindi():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # all_langs = eu_langs.split(",") + in_langs.split(",")
    # bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_indi_toindi():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_ne_tone():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "ne_NP"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_si_tosi():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "si_LK"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_gu_togu():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "gu_IN"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_hi_tohi():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions

# @register_directions
# def en_ne_toen():
#     # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
#     #   Onlt india monolingual data is required
#     eu_langs = "en_XX"
#     in_langs = "ne_NP"
#     bt_langs = in_langs
#     bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
#     bt_directions = ",".join(list(itertools.chain.from_iterable([
#         [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
#     )
#     return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_si_toen():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "si_LK"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_gu_toen():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "gu_IN"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_hi_toen():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_in_toen():
    # TODO: toeuro means BT train on indi->eu only, direction must be: eu-indi
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_in_toin():
    # TODO: toindi means BT train on en->indi only, direction must be: indi-eu 
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN"
    bt_langs = in_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in in_langs.split(",") for y in eu_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_ne_toen():
    # TODO: toeuro means BT train on indi->eu only, direction must be: eu-indi
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "ne_NP"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def euroindi_toeuro():
    # TODO: toeuro means BT train on indi->eu only, direction must be: eu-indi
    #   Onlt india monolingual data is required
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_indi_toen():
    # TODO: toeuro means BT train on indi->eu only, direction must be: eu-indi
    #   Onlt india monolingual data is required
    eu_langs = "en_XX"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in eu_langs.split(",") for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def eu_xin():
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: eu->in, and in->in
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    bt_directions = ",".join(
        list(itertools.chain.from_iterable([
            [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])) + 
        [f"{x}-{y}" for x in in_langs.split(",") for y in in_langs.split(",") if x != y]
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def xeu_in():
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: eu->in, and in->in
    all_langs = eu_langs.split(",") + in_langs.split(",")
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    bt_directions = ",".join(
        list(itertools.chain.from_iterable([
            [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])) + 
        [f"{x}-{y}" for x in eu_langs.split(",") for y in eu_langs.split(",") if x != y]
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_centric():
    eu_langs = "en_XX"
    in_langs = "hi_IN,ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})

    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"en_XX-{y}", f"{y}-en_XX"] for y in in_langs.split(",")]))
    )
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enne():
    eu_langs = "en_XX"
    in_langs = "ne_NP"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ensi():
    eu_langs = "en_XX"
    in_langs = "si_LK"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennehi():
    eu_langs = "en_XX"
    in_langs = "ne_NP,hi_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennesi():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennesi_all():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    all_langs = eu_langs.split(",") + in_langs.split(",")
    # bt_directions = ",".join(list(itertools.chain.from_iterable([
    #     [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennesigu_all():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    all_langs = eu_langs.split(",") + in_langs.split(",")
    # bt_directions = ",".join(list(itertools.chain.from_iterable([
    #     [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    bt_directions = ",".join([f"{x}-{y}" for x in all_langs for y in all_langs if x != y])
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennesihi():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK,hi_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennehigu():
    eu_langs = "en_XX"
    in_langs = "ne_NP,hi_IN,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def ennehisigu():
    eu_langs = "en_XX"
    in_langs = "ne_NP,si_LK,hi_IN,gu_IN"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enfresdesi():
    eu_langs = "en_XX,fr_XX,es_XX,de_DE"
    # eu_langs = "en_XX"
    in_langs = "si_LK"
    bt_langs = ",".join([eu_langs, in_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in eu_langs.split(",") for y in in_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


# ====== European pairs ================================
@register_directions
def enfrde():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fr_XX,de_DE"
    bt_langs = ",".join([en_langs, eu_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


# ====== European pairs ================================
@register_directions
def enfrdero():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fr_XX,de_DE,ro_RO"
    bt_langs = ",".join([en_langs, eu_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


# ====== Uralic European pairs ================================
@register_directions
def enfilvet():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fi_FI,lv_LV,et_EE"
    bt_langs = ",".join([en_langs, eu_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enfilvet_touralic():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fi_FI,lv_LV,et_EE"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enfilvet_toen():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fi_FI,lv_LV,et_EE"
    bt_langs = en_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_lv_tolv():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "lv_LV"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_et_toet():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "et_EE"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_fi_tofi():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = "fi_FI"
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions



# ====== ALl low-resource pairs ================================
@register_directions
def en_nehisigufiesettrkk():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = ",".join(["ne_NP,si_LK,hi_IN,gu_IN", "fi_FI,lv_LV,et_EE", "tr_TR,kk_KZ"])
    bt_langs = ",".join([en_langs, eu_langs])
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}", f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_nehisigufiesettrkk_tolowres():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = ",".join(["ne_NP,si_LK,gu_IN", "lv_LV,et_EE", "kk_KZ"])
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def en_nehisigufiesettrkk_toen():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = ",".join(["ne_NP,si_LK,gu_IN", "lv_LV,et_EE", "kk_KZ"])
    bt_langs = en_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enkk_tokk():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = ",".join(["kk_KZ"])
    bt_langs = eu_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{y}-{x}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


@register_directions
def enkk_toen():
    en_langs = "en_XX"
    # in_langs = "fr_XX,es_XX,de_DE"
    eu_langs = ",".join(["kk_KZ"])
    bt_langs = en_langs
    bt_lang_pairs = json.dumps({"bt": ",".join([f"{x}-{x}" for x in bt_langs.split(",")])})
    # todo: euro -> indi
    bt_directions = ",".join(list(itertools.chain.from_iterable([
        [f"{x}-{y}"] for x in en_langs.split(",") for y in eu_langs.split(",")])))
    return bt_langs, bt_lang_pairs, bt_directions


# ================================================================================
N_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


@register_rank_to_langs
def enindi_hi_4g():
    assert N_GPUS == 4, f'{N_GPUS=} not 4'
    rank_to_langs = json.dumps({
        0: "en_XX-en_XX,ne_NP-ne_NP,hi_IN-hi_IN", # BT:en-(nesihi) => en-ne due to filter, BT:ne->en
        1: "en_XX-en_XX,si_LK-si_LK,hi_IN-hi_IN",
        2: "en_XX-en_XX,hi_IN-hi_IN",
        3: "en_XX-en_XX,gu_IN-gu_IN,hi_IN-hi_IN",
    })
    return rank_to_langs


@register_rank_to_langs
def enindi_4g():
    assert N_GPUS == 4, f'{N_GPUS=} not 4'
    rank_to_langs = json.dumps({
        0: "en_XX-en_XX,ne_NP-ne_NP", # BT:en-(nesihi) => en-ne due to filter, BT:ne->en
        1: "en_XX-en_XX,si_LK-si_LK",
        2: "en_XX-en_XX,hi_IN-hi_IN",
        3: "en_XX-en_XX,gu_IN-gu_IN",
    })
    return rank_to_langs


@register_rank_to_langs
def enuralic_3g():
    assert N_GPUS == 3, f'{N_GPUS=} not 3'
    rank_to_langs = json.dumps({
        0: "en_XX-en_XX,fi_FI-fi_FI", # BT:en-(nesihi) => en-ne due to filter, BT:ne->en
        1: "en_XX-en_XX,lv_LV-lv_LV",
        2: "en_XX-en_XX,et_EE-et_EE",
    })
    return rank_to_langs



def get_bt_direction(name):
    return BT_DIRECTIONS[name]()


def get_rank_to_langs(name):
    return RANK_TO_LANGS[name]()


if __name__ == "__main__":
    print(get_bt_direction("euroindi_toindi"))
