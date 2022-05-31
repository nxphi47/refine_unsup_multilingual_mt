from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch
from fairseq.data import Dictionary


class EncoderLangtok(Enum):
    """
    Prepend to the beginning of source sentence either the
    source or target language token. (src/tgt).
    """

    src = "src"
    tgt = "tgt"


class LangTokSpec(Enum):
    main = "main"
    mono_dae = "mono_dae"


class LangTokStyle(Enum):
    multilingual = "multilingual"
    mbart = "mbart"


@torch.jit.export
def get_lang_tok(
    lang: str, lang_tok_style: str, spec: str = LangTokSpec.main.value
) -> str:
    # TOKEN_STYLES can't be defined outside this fn since it needs to be
    # TorchScriptable.
    TOKEN_STYLES: Dict[str, str] = {
        LangTokStyle.mbart.value: "[{}]",
        LangTokStyle.multilingual.value: "__{}__",
    }

    if spec.endswith("dae"):
        lang = f"{lang}_dae"
    elif spec.endswith("mined"):
        lang = f"{lang}_mined"
    style = TOKEN_STYLES[lang_tok_style]
    return style.format(lang)


@torch.jit.export
def get_lang_from_lang_tok(
    lang_tok: str, lang_tok_style: str, spec: str = LangTokSpec.main.value
) -> str:
    if lang_tok_style == LangTokStyle.mbart.value:
        assert len(lang_tok) > 2
        return lang_tok[1:-1]
    elif lang_tok_style == LangTokStyle.multilingual.value:
        assert len(lang_tok) > 4
        return lang_tok[2:-2]
    else:
        raise ValueError(f'{lang_tok_style} wrong, with {lang_tok=}')



def augment_dictionary(
    dictionary: Dictionary,
    language_list: List[str],
    lang_tok_style: str,
    langtoks_specs: Sequence[str] = (LangTokSpec.main.value,),
    extra_data: Optional[Dict[str, str]] = None,
) -> None:
    for spec in langtoks_specs:
        for language in language_list:
            dictionary.add_symbol(
                get_lang_tok(lang=language, lang_tok_style=lang_tok_style, spec=spec)
            )

    if lang_tok_style == LangTokStyle.mbart.value or (
        extra_data is not None and LangTokSpec.mono_dae.value in extra_data
    ):
        dictionary.add_symbol("<mask>")
