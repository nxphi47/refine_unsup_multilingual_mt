
# from indicnlp import normalize
# from indicnlp.normalize import indic_normalize
# from indicnlp.tokenize import indic_tokenize

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import argparse
import fileinput
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--indic-nlp-path", required=True,
    #                     help="path to Indic NLP Library root")
    parser.add_argument("--language", required=True)
    parser.add_argument("--remove-nuktas", default=False, action="store_true")
    parser.add_argument("input", help="input file; use - for stdin")
    args = parser.parse_args()

    # try:
    #     sys.path.extend([
    #         args.indic_nlp_path,
    #         os.path.join(args.indic_nlp_path, "src"),
    #     ])
    #     from indicnlp.tokenize import indic_tokenize
    #     from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    # except:
    #     raise Exception(
    #         "Cannot load Indic NLP Library, make sure --indic-nlp-path is correct"
    #     )

    # create normalizer
    # factory = IndicNormalizerFactory()
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer(
        args.language, remove_nuktas=args.remove_nuktas,
    )

    # normalize and tokenize
    for line in fileinput.input([args.input], openhook=fileinput.hook_compressed):
        # line = normalizer.normalize(line.decode("utf-8"))
        # line = " ".join(indic_tokenize.trivial_tokenize(line, args.language))
        # sys.stdout.write(line.encode("utf-8"))
        line = normalizer.normalize(line)
        line = " ".join(indic_tokenize.trivial_tokenize(line, args.language))
        sys.stdout.write(line)
        

if __name__ == '__main__':
    main()

