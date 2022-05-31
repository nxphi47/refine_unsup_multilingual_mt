
import os
import argparse

pretrain_mbart_models = {
    "mb25": {
        "mbcc25": f'pret_models/mbart.cc25.v2/model.pt',
    },
    "criss": {
        "cri": f"pret_models/criss/criss_checkpoints/criss.3rd.pt",
        "gftlrtrkk": f"train_fairseq/c25c100lowres-u/gftmb25.cri.gtmuse.M0.B1.en_nehisigufiesettrkk.enlowres_x_6g.uniform.noMainD.wce.lr3e5.mtok12h.ufre2.vep10m.s86.6gpu/checkpoint_1_15000.sr5.pt",
        "gftlrtrkk2kk": f"train_fairseq/c25c100trkk-u/mb25.gftlrtrkk.cac.gftlrtrkk.sh.tmuse.M0.B1.enkk_tokk.uniform.noMainD.wce.lr3e5.mtok12h.ufre2.vep10m.s85.2gpu/checkpoint_1_20000.pt",
        "gftlrtrkk2en": f"train_fairseq/c25c100trkk-u/mb25.gftlrtrkk.cac.gftlrtrkk.sh.tmuse.M0.B1.enkk_toen.uniform.noMainD.wce.lr3e5.mtok12h.ufre2.vep10m.s17.2gpu/checkpoint_1_35000.pt",

    }
}


def get_pretrained_mbart_models(root, name, category='criss'):
    path = pretrain_mbart_models[category][name]
    if os.path.exists(os.path.join(root, path)):
        return os.path.join(root, path)
    else:
        # try to add donotdelete prefix
        parts = path.split("/")
        dnd_path = os.path.join(
            root,
            "/".join(parts[:-2] + ["donotdelete." + parts[-2]] + parts[-1:])
        )
        print(f'Try getting alternative path: {dnd_path}')
        assert os.path.exists(dnd_path), f'not found: {dnd_path=}, maybe root false'
        return dnd_path


aug_datasets = {
    "euin2dpl": f"data_fairseq/cc25_c100m_mono_enfresde_hinesigu/filu/euroindia_all.index.com.indic-en_XX.0.2.True.uplTrue",
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', default="mbart", type=str)
    parser.add_argument('--subpack', default="criss", type=str)
    parser.add_argument('--name', default=None, type=str, required=True)
    args = parser.parse_args()

    if args.pack == "mb25":
        pack = pretrain_mbart_models
    else:
        raise ValueError(f"{args.pack=} not found")
    
    print(f"{args.subpack=}, {args.name=}")
    subpack = pack[args.subpack]
    print(f"{subpack[args.name]}")
    
