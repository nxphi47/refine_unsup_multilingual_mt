import os
import argparse
import json
import itertools

# def bt_direction_pack(bt_direct_name="euroindia"):


def lambda_to_name(lambda_str):
    packs = [x.split(":") for x in lambda_str.split(",")]
    if len(packs) == 1:
        assert len(packs[0]) == 1, f'{lambda_str} invalid'
        return packs[0][0]
    packs = [(int(x[0]), float(x[1])) for x in packs]
    names = [
        f"{x[0] // 1000}p{int(x[1] * 100)}"
        for x in packs if x[0] > 0
    ]
    lname = "t".join(names)
    return lname





if __name__ == "__main__":
    lambda_ = "0:1,5000:0.1,10000:0.01"
    # lambda_ = "0"
    print(lambda_to_name(lambda_))
