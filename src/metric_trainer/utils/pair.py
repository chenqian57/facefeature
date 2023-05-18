import glob
import os.path as osp
import random
import numpy as np
from tqdm import tqdm


def gen_pair(
    pic_root, save_path="pair.txt", sim_ratio=0.5, total_num=10000, interval=500
):
    """Generate pairs for test.

    Args:
        pic_root (str): The root of cropped test images,
            pic_root/category_id/{category_id}_{img_id}_{id}_{suffix}.jpg .
    """
    cat_dirs = glob.glob(osp.join(pic_root, "*"))
    simIndex = random.choices(range(len(cat_dirs)), k=int(total_num * sim_ratio))
    # index2 = itertools.combinations(range(len(cat_dirs)), 2)
    count = 0
    with open(save_path, "w") as f:
        # for i in simIndex:
        while count < int(sim_ratio * total_num):
            simIndex = random.randint(0, len(cat_dirs) - 1)
            img_names = glob.glob(osp.join(cat_dirs[simIndex], "*"))
            if len(img_names) < 2:
                continue
            pairs = sim_imgs(img_names)
            flag = 1
            f.write(f"{pairs[0]} {pairs[1]} {flag} {count // interval}" + "\n")
            count += 1
            print(count)
        while count < total_num:
            diffIndex1 = random.randint(0, len(cat_dirs) - 1)
            diffIndex2 = random.randint(0, len(cat_dirs) - 1)
            if diffIndex1 == diffIndex2:
                continue
            img_names1 = glob.glob(osp.join(cat_dirs[diffIndex1], "*"))
            img_names2 = glob.glob(osp.join(cat_dirs[diffIndex2], "*"))
            if len(img_names1) == 0 or len(img_names2) == 0:
                continue
            pairs = diff_imgs(img_names1, img_names2)
            flag = -1
            f.write(f"{pairs[0]} {pairs[1]} {flag} {count // interval}" + "\n")
            count += 1
            print(count)


def sim_imgs(img_names):
    """
    Args:
        img_names (List): root/category_id/*
    """
    indexs = random.sample(range(len(img_names)), 2)
    return img_names[indexs[0]], img_names[indexs[1]]


def diff_imgs(img_names1, img_names2):
    """
    Args:
        img_names1 (List): root/category_id1/*
        img_names2 (List): root/category_id2/*
    """
    index1 = random.randint(0, len(img_names1) - 1)
    index2 = random.randint(0, len(img_names2) - 1)
    return img_names1[index1], img_names2[index2]

def parse_pair(pair_path):
    with open(pair_path, 'r') as f:
        pairs = [p.strip() for p in f.readlines()]

    random.shuffle(pairs)
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for p in tqdm(pairs, total=len(pairs)):
        nameL, nameR, flag, fold = p.split(' ')
        nameLs.append(nameL)
        nameRs.append(nameR)
        flags.append(int(flag))
        folds.append(int(fold))

    return [nameLs, nameRs, np.array(flags), np.array(folds)]
