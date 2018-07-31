import argparse
import pandas as pd
import numpy as np
import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Preprocess the datasets')

parser.add_argument('--dir', '-d', required=True)
parser.add_argument('--size', '-s', default=300, type=int)


def create_map_files(src, dest):
    """Extract the required columns from the json structure and stores as csv.

    Args:
        src: source annotation.json file
        dest: destination file name
    """
    data = json.load(open(src))
    imgs = pd.DataFrame(data['images'])
    if dest is not 'test':
        print('In if: {}'.format(dest))
        annotations = pd.DataFrame(data['annotations'])
        annotations = annotations.replace(
            {'category_id': {0: 'no_animal', 1: 'animal'}})
        df = pd.merge(imgs, annotations[['category_id', 'image_id']],
                      left_on='id', right_on='image_id').drop('image_id', axis=1)
        df = df[['file_name', 'location', 'category_id']]
    else:
        print('In else: {}'.format(dest))
        df = imgs[['file_name', 'id', 'location']]
    df.to_csv('data/{}.csv'.format(dest), index=False)


def transfer_data(dir, folder, df):
    """Transfer the files to their respective folder"""
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        src = dir / row.file_name
        dest = dir / folder / row.category_id / src.name
        src.rename(dest)


def read_img(name):
    """Read image from a file."""
    return Image.open(name)


def write_img(img, name):
    """Write image to a file."""
    img.save(name, format='JPEG', quality=90)


def resize_img(img, w=224, h=224):
    """Resize image to the specified w,h."""
    return img.resize((w, h), Image.ANTIALIAS)


def compute_w_h(img, sz=300):
    """Compute new width and height of image by maintaining the ratio.

    Args:
        img: PIL.Image
        sz: min size of width or height

    Returns:
        new width and height, by maintaining the ratio
    """
    w, h = img.size

    if w < h:
        _w, _h = sz, int(sz * (h / w))
    elif h < w:
        _h, _w = sz, int(sz * (w / h))
    else:
        _h, _w = h, w

    return _w, _h


def resize_and_save(name):
    """Resize and save the image by maintaining the ratio."""
    img = read_img(name)
    w, h = compute_w_h(img)
    img = resize_img(img, w, h)
    write_img(img, name)


def main(dir):
    # Create the CSV files
    map_files = [
        (dir/'train_annotations.json', 'train'),
        (dir/'val_annotations.json', 'val'),
        # (dir/'test_information.json', 'test')
    ]

    print('Creating map files')
    for src, dest in map_files:
        create_map_files(src, dest)
    print('Done!')

    # Moved the images to their respective directory
    train_animal = dir/'train'/'animal'
    train_no_animal = dir/'train'/'no_animal'
    val_animal = dir/'valid'/'animal'
    val_no_animal = dir/'valid'/'no_animal'

    train_animal.mkdir(parents=True, exist_ok=True)
    train_no_animal.mkdir(parents=True, exist_ok=True)
    val_animal.mkdir(parents=True, exist_ok=True)
    val_no_animal.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(dir/'train.csv')
    val_df = pd.read_csv(dir/'val.csv')

    print('Tranfering data to their respective folders')
    transfer_data(dir, 'train', train_df)
    transfer_data(dir, 'valid', val_df)
    print('Done!')

    # Balance the dataset
    print('Balancing the dataset')
    count = 0
    for file in val_no_animal.iterdir():
        if count < 3781:
            src = file
            dest = train_no_animal / file.name
            shutil.move(src, dest)
        else:
            break
        count += 1
    print('Done!')

    # Resize the images
    print('Resizing the images')
    with ProcessPoolExecutor() as executor:
        executor.map(resize_and_save, dir.glob('*/*/*.jpg'))
    print('Done!')


if __name__ == '__main__':
    args = parser.parse_args()
    dir, size = args.dir, args.size
    cwd = Path.cwd()
    data_dir = cwd / dir
    print(f'Parent directory for images: {data_dir}')
    main(data_dir)
    print('Done with data pre-processing!')
