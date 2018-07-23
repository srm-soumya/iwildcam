import argparse
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Resize images by maintaining the ratio to the smaller size specified')

parser.add_argument('--dir', '-d', required=True)
parser.add_argument('--size', '-s', default=300, type=int)


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
    """Run the image resize process parallely."""
    with ProcessPoolExecutor() as executor:
        executor.map(resize_and_save, dir.glob('*/*/*.jpg'))


if __name__ == '__main__':
    args = parser.parse_args()
    dir, size = args.dir, args.size
    cwd = Path.cwd()
    data_dir = cwd / dir
    print(f'Parent directory for images: {data_dir}')
    main(data_dir)
    print('Done resizing images!')
