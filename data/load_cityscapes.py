from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs=1, help='root dir', help='Dataset root directory. Should contain train, eval, and test subdirectories.')
    args = parser.parse_args()

    root = Path(args.root[0])
    folder = Path('cityscapes')
    folder.mkdir(exist_ok=True, parents=False)

    for t in ['train', 'eval', 'test']:
        path = root / t

        image_list = [file for file in path.glob('**/*') if file.is_file() and file.suffix.lower() == '.png']

        # set removes duplicates due to *_disparity.png, *_rightImg8bit.png, *_leftImg8bit.png
        names = list({'_'.join(str(f).split('_')[:-1]) for f in image_list})
        names.sort()

        files = []
        for name in names:
            left = name + '_leftImg8bit.png'
            right = name + '_rightImg8bit.png'
            files.append(f'{left}, {right}')

        print(f'writing {t} files to {path}')
        with open(folder / f'{t}.txt', 'w') as text_file:
            text_file.write('\n'.join(files))