from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs=1, help='root dir', help='Dataset root directory. Should contain train, eval, and test subdirectories.')
    args = parser.parse_args()

    root = Path(args.root[0])
    folder = Path('instereo2k')
    folder.mkdir(exist_ok=True, parents=False)

    for t in ['train', 'test']:
        path = root / t

        names = [f for f in path.iterdir() if f.is_dir()]

        files = []
        for name in names:
            left = Path(name) / 'left.png'
            right = Path(name) / 'right.png'
            files.append(f'{left}, {right}')

        print(f'writing {t} files to {path}')
        with open(folder / f'{t}.txt', 'w') as text_file:
            text_file.write('\n'.join(files))

