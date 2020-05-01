from pathlib import Path
import pandas as pd

def main():
    data_dir = Path('/home/vladbakhteev/data/aorta')
    img_paths = sorted([p.stem + '.nii' for p in data_dir.glob('imageCT*.nii')])
    mask_paths = sorted([p.stem + '.nii' for p in data_dir.glob('result*.nii')])

    df = pd.DataFrame({'img_path': img_paths, 'mask_path': mask_paths})
    df['train'] = True
    df.loc[len(df)-2:, 'train'] = False    # 2 samples for validation

    df.to_csv(data_dir / 'train.csv', index=False)


if __name__=='__main__':
    main()