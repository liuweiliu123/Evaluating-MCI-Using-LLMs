import json
import random


def five_fold_split(json_file):

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.shuffle(data)

    fold_size = len(data) // 5

    folds = []
    for i in range(5):

        test_data = data[i * fold_size: (i + 1) * fold_size] if i < 4 else data[i * fold_size:]

        train_data = data[: i * fold_size] + data[(i + 1) * fold_size:] if i < 4 else data[: i * fold_size]

        folds.append({
            'fold': i + 1,
            'train_data': train_data,
            'test_data': test_data
        })

    return folds


if __name__ == '__main__':

    json_file_path = '/Users/apple/PycharmProjects/pythonProject36/Chinese_audioTotext_all.json'
    folds = five_fold_split(json_file_path)

    for fold_info in folds:
        fold_idx = fold_info['fold']
        train_data = fold_info['train_data']
        test_data = fold_info['test_data']


        with open(f'train_fold_{fold_idx}.json', 'w', encoding='utf-8') as ft:
            json.dump(train_data, ft, ensure_ascii=False, indent=2)

        with open(f'test_fold_{fold_idx}.json', 'w', encoding='utf-8') as fs:
            json.dump(test_data, fs, ensure_ascii=False, indent=2)

        print(f'Fold {fold_idx}: Train size = {len(train_data)}, Test size = {len(test_data)}')
