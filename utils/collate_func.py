import torch

def collate_func(batch, min_crop=32, max_crop=128):
    crop_sizes = [2 ** i for i in range(int(torch.log2(min_crop)), int(torch.log2(max_crop)) + 1)]
    crop_idx = torch.randint(len(crop_sizes))
    crop_size = crop_sizes[crop_idx]

    x_start = np.random.randint(0, h - crop_size)
    y_start = np.random.randint(0, w - crop_size)

    lr = torch.FloatTensor([item[0] for item in batch])
    hr = torch.FloatTensor([item[1] for item in batch])

    lr = lr[...,x_start:x_start+crop_size, y_start:y_start+crop_size]
    hr = hr[...,x_start:x_start+crop_size, y_start:y_start+crop_size]

    return [lr, hr]