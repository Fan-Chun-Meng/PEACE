import argparse

import torch

from lib.new_dataLoader import ParseData
from lib.picture import draw_picture
from lib.evaluate_spring_trajectories import fake_evaluate_spring_trajectories

def update_dataLoader(data_loaders,device,original_data_loader,train_losses,args):
    hist = []
    tar = []
    sys_para = []
    for i in range(data_loaders['train']['num_batch']):
        # Get a batch of data
        encoder_batch = next(data_loaders['train']['encoder'])
        decoder_batch = next(data_loaders['train']['decoder'])
        graph_batch = next(data_loaders['train']['graph'])
        sys_para_batch = next(data_loaders['train']['sys_para'])

        # Move data to device
        encoder_batch = encoder_batch.to(device)
        decoder_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in decoder_batch.items()}
        sys_para_batch = sys_para_batch.to(device)
        tar.append(decoder_batch['data'].view(encoder_batch.batch_size, 10, -1, 4))
        hist.append(encoder_batch.x.view(encoder_batch.batch_size, 10, -1, 4))
        sys_para.append(sys_para_batch)

    max_val = original_data_loader.max_loc
    min_val = original_data_loader.min_loc

    predictions = torch.cat(train_losses['predictions'], dim=0)
    hist = torch.cat(hist, dim=0)
    hist_ori = hist
    tar = torch.cat(tar, dim=0)
    tar_ori = tar
    sys_para = torch.cat(sys_para, dim=0)
    mean = tar - predictions
    tar_ori[..., 0:2] = tar[..., 0:2] * (max_val - min_val) + min_val
    # tar[..., 0:2] = (tar[..., 0:2] - min_val)*2/(max_val - min_val)-1
    predictions = tar_ori - mean
    hist_ori[..., 0:2] = hist[..., 0:2] * (max_val - min_val) + min_val
    print(f'开始生成图像')
    draw_picture(hist_ori, predictions, 0, 0, 0, 'plot')
    sys_dict = {}
    for i in range(0,sys_para.shape[0]):
        sys_dict[str(i)] = sys_para[i]
    df, high_conf_samples = fake_evaluate_spring_trajectories('plot',sys_dict)
    high_conf_idx = list(df[df["confidence"] >= 90].index)
    print(f'开始更新数据集，共补充{str(len(high_conf_idx))}条数据')
    high_conf_idx = torch.tensor(high_conf_idx, dtype=torch.long)
    hist = hist[high_conf_idx]
    tar = tar[high_conf_idx]
    data_args = argparse.Namespace(
        random_seed=42,
        total_ode_step=24,
        total_ode_step_train=24,
        cutting_edge=True,
        extrap_num=args.extrap_num,
        condition_num=12,
        mode=args.mode,
        suffix=args.suffix,
        num_domains=args.num_domains
    )
    data_loader_new = ParseData(
        dataset_path=args.id_data_path,  # Use ID data path as main data path
        args=data_args,
        suffix=args.suffix,
        mode="extrap",
        enable_normalization=not args.disable_norm
    )
    train_loader = data_loader_new.load_data(
        sample_percent=0.7,
        batch_size=args.batch_size,
        data_type="train",
        domain_id=0,
        new_data=[hist,tar]
    )
    data_loaders = {
        'train': {
            'encoder': train_loader[0],
            'decoder': train_loader[1],
            'graph': train_loader[2],
            'sys_para': train_loader[3],
            'num_batch': train_loader[4]
        },
        'val':data_loaders['val'],
        'id_test':data_loaders['id_test'],
        'ood_test':data_loaders['ood_test']
    }
    print(f'更新完成')
    return data_loaders