import torch

import numpy as np
import torch.optim as optim
from torch.cuda import amp

from tqdm import tqdm
import time

from dataset import dataloader_semseg as dataloader
from tools import dir_utils
from configs.load_yaml import load_yaml
from tools.lr_warmup_scheduler import GradualWarmupScheduler

def main(yaml_file,test_mode=False):
    ######### prepare environment ###########
    # if torch.cuda.is_available():
    device = torch.device('cuda') 
    #     device_ids = [i for i in range(torch.cuda.device_count())]
    #     print('===> using GPU {} '.format(device_ids))
    # else:
    #     device = torch.device('cpu')
    #     print('===> using CPU !!!!!')

    opt = load_yaml(yaml_file,saveYaml2output=True)
    epoch = opt.OPTIM.NUM_EPOCHS

    model_dir  = opt.SAVE_DIR+'models/'
    dir_utils.mkdir_with_del(model_dir)

    ######### dataset ###########
    train_dataset = dataloader.Noise_Dataset(opt.DATASET.TRAIN_CSV, leads=opt.DATASET_CUSTOME.LEADS, 
                                                date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
                                                n_max_cls=opt.DATASET_CUSTOME.OUT_C,
                                                random_crop=True,
                                                transform = dataloader.get_transform(train=True)
                                                )

    val_dataset = dataloader.Noise_Dataset(opt.DATASET.VAL_CSV, leads=opt.DATASET_CUSTOME.LEADS, 
                                                date_len=opt.DATASET_CUSTOME.INPUT_LENGTH, 
                                                n_max_cls=opt.DATASET_CUSTOME.OUT_C,
                                                random_crop=False,
                                                transform = dataloader.get_transform(train=False)
                                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, 
                                                    shuffle=True, num_workers=4,
                                                    prefetch_factor=3,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, 
                                                    shuffle=False, num_workers=4,
                                                    prefetch_factor=3,
                                                    persistent_workers=False, #maintain woker alive even consumed
                                                    # drop_last=True,
                                                    )

    dataset_sizes = {'train':len(train_dataset),
                     'val':len(val_dataset)}

    print('===> Loading datasets done')

    ######### model ###########

    if opt.MODEL.MODE == 'unet_transformer_upsample':
        from models.unet_swin_transformer_1d_upsample import Model
        model = Model(in_c=1,
                    out_c=opt.DATASET_CUSTOME.OUT_C, \
                    img_size=opt.DATASET_CUSTOME.INPUT_LENGTH, \
                    embed_dim=opt.MODEL.EMBED_DIM, \
                    patch_size=opt.MODEL.PATCH_SIZE, \
                    window_size=opt.MODEL.WINDOW_SIZE, \
                    depths=opt.MODEL.DEPTHS, \
                    num_heads=opt.MODEL.N_HEADS, \
                    # denoise_mode=opt.MODEL.Denoise_Mode, \
                    ).to(device)
    else:
        print('{} unrecoginze model'.format(opt.MODEL.MODE))
        assert 1>2

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids = device_ids)

    ######### optim ########### d
    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)
    
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1]).to(device))
    grad_scaler = amp.GradScaler()
    start_epoch = 0
    
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    # for epoch in range(start_epoch, 2 + 1):
        epoch_start_time = time.time()
        epoch_train_loss = 0 

        #### train ####
        model.train()
        for i, data in enumerate(tqdm(train_dataloader), 0):
        # for i, data in enumerate(train_dataloader):
            inputs = data['input'].to(device)
            labels = data['label'].to(device) 

            optimizer.zero_grad()
            # with torch.set_grad_enabled(True):
            torch.set_grad_enabled(True)
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_train_loss += loss.item() * inputs.size(0)
        train_loss_mean = epoch_train_loss / dataset_sizes['train']

        #### Evaluation ####
        model.eval()
        epoch_val_loss = 0
        for data in val_dataloader:
            inputs = data['input'].to(device)
            labels = data['label'].to(device)
            
            torch.set_grad_enabled(False)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            epoch_val_loss += loss.item()* inputs.size(0)

        val_loss_mean = epoch_val_loss / dataset_sizes['val']
        scheduler.step()

        save_path = model_dir+'model_epoch_{}_val_{:.6f}.pth'.format(epoch,val_loss_mean)
        torch.save(model, save_path)
        print(save_path)

        # assert 1>2
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}s \t train Loss: {:.6f} val loss: {:.6f}".format(
                epoch, time.time()-epoch_start_time, train_loss_mean, val_loss_mean))
        print("------------------------------------------------------------------")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="train")
    parser.add_argument("-c", "--config", type=str, 
                        default=None,
                        help="path to yaml file")
    args = parser.parse_args()

    main(args.config,test_mode=False)

