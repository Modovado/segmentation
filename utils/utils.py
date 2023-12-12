from __future__ import annotations
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

DEFAULT_SEED: int = 42


# palette for potsdam and vaihingen
# https://www.kaggle.com/code/damminhtien/fcn8s-vgg16-bn/notebook
palette_potsdam = \
    {0: (255, 255, 255),  # Impervious surfaces (white)
     1: (0, 0, 255),  # Buildings (blue)
     2: (0, 255, 255),  # Low vegetation (cyan)
     3: (0, 255, 0),  # Trees (green)
     4: (255, 255, 0),  # Cars (yellow)
     5: (255, 0, 0),  # Clutter (red)
     6: (0, 0, 0)}  # Undefined (black)

# palette_loveda = \
#     {0: (255, 255, 255),  # Background (white) + DontCare
#      1: (255, 0, 0),  # Building (red)
#      2: (255, 255, 0),  # Road (yellow)
#      3: (0, 0, 255),  # Water (blue)
#      4: (159, 129, 183),   # Barren (purple)
#      5: (0, 255, 0),  # Forest (green)
#      6: (255, 195, 128)}  # Agricultural (orange)

palette_loveda = \
    {255: (0, 0, 0),  # Ignore (black)
     0: (255, 255, 255),  # Background (white)
     1: (255, 0, 0),  # Building (red)
     2: (255, 255, 0),  # Road (yellow)
     3: (0, 0, 255),  # Water (blue)
     4: (159, 129, 183),   # Barren (purple)
     5: (0, 255, 0),  # Forest (green)
     6: (255, 195, 128)}  # Agricultural (orange)

# palette for satellite
palette_satellite = \
    {0: (0, 0, 0),  # Undefined/Background (black)
     1: (255, 255, 255)}  # Target (white)


def seed_everything(seed=DEFAULT_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# now monitor can handle both loss and metric
def train(device, epoch, model, train_loader, optimizer, scheduler, loss_fns, metric_fns, monitor, writer, *args,
          loss_weights=None):
    # scheduler
    if scheduler is not None:
        scheduler.step(epoch)

    # [loss_weight]
    if loss_weights is None:
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader, total=len(train_loader))):

        num_batches_per_epoch = len(train_loader)
        num_updates = epoch * num_batches_per_epoch

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = train_data['image'].float().to(device), train_data['label'].float().to(device)
        outputs, aux_loss = model(inputs)
        outputs = model(inputs)
        # loss functions
        for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
            # weight(lambda) * loss
            loss = loss_weight * loss_fn(outputs, labels.long())
            name = loss_fn.__class__.__name__
            loss_dict[name].update(loss.item())

            # total loss
            total_loss += loss

        # aux_loss
        total_loss += aux_loss
        loss_dict['AuxLoss'].update(aux_loss.item())  # ONLY for the model has additional intermediate loss
        loss_dict['TotalLoss'].update(total_loss.item())

        # metric
        for i, metric_fn in enumerate(metric_fns):
            metric = metric_fn(outputs, labels.long())
            name = metric_fn.__class__.__name__
            metric_dict[name].update(metric)

        # metric functions --> MetricMeter
        metricmeter.update(outputs, labels, from_logits=True)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # scheduler
        if scheduler is not None:
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'train/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value.avg, epoch + 1)

    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value, epoch + 1)

    # print(monitor)


# ONLY for the model has additional intermediate loss
# now monitor can handle both loss and metric
def train_no_aux_loss(device, epoch, model, train_loader, optimizer, scheduler, loss_fns, metric_fns, monitor, writer,
                      *args, loss_weights=None):
    # [TIMM_SCHED]
    # scheduler
    # if scheduler is not None:
    #     scheduler.step(epoch)

    # [loss_weight]
    if loss_weights is None:
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader, total=len(train_loader))):

        num_batches_per_epoch = len(train_loader)
        num_updates = epoch * num_batches_per_epoch

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = train_data['image'].float().to(device), train_data['label'].float().to(device)
        outputs = model(inputs)
        # loss functions
        for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
            # weight(lambda) * loss
            loss = loss_weight * loss_fn(outputs, labels.long())
            name = loss_fn.__class__.__name__
            loss_dict[name].update(loss.item())

            # total loss
            total_loss += loss

        # no_aux_loss
        loss_dict['TotalLoss'].update(total_loss.item())

        # metric
        for i, metric_fn in enumerate(metric_fns):
            metric = metric_fn(outputs, labels.long())
            name = metric_fn.__class__.__name__
            metric_dict[name].update(metric)

        # metric functions --> MetricMeter
        metricmeter.update(outputs, labels, from_logits=True)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # [PYTORCH_SCHED]
        scheduler.step()  # for dc_swim / abcnet

        # [TIMM_SCHED]
        # scheduler
        # if scheduler is not None:
        #     num_updates += 1
        #     scheduler.step_update(num_updates=num_updates)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'train/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value.avg, epoch + 1)

    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value, epoch + 1)

    # print(monitor)


# now monitor can handle both loss and metric
def aux_train(device, epoch, model, train_loader, optimizer, scheduler, loss_fns, metric_fns, monitor, writer, *args,
              loss_weights=None):
    # scheduler
    if scheduler is not None:
        scheduler.step(epoch)

    # [loss_weight]
    if loss_weights is None:
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'Aux_Output_Loss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader, total=len(train_loader))):

        num_batches_per_epoch = len(train_loader)
        num_updates = epoch * num_batches_per_epoch

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = train_data['image'].float().to(device), train_data['label'].float().to(device)
        outputs, aux_outputs, aux_loss = model(inputs)

        # loss functions
        for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
            # weight(lambda) * loss
            loss = loss_weight * (loss_fn(outputs, labels.long()) + loss_fn(aux_outputs, labels.long()))
            name = loss_fn.__class__.__name__
            loss_dict[name].update(loss.item())

            # total loss
            total_loss += loss

        # aux_loss
        total_loss += aux_loss
        loss_dict['AuxLoss'].update(aux_loss.item())
        loss_dict['TotalLoss'].update(total_loss.item())

        # metric
        for i, metric_fn in enumerate(metric_fns):
            metric = metric_fn(outputs, labels.long())
            name = metric_fn.__class__.__name__
            metric_dict[name].update(metric)

        # metric functions --> MetricMeter
        metricmeter.update(outputs, labels, from_logits=True)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # scheduler
        if scheduler is not None:
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'train/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value.avg, epoch + 1)

    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value, epoch + 1)

    # print(monitor)


# ABCNet
def aux_train_no_aux_loss(device, epoch, model, train_loader, optimizer, scheduler, loss_fns, metric_fns, monitor,
                          writer, *args, loss_weights=None):
    # scheduler
    if scheduler is not None:
        scheduler.step(epoch)

    # [loss_weight]
    if loss_weights is None:
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'Aux_Output_Loss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.train()
    for batch_idx, train_data in enumerate(tqdm(train_loader, total=len(train_loader))):

        num_batches_per_epoch = len(train_loader)
        num_updates = epoch * num_batches_per_epoch

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = train_data['image'].float().to(device), train_data['label'].float().to(device)
        outputs, aux_outputs_1, aux_outputs_2 = model(inputs)
        # feat_out, feat_out16, feat_out32

        # loss functions
        for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
            # weight(lambda) * loss
            loss = loss_weight * (
                        loss_fn(outputs, labels.long()) + loss_fn(aux_outputs_1, labels.long()) + loss_fn(aux_outputs_2,
                                                                                                          labels.long()))
            name = loss_fn.__class__.__name__
            loss_dict[name].update(loss.item())

            # total loss
            total_loss += loss

        # no_aux_loss
        loss_dict['TotalLoss'].update(total_loss.item())

        # metric
        for i, metric_fn in enumerate(metric_fns):
            metric = metric_fn(outputs, labels.long())
            name = metric_fn.__class__.__name__
            metric_dict[name].update(metric)

        # metric functions --> MetricMeter
        metricmeter.update(outputs, labels, from_logits=True)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # scheduler
        if scheduler is not None:
            num_updates += 1
            scheduler.step_update(num_updates=num_updates)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'train/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value.avg, epoch + 1)

    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[train] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'train/{metric_type}', metric_value, epoch + 1)

    # print(monitor)


# now monitor can handle both loss and metric
def val(device, epoch, model, val_loader, loss_fns, metric_fns, monitor, writer, save_path,
        *args, loss_weights=None):
    save_metrics = ['IoU_Score', 'F1_Score', 'MetricMeter_F1', 'MetricMeter_IoU']

    # [loss_weight]
    if loss_weights is None:  #
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.eval()
    for batch_idx, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
        # print(f'val {batch_idx}')

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = val_data['image'].float().to(device), val_data['label'].float().to(device)

        # print(f'{val_data["image_name"]=}')
        with torch.no_grad():
            outputs, aux_loss = model(inputs)

            # loss functions
            for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
                # weight(lambda) * loss
                loss = loss_weight * loss_fn(outputs, labels.long())
                name = loss_fn.__class__.__name__
                loss_dict[name].update(loss.item())

                # total loss
                total_loss += loss

            # aux_loss
            total_loss += aux_loss

            loss_dict['AuxLoss'].update(aux_loss.item())
            loss_dict['TotalLoss'].update(total_loss.item())

            for i, metric_fn in enumerate(metric_fns):
                metric = metric_fn(outputs, labels.long())
                name = metric_fn.__class__.__name__
                metric_dict[name].update(metric)

            # metric functions --> MetricMeter
            metricmeter.update(outputs, labels, from_logits=True)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[val] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'val/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[val] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'val/{metric_type}', metric_value.avg, epoch + 1)
    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[val] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'val/{metric_type}', metric_value, epoch + 1)

    for save_metric in save_metrics:
        if monitor.save(save_metric):
            # torch.save(model.state_dict(), f'{save_path}{save_metric}{epoch + 1}.pth') # save the num_epoch
            torch.save(model.state_dict(), f'{save_path}{save_metric}.pth')  # don't


    print(monitor)


def val_no_aux_loss(device, epoch, model, val_loader, loss_fns, metric_fns, monitor, writer, save_path,
                    *args, loss_weights=None):
    save_metrics = ['IoU_Score', 'F1_Score', 'MetricMeter_F1', 'MetricMeter_IoU']

    # [loss_weight]
    if loss_weights is None:  #
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.eval()
    for batch_idx, val_data in enumerate(tqdm(val_loader, total=len(val_loader))):
        # print(f'val {batch_idx}')

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = val_data['image'].float().to(device), val_data['label'].float().to(device)

        # print(f'{val_data["image_name"]=}')
        with torch.no_grad():
            outputs = model(inputs)

            # loss functions
            for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
                # weight(lambda) * loss
                loss = loss_weight * loss_fn(outputs, labels.long())
                name = loss_fn.__class__.__name__
                loss_dict[name].update(loss.item())

                # total loss
                total_loss += loss

            # no_aux_loss
            loss_dict['TotalLoss'].update(total_loss.item())

            for i, metric_fn in enumerate(metric_fns):
                metric = metric_fn(outputs, labels.long())
                name = metric_fn.__class__.__name__
                metric_dict[name].update(metric)

            # metric functions --> MetricMeter
            metricmeter.update(outputs, labels, from_logits=True)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[val] epoch:{epoch + 1:4}, {loss_type:10}: {loss_value.avg}\n')

        # monitor
        monitor.update(loss_type, epoch + 1, loss_value.avg, 'loss')

        # writer
        writer.add_scalar(f'val/{loss_type}', loss_value.avg, epoch + 1)

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[val] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value.avg}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value.avg, 'metric')

        # writer
        writer.add_scalar(f'val/{metric_type}', metric_value.avg, epoch + 1)
    # metricmeter.dict() MetricMeter
    for metric_type, metric_value in metricmeter.dict().items():
        print(f'[val] epoch:{epoch + 1:4}, {metric_type:10}: {metric_value}\n')

        # monitor
        monitor.update(metric_type, epoch + 1, metric_value, 'metric')

        # writer
        writer.add_scalar(f'val/{metric_type}', metric_value, epoch + 1)

    for save_metric in save_metrics:
        if monitor.save(save_metric):
            torch.save(model.state_dict(), f'{save_path}{save_metric}{epoch + 1}.pth')

    print(monitor)



# no monitor
# pred_path for  visualization save
def test(device, model, test_loader, loss_fns, metric_fns, pred_path, *args, loss_weights=None, palette=None):

    # palette
    if palette is 'potsdam' or 'vaihingen':
        palette = palette_potsdam

    elif palette == 'loveda':
        palette = palette_loveda

    elif palette == 'satellite':
        palette = palette_satellite

    save_metrics = ['IoU_Score', 'F1_Score', 'MetricMeter_F1', 'MetricMeter_IoU']

    # [loss_weight]
    if loss_weights is None:  #
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.eval()

    image_index = 0

    for batch_idx, test_data in enumerate(tqdm(test_loader, total=len(test_loader))):

        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = test_data['image'].float().to(device), test_data['label'].float().to(device)

        if torch.isinf(inputs).any() or torch.isnan(inputs).any():
            print(f"NaN or Inf : {test_data['label_name']}")

        with torch.no_grad():
            outputs, aux_loss = model(inputs)
            if torch.isinf(outputs).any() or torch.isnan(outputs).any():
                print(f"NaN or Inf : {test_data['label_name']}")
            # loss functions
            for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
                # weight(lambda) * loss
                loss = loss_weight * loss_fn(outputs, labels.long())
                # print(loss)
                name = loss_fn.__class__.__name__
                loss_dict[name].update(loss.item())

                # total loss
                total_loss += loss

            # aux_loss
            total_loss += aux_loss

            loss_dict['AuxLoss'].update(aux_loss.item())
            loss_dict['TotalLoss'].update(total_loss.item())

            for i, metric_fn in enumerate(metric_fns):
                metric = metric_fn(outputs, labels.long())
                name = metric_fn.__class__.__name__
                metric_dict[name].update(metric)

            # metric functions --> MetricMeter
            metricmeter.update(outputs, labels, from_logits=True)

            # visualization
            for (output, label) in zip(outputs, labels):

                # reshape
                output = output.view(1, *(output.shape))
                label = label.view(1, *(label.shape))

                image_visualization(output, label, pred_path, f'{image_index:04d}_', palette, ignore_index=255)
                image_index += 1
            # image_visualization(outputs, labels, pred_path, batch_idx, palette)

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[test] {loss_type:10}: {loss_value.avg}\n')

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[test] {metric_type:10}: {metric_value.avg}\n')

    print(metricmeter)


def test_no_aux_loss(device, model, test_loader, loss_fns, metric_fns, pred_path, *args, loss_weights=None, palette=None):

    # palette
    if palette is None:
        palette = palette_potsdam

    elif palette == 'loveda':
        palette = palette_loveda

    elif palette == 'satellite':
        palette = palette_satellite

    save_metrics = ['IoU_Score', 'F1_Score', 'MetricMeter_F1', 'MetricMeter_IoU']

    # [loss_weight]
    if loss_weights is None:  #
        loss_weights = [1] * len(loss_fns)
    assert len(loss_fns) == len(
        loss_weights), f'num_loss `{len(loss_fns)}` and num_weight `{len(loss_weights)}` not matching'

    """
    [AuxLoss] -> for intermediate part
    [TotalLoss] -> sum of all losses
    """
    # loss_dict
    loss_dict = {'AuxLoss': AverageMeter(), 'TotalLoss': AverageMeter()}
    for loss_fn in loss_fns:
        name = loss_fn.__class__.__name__
        loss_dict[name] = AverageMeter()

    # metric_dict
    metric_dict = {}
    for metric_fn in metric_fns:
        name = metric_fn.__class__.__name__
        metric_dict[name] = AverageMeter()

    metricmeter = MetricMeter()

    # model
    model = model.to(device)
    model.eval()

    image_index = 0

    for batch_idx, test_data in enumerate(tqdm(test_loader, total=len(test_loader))):
        # loss
        total_loss = torch.zeros(1).to(device)
        inputs, labels = test_data['image'].float().to(device), test_data['label'].float().to(device)

        if torch.isinf(inputs).any() or torch.isnan(inputs).any():
            print(f"NaN or Inf : {test_data['label_name']}")

        with torch.no_grad():
            outputs = model(inputs)
            if torch.isinf(outputs).any() or torch.isnan(outputs).any():
                print(f"NaN or Inf : {test_data['label_name']}")
            # loss functions
            for i, (loss_fn, loss_weight) in enumerate(zip(loss_fns, loss_weights)):
                # weight(lambda) * loss
                loss = loss_weight * loss_fn(outputs, labels.long())
                # print(loss)
                name = loss_fn.__class__.__name__
                loss_dict[name].update(loss.item())

                # total loss
                total_loss += loss

            # no_aux_loss
            loss_dict['TotalLoss'].update(total_loss.item())

            for i, metric_fn in enumerate(metric_fns):
                metric = metric_fn(outputs, labels.long())
                name = metric_fn.__class__.__name__
                metric_dict[name].update(metric)

            # metric functions --> MetricMeter
            metricmeter.update(outputs, labels, from_logits=True)

            # visualization
            for (output, label) in zip(outputs, labels):
                image_visualization(output, label, pred_path, f'{image_index:04d}', palette, ignore_index=5)
                image_index += 1

    # loss_dict AverageMeter
    for loss_type, loss_value in loss_dict.items():
        print(f'[test] {loss_type:10}: {loss_value.avg}\n')

    # metric_dict AverageMeter
    for metric_type, metric_value in metric_dict.items():
        print(f'[test] {metric_type:10}: {metric_value.avg}\n')

    print(metricmeter)


def train_val_test_split(dataset: list, train: float, val: float, test: float, random_state=42):
    if train + val + test != 1:  # need to normalize
        data_list = np.array([train, val, test])
        train, val, test = data_list / data_list.sum()  # proportion

    val_on_train_val = val / (train + val)
    train_val_data, test_data = train_test_split(dataset, test_size=test, random_state=random_state)
    train_data, val_data = train_test_split(train_val_data, test_size=val_on_train_val, random_state=random_state)

    return train_data, val_data, test_data


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# metric val
class MetricMeter:  # for metric
    def __init__(self):
        self.reset()

    def reset(self):
        self.cm = 0.

    def confusionmatrix_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor, threshold, from_logits):

        classes = y_pred.size(1)

        if from_logits:
            if classes == 1:
                # sigmoid
                y_pred = F.logsigmoid(y_pred).exp()
                # binarize
                # > threshold -> True, 1 ; <= threshold -> False, 0
                y_pred = (y_pred > threshold).float()
            else:
                # multi-class
                # Apply `torch.argmax` to get class index with max probabilities
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            # multi-label ?

        # check shape
        assert y_true.view(-1).shape == y_pred.view(-1).shape

        # 1-channel prediction but still a 2-classes [0, 1] problem
        if classes == 1:
            classes += 1

        # confusion matrix size
        matrix_size = (classes, classes)

        # confusion matrix index
        cm_index = np.arange(classes)

        # tensor flatten & to cpu & to numpy
        y_true = y_true.view(-1).cpu().numpy()
        y_pred = y_pred.view(-1).cpu().numpy()

        # unique
        y_true_unique = np.unique(y_true)

        # check is there `any` index in y_true is also in cm_index
        if not np.isin(y_true_unique, cm_index).any():
            # prevent `ValueError: At least one label specified must be in y_true`
            cm = np.zeros(matrix_size)
        else:
            cm = confusion_matrix(y_true, y_pred, labels=cm_index)

        return cm

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, threshold=0.5, from_logits=False, n=1):

        self.cm += self.confusionmatrix_tensor(y_pred, y_true, threshold, from_logits)

        # tp, tn, fp, fn
        tp = np.diag(self.cm)
        fp = np.sum(self.cm, axis=0) - np.diag(self.cm)
        fn = np.sum(self.cm, axis=1) - np.diag(self.cm)
        tn = np.sum(self.cm) - np.sum(self.cm, axis=0) - np.sum(self.cm, axis=1) + np.diag(self.cm)

        with np.errstate(divide='ignore', invalid='ignore'):
            """"
            self.o_acc = (tp / (tp + fp + tn + fn)).sum()
            self.prec_mean = (tp / (tp + fp)).mean()
            self.rec_mean = (tp / (tp + fn)).mean()
            self.f1_mean = (tp / (tp + 0.5 * (fp + fn))).mean()
            self.iou_mean = (tp / (tp + fp + fn)).mean()

            self.o_acc_list = tp / (tp + fp + tn + fn)
            self.prec_list = tp / (tp + fp)
            self.rec_list = tp / (tp + fn)
            self.f1_list = tp / (tp + 0.5 * (fp + fn))
            self.iou_list = tp / (tp + fp + fn)

            """
            # [all classes]
            # For multi-class classification, per-class accuracy is the same as per-class recall
            self.acc_list = self.rec_list = tp / (tp + fn)
            self.prec_list = tp / (tp + fp)
            self.f1_list = tp / (tp + 0.5 * (fp + fn))
            self.iou_list = tp / (tp + fp + fn)
            self.dice_list = 2 * tp / (2 * tp + fp + fn)

            self.acc_mean = self.rec_mean = self.rec_list.mean()
            self.prec_mean = self.prec_list.mean()
            self.f1_mean = self.f1_list.mean()
            self.iou_mean = self.iou_list.mean()
            self.dice_mean = self.dice_list.mean()

            # overall
            self.o_acc_list = tp / (tp + fp + tn + fn)
            self.o_acc = self.o_acc_list.sum()

            # [select_classes]
            # self.acc_list_s = self.rec_list_s = self.rec_list[self.select]
            # self.prec_list_s = self.prec_list[self.select]
            # self.f1_list_s = self.f1_list[self.select]
            # self.iou_list_s = self.iou_list[self.select]
            # self.dice_list_s = self.dice_list[self.select]
            #
            # self.acc_mean_s = self.rec_mean_s = self.rec_list_s.mean()
            # self.prec_mean_s = self.rec_list_s.mean()
            # self.f1_mean_s = self.f1_list_s.mean()
            # self.iou_mean_s = self.iou_list_s.mean()
            # self.dice_mean_s = self.dice_list_s.mean()

    def __str__(self):
        # return " | ".join()
        return f'\n' \
               f'[Value]\n' \
               f'\n' \
               f'Overall Accuracy : {self.o_acc}\n' \
               f'Precision mean   : {self.prec_mean}\n' \
               f'Recall mean      : {self.rec_mean}\n' \
               f'F1_score mean    : {self.f1_mean}\n' \
               f'IoU mean         : {self.iou_mean}\n' \
               f'\n' \
               f'[List]\n' \
               f'\n' \
               f'Overall Accuracy List: {self.o_acc_list}\n' \
               f'Precision List       : {self.prec_list}\n' \
               f'Recall List          : {self.rec_list}\n' \
               f'F1_score List        : {self.f1_list}\n' \
               f'IoU List             : {self.iou_list}\n' \
               f'\n' \
               f'Confusion Matrix :\n' \
               f'\n' \
               f'{self.cm}\n'

    def dict(self):
        return {'MetricMeter_F1': self.f1_mean, 'MetricMeter_IoU': self.iou_mean}


"""Stores the loss / metric value and epoch"""

# storing the best epoch for each loss and metric throughout train/val
# __str__ and save
# x.`save(metric_name)` will show in this epoch should save model or not
class EpochMonitor:
    """Stores the best (performance) loss / metric value and epoch"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"epoch": 0, "value": 0., "save": False})

    def update(self, metric_name: str, epoch: int, value: float, format: str, n=1):

        assert format in ['loss', 'metric']
        metric = self.metrics[metric_name]

        # if first epoch
        # or if metric value is greater than previous
        # or if loss   value is fewer   than previous

        if epoch == 1 or \
                value > metric["value"] and format == 'metric' or value < metric["value"] and format == 'loss':
            metric["epoch"] = epoch
            metric["value"] = value
            metric["save"] = True
        else:
            metric["save"] = False

    def __str__(self):
        return f'\n'.join(
            f'[{metric_name:10} Best Epoch: {metric_value["epoch"]:4}, Best Value: {metric_value["value"]}\n'
            for (metric_name, metric_value) in self.metrics.items())

    # save when better outcome (metric["save"] = True)
    def save(self, metric_name):
        return self.metrics[metric_name]["save"]



def convert_to_color(arr_2d, palette):
    """ Numeric labels to RGB-color encoding """
    # index to colored
    # [B, H, W] --> [B, H, W, 3]
    # print(arr_2d.shape)

    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], arr_2d.shape[2], 3), dtype=np.uint8)
    for c, i in palette.items():
        # m is mask (label)
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


def image_visualization(y_pred, y_true, save_folder, save_name, palette=palette_potsdam, mode='multiclass',
                        from_logits=True, threshold=0.5, ignore_index=None):
    assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

    if from_logits:
        if mode == 'multiclass':  # MULTICLASS_MODE -> argmax
            y_pred = torch.argmax(y_pred, dim=1)
        else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid & binarize
            y_pred = F.logsigmoid(y_pred).exp()
            y_pred = (y_pred > threshold).float()

    pred_np = y_pred.detach().cpu().numpy().astype('int')
    label_np = y_true.detach().cpu().numpy().astype('int')

    # ignore map [True = not_ignore | False = ignore]
    ignore_map = ~(label_np == ignore_index)
    # 1 channel to 3 channels(same shape as after `convert_to_color`)
    # https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
    ignore_map = np.stack((ignore_map,) * 3, axis=-1)

    # index to RGB
    label_original = convert_to_color(label_np, palette)
    pred_original = convert_to_color(pred_np, palette)

    # apply ignore
    label_apply_ignore = label_original * ignore_map
    pred_apply_ignore = pred_original * ignore_map

    # check shape
    assert label_original.shape == pred_original.shape

    # batch
    batch_size = label_original.shape[0]

    for batch in range(batch_size):
        label_per_batch = label_apply_ignore[batch]
        pred_per_batch = pred_apply_ignore[batch]

        # print(label_per_batch.shape)
        # print(pred_per_batch.shape)

        # https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
        im_label = Image.fromarray((label_per_batch * 1).astype(np.uint8)).convert('RGB')
        im_pred = Image.fromarray((pred_per_batch * 1).astype(np.uint8)).convert('RGB')

        im_label.save(f'{save_folder}{save_name}label.jpeg')
        im_pred.save(f'{save_folder}{save_name}predict.jpeg')


