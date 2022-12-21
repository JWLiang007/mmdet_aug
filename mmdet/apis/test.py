# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os 
import pickle
import shutil
import tempfile
import time

import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.utils.visual_utils import tsne_fig
from mmdet.core import encode_mask_results
from mmdet.core.post_processing.bbox_nms import multiclass_nms

feature_buffer = []
# fc_feat_buffer = []
label_buffer = []
def regitster_hooks(p_module,prefix = 'tsne'):
    def hook_bbox_head_forward(module, input, output):
        # global feature_buffer
        bbox_feature  = input[0]
        cls_score ,bbox_pred= output
        cls_score = F.softmax(cls_score, dim=-1)
        p_module.register_buffer(prefix+'_'+'cls_score',cls_score)
        p_module.register_buffer(prefix+'_'+'bbox_feature',bbox_feature)
        p_module.register_buffer(prefix+'_'+'bbox_pred',bbox_pred)
        # feature_buffer['cls_score'] = cls_score
        # feature_buffer['bbox_feature'] = bbox_feature
        # feature_buffer['bbox_pred'] = bbox_pred
        pass

    def hook_bbox_extracter_forward(module, input, output):
        # global feature_buffer
        rois = input[1]
        p_module.register_buffer(prefix+'_'+'rois',rois)
        # feature_buffer['rois'] = rois
        pass

    def hook_bbox_fc_cls_forward(module, input, output):
        # global feature_buffer
        fc_cls = input[0]
        p_module.register_buffer(prefix+'_'+'fc_cls',fc_cls)
        # feature_buffer['rois'] = rois
        pass

    return hook_bbox_head_forward,hook_bbox_extracter_forward,hook_bbox_fc_cls_forward

def get_feature(module,meta,test_cfg,prefix = 'tsne'):

    # buffer_dict = dict(module.named_buffers())
    bbox_pred = module.get_buffer(prefix+'_'+'bbox_pred')
    rois = module.get_buffer(prefix+'_'+'rois')
    scores = module.get_buffer(prefix+'_'+'cls_score')
    bbox_feature = module.get_buffer(prefix+'_'+'bbox_feature')
    fc_cls_feature = module.get_buffer(prefix+'_'+'fc_cls')

    img_shape = meta['img_shape']
    scale_factor = meta['scale_factor']

    modules = dict(module.named_modules())
    suffix = '' if not hasattr(module.roi_head,'num_stages') else "."+str(module.roi_head.num_stages-1)
    bboxes = modules['roi_head.bbox_head'+suffix].bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
    if scale_factor is not None and bboxes.size(0) > 0:
        scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
            bboxes.size()[0], -1)
    det_bboxes, det_labels,ids = multiclass_nms(bboxes, scores,
                                                    # test_cfg.score_thr,
                                                    0.7,
                                                    test_cfg.nms,
                                                    test_cfg.max_per_img,return_inds=True)
    if det_labels.shape[0] == 0:
        return
    global feature_buffer, label_buffer
    # fc_feat_buffer.append( fc_cls_feature[ids//(scores.size()[1]-1)].flatten(1)  )
    feature_buffer.append( fc_cls_feature[ids//(scores.size()[1]-1)].flatten(1)  )

    label_buffer.append(det_labels)
    # return bbox_feature[ids//scores.size()[1]].flatten(1) ,det_labels

def plot_tsne(module,save_path,cls_list=None,num_per_cls=-1):
    global feature_buffer, label_buffer
    classes = copy.deepcopy(module.CLASSES)

    features = torch.cat(feature_buffer,0)
    labels = torch.cat(label_buffer,0)
    # sort_idx = torch.argsort(labels)
    indicator = torch.zeros_like(labels) == 0
    num_cls = len(classes)
    if cls_list is None or len(cls_list) == 0:
        cls_list=list(range(len(classes)))
    # if cls_list is not None and len(cls_list) > 0 :
    num_cls = len(cls_list)
    indicator = torch.zeros_like(labels) != 0
    for sub_cls in cls_list:
        idx = torch.where(labels == sub_cls)[0]
        idx = idx[torch.randperm(len(idx))[:num_per_cls]]

        indicator[idx] = True
        # classes = classes[cls_list]
    features = features[indicator]
    labels = labels[indicator]
    # classes = args.class_names + ["poisoned"]
    sort_idx = torch.argsort(labels)
    features = features[sort_idx]
    labels = labels[sort_idx]
    label_class = [classes[i].capitalize() for i in labels]

    # Plot T-SNE
    custom_palette = sns.color_palette("hls", num_cls)
    fig = tsne_fig(
        features,
        label_class,
        title="t-SNE Embedding",
        xlabel="Dim 1",
        ylabel="Dim 2",
        custom_palette=custom_palette,
        size=(10, 10),
        n_iter = 1000,
    )
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    args=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    is_plot_tsne = False
    if args is not None :
        is_plot_tsne = getattr(args,'plot_tsne', False)
    if is_plot_tsne:
        modules = dict(model.module.named_modules())
        hook_bbox_head_forward,hook_bbox_extracter_forward,hook_bbox_fc_cls_forward = regitster_hooks(model.module)
        suffix = '' if not hasattr(model.module.roi_head,'num_stages') else "."+str(model.module.roi_head.num_stages-1)
        modules['roi_head.bbox_head'+suffix].register_forward_hook(hook_bbox_head_forward)
        modules['roi_head.bbox_roi_extractor'+suffix].register_forward_hook(hook_bbox_extracter_forward)
        modules['roi_head.bbox_head'+suffix+'.fc_cls'].register_forward_hook(hook_bbox_fc_cls_forward)
    # model.module.roi_head.bbox_head.register_forward_hook(hook_bbox_head_forward)
    # model.module.roi_head.bbox_roi_extractor.register_forward_hook(hook_bbox_extracter_forward)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if is_plot_tsne :
                get_feature(model.module,meta = data['img_metas'][0].data[0][0],test_cfg=model.module.test_cfg['rcnn'])
            # if i % 10 == 9:
            #     plot_tsne(model.module)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    if is_plot_tsne:
        plot_tsne(model.module,args.tsne_save_path,cls_list=args.tsne_cls)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
