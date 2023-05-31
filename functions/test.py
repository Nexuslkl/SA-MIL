import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import hausdorff, gm


def test(model, dataloader, args):
    checkpoint = args.checkpoint
    if checkpoint is None:
        path_model = os.path.join(args.work_path, 'best_model.pth')
    elif checkpoint == 'final':
        path_model = os.path.join(args.work_path, 'final_model.pth')
    else:
        path_model = os.path.join(args.work_path, checkpoint)

    model.load_state_dict(torch.load(path_model))
    model.eval()

    step = 0
    num_1 = 0
    num_0 = 0
    total_f_1 = 0
    total_f_0 = 0
    total_hd = 0
    average_f_1 = 0
    average_hd = 0
    average_f_0 = 0
    num_cls = 0

    with torch.no_grad():
        for image, label, image_show, name in dataloader:
            step += 1
            print('%dth' % step)

            preds = model(image.to(args.device))
            pred = ((preds[0] >= 0.5) + 0).squeeze(0).squeeze(0).to('cpu').numpy()

            label = label.squeeze(0).squeeze(0).int().to("cpu").numpy()
            image_show = image_show.squeeze(0).int().to("cpu").numpy()

            if step <= args.test_num_pos:
                num_1 += 1
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
                total_f_1 += f1
                average_f_1 = total_f_1 / num_1
                hausdorff_distance = hausdorff(pred, label)
                total_hd += hausdorff_distance
                average_hd = total_hd / num_1
            else:
                num_0 += 1
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)
                total_f_0 += f1
                average_f_0 = total_f_0 / num_0

            pre_cls = gm(preds[0], args.r).squeeze(0).item() >= 0.5 + 0
            gdt_cls = 1 if label.sum() > 0 else 0
            if pre_cls == gdt_cls:
                num_cls += 1

        print("F1 Pos = %.3f" % average_f_1)
        print("average HD = %.3f" % average_hd)
        print("F1 Neg = %.3f" % average_f_0)
        print("class accuracy = {:.3f}".format(num_cls/step))

