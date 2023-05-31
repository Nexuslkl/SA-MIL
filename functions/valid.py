import torch
from sklearn import metrics

def valid(model, dataloader, args):
    step = 0
    num = 0
    total_f = 0

    with torch.no_grad():
        for image, label in dataloader:
            step += 1
            if step > args.test_num_pos:
                break
            num += 1

            preds = model(image.to(torch.device(args.device)))
            pred = ((preds[0] >= 0.5) + 0).to('cpu').numpy()
            label = label.int().to("cpu").numpy()

            if label.sum().item() > 0:
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
            else:
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)

            total_f += f1
        average_f = total_f/num

        return average_f