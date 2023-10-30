#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def label_accuracy(ground_truth, predicted, mapping):
    # Get unique labels from ground truth and predictions
    unique_labels = set(ground_truth + predicted)
    
    accuracy_per_label = {}
    
    for label in unique_labels:
        # True positives
        tp = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == label and pred == label)
        
        # False positives
        fp = sum(1 for gt, pred in zip(ground_truth, predicted) if gt != label and pred == label)
        
        # False negatives
        fn = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == label and pred != label)
        
        # True negatives
        # tn = sum(1 for gt, pred in zip(ground_truth, predicted) if gt != label and pred != label)
        
        # accuracy = (tp + tn) / (tp + fp + fn + tn)
        accuracy = 2*tp / (2*tp + fp + fn)
        
        accuracy_per_label[mapping[label]] = round(accuracy, 4)
    
    return accuracy_per_label

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], dtype=np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    mapping = {
        "SIL": "0",
        "reaching_for_needle_with_right_hand": "1",
        "positioning_needle": "2",
        "pushing_needle_through_tissue": "3",
        "transferring_needle_from_left_to_right": "4",
        "moving_to_center_with_needle_in_grip": "5",
        "pulling_suture_with_left_hand": "6",
        "pulling_suture_with_right_hand": "7",
        "orienting_needle": "8",
        "using_right_hand_to_help_tighten_suture": "9",
        "loosening_more_suture": "10",
        "dropping_suture_at_end_and_moving_to_end_points": "11",
        "reaching_for_needle_with_left_hand": "12",
        "making_C_loop_around_right_hand": "13",
        "reaching_for_suture_with_right_hand": "14",
        "pulling_suture_with_both_hands": "15"
    }
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--root', default='')

    args = parser.parse_args()

    ground_truth_path = args.root+args.dataset+"/groundTruth/"
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = args.root+args.dataset+"/splits/test.split"+args.split+".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0
    gt_all =[]
    recog_all = []

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        gt_all.extend(gt_content)

        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
        recog_all.extend(recog_content)

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
    print(label_accuracy(gt_all, recog_all, mapping))
    print("Acc: %.4f" % (100*float(correct)/total))
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

if __name__ == '__main__':
    main()
