import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms

from model import E2EModel


def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video


def load_video(video_path, dim=224, sample_rate=3):
    vidcap = cv2.VideoCapture(video_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    if (width == 0) or (height == 0):
        print(video_path, 'not successfully loaded, drop ..')
        quit()

    images = []
    frame_indices = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    success, image = vidcap.read()
    count = 1
    while success:
        if count % sample_rate == 1:
            image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(transform(image))
            frame_indices.append(count)
        success, image = vidcap.read()
        count += 1
    vidcap.release()

    images = torch.stack(images, dim=0)
    return images, frame_indices, fps, nb_frames


@torch.no_grad()
def main(args):


    imgs, frame_indices, fps, total_frames = load_video(args.video)
    seconds = total_frames / fps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = E2EModel()
    state_dict = torch.load('E2EModel.pth')
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    inputs = imgs.to(device)[None]
    scores = model(inputs)[0].cpu().numpy()

    threshold = args.threshold
    det_t = np.array(get_idx_from_score_by_threshold(threshold=threshold,
                                                     seq_indices=frame_indices,
                                                     seq_scores=scores)) / fps

    print(f'Video path: {args.video}')
    print(f'Sampled frames: {len(frame_indices)}')
    print(f'Video FPS: {fps:.0f}')
    print(f'Video Duration: {seconds:.1f}s')
    print(f'Number Boundaries: {len(det_t)}')
    print(list(map(lambda t: round(t, 1), det_t)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default='/root/autodl-fs/lwl/data/UCF_Crime_test/test_videos/Abuse030_x264.mp4')
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()
    main(args)
