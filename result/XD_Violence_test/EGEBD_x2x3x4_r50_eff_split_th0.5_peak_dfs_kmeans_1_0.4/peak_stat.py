
import json

def find_peaks_and_analyze(json_file_path, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    查找每个视频中置信度的peak，并统计不同阈值下的peak数量。
    """
    all_peaks = []
    peak_counts = {threshold: 0 for threshold in thresholds}

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for video_name, video_data in data.items():
        confidence_scores = video_data['pred']
        n = len(confidence_scores)
        for i in range(n):
            left = confidence_scores[i - 1] if i - 1 >= 0 else float('-inf')
            right = confidence_scores[i + 1] if i + 1 < n else float('-inf')
            curr = confidence_scores[i]
            # 当前帧大于等于左右两边
            if curr >= left and curr >= right:
                all_peaks.append((video_name, i, curr))
                for threshold in thresholds:
                    if curr >= threshold:
                        peak_counts[threshold] += 1

    return all_peaks, peak_counts

if __name__ == "__main__":
    json_file_path = '/root/autodl-tmp/data/XD_test/EGEBD_x2x3x4_r50_eff_split_th0.5_peak_dfs_kmeans_1_0.4/pred.json'
    all_peaks, peak_counts = find_peaks_and_analyze(json_file_path)

    print("All Peaks (video_name, frame_index, confidence_score):")
    for peak in all_peaks:
        print(peak)

    print("\nPeak Counts at Different Thresholds:")
    for threshold, count in peak_counts.items():
        print(f"Threshold {threshold}: {count} peaks")