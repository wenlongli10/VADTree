'''
/root/autodl-tmp/data/XD_test/EGEBD_x2x3x4_r50_eff_split_th0.5_peak_dfs_kmeans_1_0.4/dfs_fine_sences.json和/root
/autodl-tmp/data/XD_test/EGEBD_x2x3x4_r50_eff_split_th0.5_peak_dfs_kmeans_1_0.4/dfs_coarse_sences.json
中的数据为如下所示的包含多个视频的字典：
{
    "Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4": {
        "scenes": [
            [
                0,
                2
            ],
            [
                2,
                24
            ],
...
其中每个视频的键对应一个字典，字典中有一个键"scenes"，其值是一个列表，列表中的每个元素是一个包含两个整数的列表，表示视频片段的起始和结束时间（单位为帧）。你要记录两个json
中的所有视频的边界位置帧，并分别统计他们的数量，然后两个json中的同一视频边界位置帧进行去重，再统计两个json中总的边界数量

'''

import json

def process_scene_boundaries(fine_json_path, coarse_json_path):
    """
    Processes scene boundaries from two JSON files, counts them, removes duplicates, and calculates the total count.

    Args:
        fine_json_path (str): Path to the JSON file containing fine-grained scene boundaries.
        coarse_json_path (str): Path to the JSON file containing coarse-grained scene boundaries.

    Returns:
        tuple: A tuple containing:
            - fine_count (int): Number of scene boundaries in the fine-grained JSON.
            - coarse_count (int): Number of scene boundaries in the coarse-grained JSON.
            - total_unique_count (int): Total number of unique scene boundaries after removing duplicates.
    """

    def extract_boundaries(json_path):
        """Extracts scene boundaries from a JSON file."""
        boundaries = set()
        with open(json_path, 'r') as f:
            data = json.load(f)
            for video_data in data.values():
                for scene in video_data['scenes']:
                    boundaries.add(scene[0])  # Start frame
                    boundaries.add(scene[1])  # End frame
        return boundaries

    fine_boundaries = extract_boundaries(fine_json_path)
    coarse_boundaries = extract_boundaries(coarse_json_path)

    fine_count = len(fine_boundaries)
    coarse_count = len(coarse_boundaries)

    total_unique_boundaries = fine_boundaries.union(coarse_boundaries)
    total_unique_count = len(total_unique_boundaries)

    return fine_count, coarse_count, total_unique_count


if __name__ == "__main__":
    fine_json_path = '/root/autodl-tmp/data/XD_test/EGEBD_x2x3x4_r50_eff_split_th0.5_peak_dfs_kmeans_1_0.4/dfs_fine_sences.json'
    coarse_json_path = '/root/autodl-tmp/data/XD_test/EGEBD_x2x3x4_r50_eff_split_th0.5_peak_dfs_kmeans_1_0.4/dfs_coarse_sences.json'

    fine_count, coarse_count, total_unique_count = process_scene_boundaries(fine_json_path, coarse_json_path)

    print(f"Number of scene boundaries in fine-grained JSON: {fine_count}")
    print(f"Number of scene boundaries in coarse-grained JSON: {coarse_count}")
    print(f"Total number of unique scene boundaries: {total_unique_count}")