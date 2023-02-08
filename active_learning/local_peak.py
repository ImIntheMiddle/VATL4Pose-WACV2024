import numpy as np
from scipy.ndimage import maximum_filter

# ピーク検出関数
def localpeak_values(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))
    # print(detected_peaks.max() * order)
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order)) # 小さいピーク値を排除（最大ピーク値のorder倍以下のピークは排除）
    return temp.compressed()

def localpeak_mean(heatmaps, filter_size=3, order=0.5):
    # compute the mean of all local peaks
    # heatmaps : [num_joints, height, width]
    local_peaks = []
    for heatmap in heatmaps:
        local_peaks.append(localpeak_values(heatmap, filter_size, order))
    # print(local_peaks)
    local_peaks = np.hstack(local_peaks)
    # print(local_peaks)
    mean_value = local_peaks.mean()
    return mean_value

# test
if __name__ == "__main__":
    heatmap = np.array([[0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 3, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 2, 2, 0, 0, 0, 0]])
    localpeaks_values = localpeak_values(heatmap)
    print(localpeaks_values)
    print("min:", localpeaks_values.min())

    heatmaps = np.array([heatmap, heatmap, heatmap])
    min_value = find_minimum_peak(heatmaps)
    print(min_value)
