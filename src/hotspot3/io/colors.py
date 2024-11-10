import matplotlib.pyplot as plt


def get_bed9_color(fdr, mode='peaks'):
    thresholds = [0.001, 0.01, 0.05, 0.1]
    cmap_tab20c = plt.get_cmap("tab20c")

    if mode == 'peaks':
        indices = [8, 9, 10, 11]
    elif mode == 'hotspots':
        indices = [12, 13, 14, 15]
    else:
        raise ValueError("Mode must be either 'peaks' or 'hotspots'.")

    colors = [
        *[tuple(int(c * 255) for c in cmap_tab20c(i)[:3]) for i in indices], (192, 192, 192)
    ]
    
    assert len(colors) == len(thresholds) + 1, "Number of colors must be one more than the number of thresholds."
    for threshold, color in zip(thresholds, colors):
        if fdr <= threshold:
            return "{},{},{}".format(*color)
    return "{},{},{}".format(*colors[-1])