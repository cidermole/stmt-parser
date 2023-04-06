import numpy as np
import cv2


def pad_image(img, pad_size):
    """Pad 8-bit RGB image on all sides with pad_size white pixels."""
    out = np.zeros((img.shape[0] + 2*pad_size, img.shape[1] + 2*pad_size, 3), dtype=img.dtype)
    out[pad_size:-pad_size, pad_size:-pad_size] = img
    out[0:pad_size,:] = 255
    out[(pad_size+img.shape[0]):,:] = 255
    out[:,0:pad_size] = 255
    out[:,(pad_size+img.shape[1]):] = 255
    return out


def int_odd(n):
    """:return: the next odd integer >= n."""
    i = int(n)
    if i % 2 == 0: i += 1
    return i


def normalize(a):
    """
    :param a: greyscale image
    :return: image normalized to brightness values [0.0-1.0]"""
    l, u = np.min(a), np.max(a)
    if (u - l) == 0:
        # for all-white, return white (1.0)
        # for all-black, return black (0.0)
        return a / a.flatten()[0] if a.flatten()[0] != 0 else a
    else:
        return (a - l) / (u - l)


def greyscale(a):
    """Turn an RGB image into greyscale values."""
    assert len(a.shape) == 3  # 2d image with colors ^= 3 dim array
    assert a.shape[2] == 3  # rgb on axis 2
    return np.average(a, axis=2)  # average rgb


def remove_median_bg(cfg, a):
    """Remove the median background of an RGB image."""
    m = cv2.medianBlur(a, int_odd(cfg['font_size'] * 1.0))  # blur factor
    rn = normalize(greyscale(a))
    mn = normalize(greyscale(m))
    #dn = normalize(rn - mn)
    dn = np.clip(rn - mn, -1.0, 0) + 1.0
    return dn


def letter_and_mask(l, w_max, h_max):
    """Center the letter in a mask box of size (h_max, w_max)."""
    # TODO: Step0_FetchFont-Copy1.ipynb has a newer version
    assert l.shape[0] < h_max
    assert l.shape[1] < w_max
    lo = np.ones((h_max, w_max), dtype=l.dtype)
    mo = np.zeros((h_max, w_max), dtype=l.dtype)
    pad_left, pad_top = (w_max - l.shape[1]) // 2, (h_max - l.shape[0]) // 2
    lo[pad_top:(pad_top+l.shape[0]), pad_left:(pad_left+l.shape[1])] = l
    mo[pad_top:(pad_top+l.shape[0]), pad_left:(pad_left+l.shape[1])] = 1
    return lo, mo


def number_matrix(a):
    """Number connected components in a coincidence matrix."""
    n = np.ones(a.shape[0], dtype=np.int64) * -1
    m = 0
    for i in range(a.shape[0]):
        if n[i] == -1:
            n[i] = m
            m += 1
        for j in range(i+1, a.shape[1]):
            if a[i,j] == 1 and n[j] == -1:
                n[j] = n[i]
    return n


def segment_letters_1(cfg, dn):
    """
    Detect letters used in an image.
    :param dn: background-removed padded greyscale image
    :returns: (nb_components, output, stats, centroids)
    """

    #
    # pre-filtering (visual filters)
    #
    
    # create kernel for vertically wide clustering of pixels ('i', '!', '?', ...)
    vg, hg = int(cfg['vert_gap_max'] * cfg['font_size']), 1
    kernel = np.ones((vg, hg))
    # smudge vertically
    f = cv2.filter2D(dn, -1, kernel) / np.sum(kernel)
    # threshold greyscale into binary image
    _, t = cv2.threshold(f, 1.0 - 1.0 / np.sum(kernel), 1.0, cv2.THRESH_BINARY)
    u = 1 - t.astype(np.int8)

    #
    # detect connected components
    #
    nb_components_p, outputs_p, stats_p, centroids_p = cv2.connectedComponentsWithStats(u, connectivity=8)

    #
    # post-filtering (select only plausible shapes)
    #

    # keep only characters, skip too large area symbols (image?)
    w_max, h_max = int(cfg['ligature_max_width'] * cfg['font_size']), int(cfg['largecap_max_size'] * cfg['font_size'])
    w_min, h_min = int(cfg['min_width'] * cfg['font_size']), int(min(cfg['punct_min_size'], cfg['smallcap_min_size']) * cfg['font_size'])

    outputs, stats, centroids = [], [], []
    for i, output, stat, centroid in zip(range(nb_components_p), outputs_p, stats_p, centroids_p):
        if i == 0: continue  # skip background
        x0, y0, w, h, net_area = stat
        if w > w_max or h > h_max: continue  # skip too large area symbols (image?)
        if w < w_min or h < h_min: continue  # skip too small area symbols (noise?)
        outputs.append(output)
        stats.append(stat)
        centroids.append(centroid)
    nb_components = len(outputs)

    return (nb_components, outputs, stats, centroids)


def vizz_segmentation(cfg, img, nb_components, output, stats, centroids):
    img_out = img.copy()
    c = np.array([255, 0, 0])  # red boxes

    letters = []
    for i, stat in zip(range(nb_components), stats):
        x0, y0, w, h, net_area = stat
    
        x1, y1 = x0 + w - 1, y0 + h - 1
        img_out[y0, x0:x1] = c
        img_out[y1, x0:x1] = c
        img_out[y0:y1, x0] = c
        img_out[y0:y1, x1] = c

    return img_out


def segment_letters_2(cfg, dn, nb_components, output, stats, centroids):
    """
    Collect standardized images of actual letters in an image.
    """

    #
    # unpack letters
    #

    letters = []
    for i, stat in zip(range(nb_components), stats):
        x0, y0, w, h, net_area = stat
        x1, y1 = x0 + w - 1, y0 + h - 1
        letters.append(dn[y0:y1, x0:x1])
    
    #
    # standardize letters to the same size
    #
    letters_norm = [letter_and_mask(l, w_max, h_max)[0] for l in letters]
    masks_norm = [letter_and_mask(l, w_max, h_max)[1] for l in letters]
    
    return letters, letters_norm, masks_norm


def segment_letters_3(cfg, letters, letters_norm, masks_norm):
    """
    Compute error matrix of similarity between actual letters.
    """

    #
    # compute error matrix of similarity
    #
    font_weight = int(np.ceil(cfg['font_weight'] * cfg['font_size']))
    krn = np.ones((font_weight, font_weight))
    ks = np.sum(krn)

    errs = np.zeros((len(letters), len(letters)), dtype=np.float64)
    for i in range(len(letters)):
        for j in range(len(letters)):
            # todo: change to use function letter_error()
            # boolean-or the masks of the two letters
            m = ((masks_norm[i] + masks_norm[j]) > 0).astype(np.float64)
            # sum of squared errors
            e = cv2.filter2D(letters_norm[i]*m - letters_norm[j]*m, -1, krn) / ks
            errs[i,j] = np.sqrt(np.sum((e)**2) / np.sum(m))
    
    return (letters_norm, masks_norm, errs)


def segment_letters(cfg, dn):
    nb_components, output, stats, centroids = segment_letters_1(cfg, dn)
    letters, letters_norm, masks_norm = segment_letters_2(cfg, nb_components, output, stats, centroids)
    letters, masks, errs = segment_letters_3(cfg, letters, letters_norm, masks_norm)
    return (letters, masks, errs)


def line_letter_stats(cfg, height, nb_components, stats):
    """
    Compute the frequency of letters on each pixel line of the document.
    """
    line_letter_freq = np.zeros(height, dtype=np.int64)

    for i, stat in zip(range(nb_components), stats):
        x0, y0, w, h, net_area = stat
        x1, y1 = x0 + w - 1, y0 + h - 1
        
        for j in range(y0, y1+1):
            line_letter_freq[j] += 1

    return line_letter_freq


def conv_padded(f, krn):
    """Like np.convolve(f, krn, mode='same'), but with constant edge padding."""
    fp = np.pad(f, (krn.shape[0], krn.shape[0]), mode='edge')
    c = np.convolve(fp, krn, mode='same')
    return c[(krn.shape[0]):(krn.shape[0]+f.shape[0])]


def line_detector(cfg, line_letter_freq):
    """
    Detect vertical text line base positions.
    """
    # criticism: this uses an average-filtering - hence, vertical positional outliers like 'g' influence the text base line position.
    # idea: it would be more robust to use a median filtering (throw away outliers). Don't ask me for windowed-median filter in numpy.
    # cf. scipy.ndimage.median_filter()
    # quick fix: increase detection threshold (cf. CharToolsDev.ipynb Out[52])

    # krn_gap_smooth: letter count normalization
    # * shift by (0.5 * line_spacing + vert_gap)
    # * smooth by font_size
    i_shift = int(0.5 * cfg['line_spacing']*cfg['font_size']) + int(cfg['vert_gap_max']*cfg['font_size'])

    ks = i_shift + int(cfg['font_size'])  # i_shift zeros, font_size ones
    krn_gap_smooth = np.zeros(ks, dtype=np.float64)
    krn_gap_smooth[i_shift:] = 1.0
    krn_gap_smooth /= np.sum(krn_gap_smooth)

    # krn_spacing: line spacing detector
    # * smooth by line_bottom_margin
    krn_spacing = np.ones(int(cfg['sep_bottom_margin'] * cfg['font_size']), dtype=np.float64)
    krn_spacing /= np.sum(krn_spacing)

    ls = conv_padded(line_letter_freq, krn_spacing)
    norm = conv_padded(np.clip(line_letter_freq, a_min=1e-6, a_max=1e6), krn_gap_smooth)
    t = ls * 1.0 / norm

    # find negative transitions through threshold
    threshold = 0.75
    pos = np.roll(t > threshold, 1)
    neg = (t <= threshold)
    line_bases = np.roll(pos & neg, -1)

    return line_bases


def vizz_lines(cfg, img, line_bases, c=None):
    img_out = img.copy()
    if c is None:
        c = np.array([0, 255, 0])  # green boxes

    i_bases = np.where(line_bases)[0]
    for i in i_bases:
        img_out[i, :] = c

    return img_out


def dirac(n, idxs, dtype=np.dtype('bool')):
    """Create a one-hot array with idxs set as 1."""
    a = np.zeros(n, dtype=dtype)
    for i in idxs:
        if i >= 0 and i < a.shape[0]:
            a[i] = 1
    return a

