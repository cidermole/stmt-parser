import numpy as np
import cv2
from bisect import bisect_left
from collections import defaultdict
import scipy.signal


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


def is_overlap(ax0, ax1, bx0, bx1):
    """Checks if two spans `ax0:ax1` and `bx0:bx1` overlap."""
    assert ax1 > ax0
    assert bx1 > bx0
    return (ax0 <= bx0 < ax1) or (ax0 <= bx1 < ax1) or (bx0 <= ax0 < bx1) or (bx0 <= ax1 < bx1)


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
    min_width = cfg['font_weight']  # minimum letter width, relative to font size (0.0 - 1.0)
    punct_min_size = cfg['font_weight']  # minimum punctuation height (e.g '.'), relative to font size (0.0 - 1.0)
    w_min, h_min = int(min_width * cfg['font_size']), int(min(punct_min_size, cfg['smallcap_min_size']) * cfg['font_size'])

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


def vizz_segmentation(cfg, img, nb_components, output, stats, centroids, c=None):
    img_out = img.copy()
    if c is None:
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


# todo: refactor: def detect_lines()
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
    """:param line_bases: one-hot vector."""
    img_out = img.copy()
    if c is None:
        c = np.array([0, 255, 0])  # green boxes

    i_bases = np.where(line_bases)[0]
    for i in i_bases:
        img_out[i, :] = c

    return img_out


def vizz_lines_2(cfg, img, line_bases, c_base=None, c_bounds=None):
    """:param line_bases: list of `LineBase()` instances."""
    img_out = img.copy()
    if c_base is None:
        c_base = np.array([0, 255, 0])  # green lines for line bases
    if c_bounds is None:
        c_bounds = np.array([0, 0, 255])  # blue lines for bounds

    y_bases = [lb.y for lb in line_bases]
    y_top_mins = [lb.y_top_min for lb in line_bases]
    y_bot_maxs = [lb.y_bot_max for lb in line_bases]

    for y in y_bases:
        img_out[y, :] = c_base
    for y in y_top_mins:
        if y < 0 or y >= img_out.shape[0]: continue
        img_out[y, :] = c_bounds
    for y in y_bot_maxs:
        if y < 0 or y >= img_out.shape[0]: continue
        img_out[y, :] = c_bounds

    return img_out


def dirac(n, idxs, dtype=np.dtype('bool')):
    """Create a one-hot array with idxs set as 1."""
    a = np.zeros(n, dtype=dtype)
    for i in idxs:
        if i >= 0 and i < a.shape[0]:
            a[i] = 1
    return a


class LineBase:
    """Describes the location of a single text line on the page."""
    # this could be extended to carry horizontal range information, too (~ bounding box)
    def __init__(self, y, y_top_min, y_bot_max):
        """
        :param y: line base position (coordinate just below regular small caps like 'a', 'e'.)
        :param y_top_min: line bounding-box top (may be negative)
        :param y_bot_max: line bounding-box bottom (may be larger than page size)
        """
        # note that "line bounding-box" defines a box in which to search for letter bounding-boxes.
        # this is relatively tight, and not all letters are expected to fit pixel-perfect into it.
        self.y = y
        self.y_top_min = y_top_min
        self.y_bot_max = y_bot_max


def detect_line_bases(cfg, dn, nb_components, stats):
    """
    Find text lines on the page. Assumes that lines span the entire page width.
    """

    # find text lines based on letter position stats
    line_letter_freq = line_letter_stats(cfg, dn.shape[0], nb_components, stats)
    line_y_bases = line_detector(cfg, line_letter_freq)  # one-hot vector

    # - compute letter bounds (currently unused)
    # - remove unreasonably tight lines

    line_yy_p = np.array([0] + list(np.where(line_y_bases)[0]) + [dn.shape[0]-1])
    line_yy = []
    line_bases = []
    bounds_yy = []
    for i_line in range(line_yy_p.shape[0]-1):
        y0, y1 = line_yy_p[i_line:i_line+2]
        #y_top_min = y0 + int(cfg['let_bottom_overlap'] * cfg['font_size'])
        y_top_min = y1 - int(cfg['font_size'] * (cfg['line_spacing'] - cfg['sep_bottom_margin']))
        #y_top_min = y0 + int((cfg['line_spacing'] - 1.0) * cfg['font_size'])  # note: very similar, but we need +2 px or sth
        y_bot_max = y1 + int((cfg['line_spacing'] - 1.0) * 1.1 * cfg['font_size'])
        
        # filter away lines with spacing unreasonably small, i.e. (y1-y0) < thr
        if (y1 - y0) < int(cfg['font_size'] * (cfg['line_spacing'] - cfg['sep_bottom_margin'])):
            continue
        
        line_yy.append(y1)
        bounds_yy.append(y_top_min)
        bounds_yy.append(y_bot_max)
        line_bases.append(LineBase(y1, y_top_min, y_bot_max))

    return line_bases


def assign_segments(line_bases, nb_components, stats, centroids):
    """
    Divide segments between individual text lines.
    """

    #
    # assign shapes to lines
    #

    line_stat_idxs = defaultdict(list)
    line_stats = defaultdict(list)

    line_yy = [lb.y for lb in line_bases]

    # in: line_yy, (nb_combonents, stats, centroids)
    # assert line_yy is ascending array of indices of lines
    for i, stat, centroid in zip(range(nb_components), stats, centroids):
        x0, y0, w, h, net_area = stat
        x1, y1 = x0 + w - 1, y0 + h - 1
        # decide which line the centroid belongs to
        i_line = bisect_left(line_yy, centroid[1])
        
        line_stat_idxs[line_yy[i_line]] += [i]
        line_stats[line_yy[i_line]] += [stat]
        
        # nb_components, output, stats, centroids

    return line_stat_idxs, line_stats


def merge_segments(stats):
    """
    Merge segments of a text line into letters. Merges individual segments of vertically divided letters like 'i', ':' into a single bounding box.
    `stats` must be from a single text line, because `y` coordinates are currently ignored.
    """
    ls = stats

    # compute local merge incidence matrix,
    # based on x-coord overlaps
    a_merge = np.zeros((len(ls), len(ls)), dtype=bool)
    for i, a_stat in zip(range(len(ls)), ls):
        ax0, _, aw, _, _ = a_stat
        ax1 = ax0 + aw - 1
        for j, b_stat in zip(range(len(ls)), ls):
            bx0, _, bw, _, _ = b_stat
            bx1 = bx0 + bw - 1
            a_merge[i,j] = is_overlap(ax0, ax1, bx0, bx1)

    # detect and number individual letters (coherent segments)
    components = number_matrix(a_merge)
    
    line_connected_stats = []
    
    # connect segments
    used = np.zeros(len(ls), dtype=bool)
    for i in range(len(ls)):
        if used[i] == 1: continue
        i_comp = np.where(components == components[i])[0]
        if len(i_comp) == 0: continue
    
        x0s, x1s, y0s, y1s, aa = [], [], [], [], []
        for j in i_comp:
            x0, y0, w, h, net_area = ls[j]
            x0s.append(x0)
            x1s.append(x0 + w - 1)
            y0s.append(y0)
            y1s.append(y0 + h - 1)
            aa.append(net_area)
            used[j] = 1

        # compute new bounding rect
        x1, y1 = np.max(x1s), np.max(y1s)
        net_area = np.sum(aa)
        x0, y0 = np.min(x0s), np.min(y0s)
        w, h = (x1 - x0 + 1), (y1 - y0 + 1)
        line_connected_stats.append((x0, y0, w, h, net_area))
    
    return line_connected_stats


def detect_pages(cfg, dn):
    """Detect pages based on dashed lines, with vertical white padding (font size).
    :returns: list of tuples of y-ranges of pages, e.g. [(0,1180),(1180,2400) ...]"""

    # define kernel to detect dashed line:
    # two or three pixels wide vertically.
    # white space around, up to font_size.

    fs = int(cfg['font_size'])
    lwt = 2  # line width tolerance (lw..lw+lwt)
    lw = 2   # line width minimum
    ampl = 0.12  # min detection amplitude (density of dash transitions)
    k3 = np.zeros(2*fs + lwt + lw)
    k3[:] = -1.0/ampl
    k3[(fs+lwt//2):(fs+lwt//2+lw)] = 1.0/ampl  # ... to reach norm 1.0
    k3[fs:(fs+lwt//2)] = 0
    k3[-(fs+lwt//2):-fs] = 0

    # apply kernel to transition counts
    idxs = detect_vertical_sep(cfg, dn, k3)

    # convert into y-ranges
    idxs = [0] + list(idxs) + [dn.shape[0]]
    res = []
    for i in range(len(idxs)-1):
        res.append((idxs[i], idxs[i+1]))

    return res


def detect_table_headers(cfg, dn):
    # define kernel to detect table headers:
    # dotted dense stripe, with height around 'font_size'.
    fs_hdr = int(cfg['font_size'])  # approx. header font size
    ampl_h = 0.1  # min detection amplitude (density of dash transitions, horizontally)
    ampl_v = 0.5  # min detection amplitude (density of dots, vertically)
    k3x = np.ones(fs_hdr) / (ampl_h * ampl_v * fs_hdr)

    # apply kernel to transition counts
    idxs = detect_vertical_sep(cfg, dn, k3x)
    return idxs


def detect_vertical_sep(cfg, dn, k3):
    """Detect vertical separators (pages, table headings) based on line shading.
    :returns: list of y bottom coordinates, e.g. [1180, 2400...]"""

    # dashed line idea: transition count detector
    # 1. clip to b/w
    # 2. convolve [1,-1], [-1,1], and threshold with 2
    # 3. sum to count transitions

    # threshold greyscale into binary image
    th = 0.5
    _, t = cv2.threshold(dn, th, 1.0, cv2.THRESH_BINARY)
    u = 1 - t.astype(np.int8)

    # edge detect
    k1, k2 = np.array([[-1,1]]), np.array([[1,-1]])
    up = u*2 - 1
    t1 = np.clip(scipy.signal.convolve(up, k1, mode='same'), a_min=0, a_max=1).astype('uint8')
    #t2 = np.clip(scipy.signal.convolve(up, k2, mode='same'), a_min=0, a_max=1).astype('uint8')
    t1s = np.sum(t1, axis=1) / t1.shape[1]
    #t2s = np.sum(t2, axis=1) / t2.shape[1]

    # t1 transitions amplitude
    t1tsa = np.clip(scipy.signal.convolve(t1s, k3), a_min=0, a_max=1) > 0.9
    # t1 transition detection (geared towards finding falling edge = line base)
    t1ts = np.clip(scipy.signal.convolve(t1tsa*2 - 1, np.array([1,-1])), a_min=0, a_max=1)
    t1ts[0] = 0 # fix edge glitches
    t1ts[-1] = 0

    return np.where(t1ts)[0]
