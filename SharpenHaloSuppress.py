import numpy as np
import cv2
import sys
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import ctypes



sys.path.append('./guide_ltm')
from simple_ISP import simpleCFA, visualizeRaw, simpleISP, loadLUT
from file_util import  img_save_raw


def gauss_7x7(img):
    h = np.array([0.026267, 0.100742, 0.225511, 0.29496, 0.225511, 0.100742, 0.026267], dtype=np.float32)
    filter = np.outer(h, h)

    R = cv2.filter2D(np.float32(img), -1, filter, anchor=(0, 0), borderType = cv2.BORDER_REPLICATE)

    return R

def CT_descriptor_kxk_sr_with_noisemap(im, stddev_map, k=7, scale=3):
    m, n = im.shape
    ct_ones = np.zeros((m, n, k, k))
    ct_zeros = np.zeros((m, n, k, k))
    hk = int((k - 1) // 2)
    for i in range(-hk, hk + 1):
        for j in range(-hk, hk + 1):
            index_i = i + hk
            index_j = j + hk
            ct_ones[hk:m - hk, hk:n - hk, index_i, index_j] = (
                    im[(hk + i):(m - hk + i), (hk + j):(n - hk + j)] < (im[hk:m - hk, hk:n - hk]) - scale * (
                stddev_map[hk:m - hk, hk:n - hk]))
            ct_zeros[hk:m - hk, hk:n - hk, index_i, index_j] = (
                    im[(hk + i):(m - hk + i), (hk + j):(n - hk + j)] > (im[hk:m - hk, hk:n - hk]) + scale * (
                stddev_map[hk:m - hk, hk:n - hk]))
    return ct_ones, ct_zeros


def weight_map_sr_with_noisemap(imgRaw, varmap, k=7, scale=3, blur=True):
    # convert imgRaw to luma
    hh, ww = imgRaw.shape
    im = 0.25 * imgRaw[0:hh:2, 0:ww:2] + 0.25 * imgRaw[1:hh:2, 0:ww:2] + \
         0.25 * imgRaw[0:hh:2, 1:ww:2] + 0.25 * imgRaw[1:hh:2, 1:ww:2]

    transformed_varmap = 0.25 * 0.25 * varmap[0:hh:2, 0:ww:2] + 0.25 * 0.25 * varmap[1:hh:2, 0:ww:2] + \
                         0.25 * 0.25 * varmap[0:hh:2, 1:ww:2] + 0.25 * 0.25 * varmap[1:hh:2, 1:ww:2]
    stddev_map = np.sqrt(transformed_varmap)

    # compute census transform (count max value of number 1s and number 0s)
    h, w = im.shape
    ct_ones, ct_zeros = CT_descriptor_kxk_sr_with_noisemap(im, stddev_map, k=k, scale=scale)
    census_ones = np.sum(np.sum(ct_ones, axis=-1), axis=-1)
    census_zeros = np.sum(np.sum(ct_zeros, axis=-1), axis=-1)

    census = np.maximum(census_ones, census_zeros) / (k * k - 1)

    if blur:
        census = gauss_7x7(census)

    census = census / census.max()

    return census

def CT_descriptor_kxk_sr(im, k=7, eps=2):
    m, n = im.shape
    ct_ones = np.zeros((m, n, k, k))
    ct_zeros = np.zeros((m, n, k, k))
    hk = int((k-1)//2)
    for i in range(-hk, hk+1):
        for j in range(-hk, hk+1):
            index_i = i + hk
            index_j = j + hk
            ct_ones[hk:m - hk, hk:n - hk, index_i, index_j] = (im[(hk+i):(m-hk+i), (hk+j):(n-hk+j)] < (im[hk:m - hk, hk:n - hk]) - np.mean(eps[hk:m - hk, hk:n - hk]))
            ct_zeros[hk:m - hk, hk:n - hk, index_i, index_j] = (im[(hk + i):(m - hk + i), (hk + j):(n - hk + j)] > (im[hk:m - hk, hk:n - hk]) + np.mean(eps[hk:m - hk, hk:n - hk]))
    return ct_ones, ct_zeros


def weight_map_sr(im, k=7, eps=2, ds=2.0, blur=True, lsc=None):
    # compute census transform (count max value of number 1s and number 0s)
    h, w = im.shape
    if ds > 1.0:
        hs, ws = int(h // ds), int(w // ds)
        im = cv2.resize(im, (ws, hs), interpolation=cv2.INTER_LINEAR)
        lsc = cv2.resize(lsc, (ws, hs), interpolation=cv2.INTER_LINEAR)

    ct_ones, ct_zeros = CT_descriptor_kxk_sr(im, k, eps*lsc)
    census_ones = np.sum(np.sum(ct_ones, axis=-1), axis=-1)
    census_zeros = np.sum(np.sum(ct_zeros, axis=-1), axis=-1)

    census = np.maximum(census_ones, census_zeros)/(k*k-1)

    if blur:
        census = gauss_7x7(census)

    if ds > 1.0:
        census = cv2.resize(census, (w, h), interpolation=cv2.INTER_LINEAR)

    census = census/census.max()

    return census

def get_mask_exist(hh, ww):
    # assume BGGR pattern
    mask_red = np.zeros((hh, ww), dtype=np.int)
    mask_green = np.zeros((hh, ww), dtype=np.int)
    mask_blue = np.zeros((hh, ww), dtype=np.int)
    mask_red[1:hh:2, 1:ww:2] = 1
    mask_green[0:hh:2, 1:ww:2] = 1
    mask_green[1:hh:2, 0:ww:2] = 1
    mask_blue[0:hh:2, 0:ww:2] = 1
    return mask_red, mask_green, mask_blue

def ED_metric_2(luma, debug_path, bpp):

    hf = np.array([[0, 0, 0],
                   [1, 0, -1],
                   [0, 0, 0]], dtype=np.float32)
    vf = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [0, -1, 0]], dtype=np.float32)
    df1 = np.array([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1]], dtype=np.float32)
    df2 = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [-1, 0, 0]], dtype=np.float32)
    agv = np.absolute(cv2.filter2D(np.float32(luma), -1, vf, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT))
    agh = np.absolute(cv2.filter2D(np.float32(luma), -1, hf, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT))
    agz = np.absolute(cv2.filter2D(np.float32(luma), -1, df1, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT))
    agn = np.absolute(cv2.filter2D(np.float32(luma), -1, df2, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT))

    ed1_metric = np.maximum(np.maximum(np.maximum(agv, agh), agz), agn)
    ed2_metric = np.clip(ed1_metric/np.maximum(luma, 1.0), 0.0, 1.0)

    cv2.imwrite(os.path.join(debug_path, 'ed1_metric.png'),
                visualizeRaw(ed1_metric/ (2**bpp - 1)*255, 8, resize=False, color=False))

    cv2.imwrite(os.path.join(debug_path, 'ed2_metric.png'),
                visualizeRaw(ed2_metric * 255, 8, resize=False, color=False))

    return ed2_metric


def interp_gain(img, X, Y, bpp, debug_path, tag):
    # build LUT?
    input = np.arange(2**bpp, dtype = np.float32)/(2**bpp)
    output = (input < X[0])*Y[0] + ((input >= X[0]) & (input < X[1]))*((Y[1]-Y[0])/(X[1] - X[0])*(input-X[0]) + Y[0]) + \
             ((input >= X[1]) & (input < X[2]))*((Y[2]-Y[1])/(X[2] - X[1])*(input-X[1]) + Y[1]) + \
             ((input >= X[2]) & (input < X[3]))*((Y[3] - Y[2]) / (X[3] - X[2]) * (input - X[2]) + Y[2]) + \
             (input >= X[3])*Y[3]

    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(input, output)
    plt.xlim([0, 1.0])
    plt.ylim([0, 2.0])
    # draw labels
    plt.title('gain')
    plt.xlabel("input")
    plt.ylabel("output")
    plt.savefig('%s/plot_gain_%s.png' % (debug_path, tag))

    gain = output[np.clip(np.round(img), 0, 2**bpp-1).astype(int)]
    return gain

def get_local_stats(imgRawPad, option, stencil_size):
    lib_path ='./sharpening/libst_local_stats_c_float.so'
    lib = ctypes.cdll.LoadLibrary(lib_path)

    imgRawPad_input = np.round(imgRawPad).astype(np.uint16)
    colors = np.zeros_like(imgRawPad_input)
    colors[::2, ::2] = 0
    colors[1::2, ::2] = 1
    colors[::2, 1::2] = 1
    colors[1::2, 1::2] = 2

    # initialize input array
    imgRawPad_input.reshape(-1)
    imgRawPad_input = imgRawPad_input.astype(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    colors.reshape(-1)
    colors = colors.astype(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    height, width = imgRawPad.shape
    max_map = np.zeros((height, width)).reshape(-1)
    min_map = np.zeros((height, width)).reshape(-1)

    # initialize output array
    max_map = max_map.astype(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    min_map = min_map.astype(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    lib.local_stats_process_cpu(imgRawPad_input, colors, max_map, min_map, ctypes.c_int(option),
                            ctypes.c_int(height), ctypes.c_int(width), ctypes.c_int(stencil_size))

    max_map_img = np.ctypeslib.as_array(
        (ctypes.c_ushort * width * height).from_address(ctypes.addressof(max_map.contents)))
    min_map_img = np.ctypeslib.as_array(
        (ctypes.c_ushort * width * height).from_address(ctypes.addressof(min_map.contents)))

    return max_map_img.copy(), min_map_img.copy()


class SharpenImpl:
    def __init__(self, args, config, metadata):

        self.output_folder = os.path.join(args.output_folder, args.input_folder.split('/')[-1], 'sharpenhalo_suppress')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.save_intermediate_result = args.save_intermediate_result
        # targeted number of bits, could be passed
        self.bpp = metadata['bpp']
        self.whitelevel = metadata['whitelevel']
        # pass config parameters
        self.ISOs = config['sharpen']['ISOs']
        self.halo_suppress_pos_noedge = config['sharpen']['halo_suppress_pos_noedge']
        self.halo_suppress_neg_noedge = config['sharpen']['halo_suppress_neg_noedge']
        self.halo_suppress_pos_edge = config['sharpen']['halo_suppress_pos_edge']
        self.halo_suppress_neg_edge = config['sharpen']['halo_suppress_neg_edge']
        self.strength_max = config['sharpen']['strength_max']
        self.strength_min = config['sharpen']['strength_min']
        self.strength_tree = config['sharpen']['strength_tree']
        self.strength_human = config['sharpen']['strength_human']
        self.census_noise_level = config['sharpen']['census_noise_level']
        self.map_strength = config['sharpen']['map_strength']
        self.ISO = metadata['ISO'][metadata['exp_order'].index('high')]
        self.dither_noise_strength = config['sharpen']['dither_noise_strength']
        self.ELL = config['no_low_expo']

    def process_raw(self, imgRawf, mask_for_dns, mask_for_sgn, blend_mask, map_skin, lsc_map):

        if self.ELL:
            return imgRawf.copy()

        # perform an interpolation between different ISOs
        idx_cur = -1
        for idx, iso in enumerate(self.ISOs):
            if self.ISO <= iso:
                idx_cur = idx
                break
        if idx_cur == 0:
            halo_suppress_pos_edge = self.halo_suppress_pos_edge[0]
            halo_suppress_neg_edge = self.halo_suppress_neg_edge[0]
            halo_suppress_pos_noedge = self.halo_suppress_pos_noedge[0]
            halo_suppress_neg_noedge = self.halo_suppress_neg_noedge[0]
            strength_max = self.strength_max[0]
            strength_min = self.strength_min[0]
            strength_human = self.strength_human[0]
            strength_tree = self.strength_tree[0]
            map_strength = self.map_strength[0]
            census_noise_level = self.census_noise_level[0]
            dither_noise_strength = self.dither_noise_strength[0]
        elif idx_cur == -1:
            halo_suppress_pos_edge = self.halo_suppress_pos_edge[-1]
            halo_suppress_neg_edge = self.halo_suppress_neg_edge[-1]
            halo_suppress_pos_noedge = self.halo_suppress_pos_noedge[-1]
            halo_suppress_neg_noedge = self.halo_suppress_neg_noedge[-1]
            strength_max = self.strength_max[-1]
            strength_min = self.strength_min[-1]
            strength_human = self.strength_human[-1]
            strength_tree = self.strength_tree[-1]
            map_strength = self.map_strength[-1]
            census_noise_level = self.census_noise_level[-1]
            dither_noise_strength = self.dither_noise_strength[-1]
        else:
            coeff = (self.ISO - self.ISOs[idx_cur-1])/(self.ISOs[idx_cur] - self.ISOs[idx_cur-1])
            halo_suppress_pos_edge = self.halo_suppress_pos_edge[idx_cur-1] * (1 - coeff) + self.halo_suppress_pos_edge[idx_cur] * coeff
            halo_suppress_neg_edge = self.halo_suppress_neg_edge[idx_cur - 1] * (1 - coeff) + \
                                     self.halo_suppress_neg_edge[idx_cur] * coeff
            halo_suppress_pos_noedge = self.halo_suppress_pos_noedge[idx_cur - 1] * (1 - coeff) + \
                                       self.halo_suppress_pos_noedge[idx_cur] * coeff
            halo_suppress_neg_noedge = self.halo_suppress_neg_noedge[idx_cur - 1] * (1 - coeff) + \
                                       self.halo_suppress_neg_noedge[idx_cur] * coeff
            strength_max = self.strength_max[idx_cur - 1] * (1 - coeff) + self.strength_max[idx_cur] * coeff
            strength_min = self.strength_min[idx_cur - 1] * (1 - coeff) + self.strength_min[idx_cur] * coeff
            strength_human = self.strength_human[idx_cur - 1] * (1 - coeff) + self.strength_human[idx_cur] * coeff
            strength_tree = self.strength_tree[idx_cur - 1] * (1 - coeff) + self.strength_tree[idx_cur] * coeff
            map_strength = self.map_strength[idx_cur - 1] * (1 - coeff) + self.map_strength[idx_cur] * coeff
            census_noise_level = self.census_noise_level[idx_cur - 1] * (1 - coeff) + self.census_noise_level[idx_cur] * coeff
            dither_noise_strength = self.dither_noise_strength[idx_cur - 1] * (1 - coeff) + self.dither_noise_strength[idx_cur] * coeff

        print('strength_tree is %.5f\n'%(strength_tree))

        kSize = 5
        kSizeh = int(kSize//2)
        hh, ww = imgRawf.shape
        hhp = hh + kSizeh*2
        wwp = ww + kSizeh*2
        rb_filter = np.array([[-1/8, 0, -1/8, 0, -1/8],
                              [0,  0,  0,  0,  0],
                              [-1/8, 0, 1, 0, -1/8],
                              [0,  0,  0,  0,  0],
                              [-1/8, 0, -1/8, 0, -1/8]], dtype=np.float32)

        g_filter = np.array([[-1/12, 0, -1/12, 0, -1/12],
                             [0,  -1/12,  0,  -1/12,  0],
                             [-1/12, 0, 1, 0, -1/12],
                             [0,  -1/12,  0,  -1/12,  0],
                             [-1/12, 0, -1/12, 0, -1/12]], dtype=np.float32)

        # generate hf signal
        # need to pad raw image first
        r = imgRawf[1:hh:2, 1:ww:2]
        r = np.pad(r, (kSizeh//2, kSizeh//2), 'symmetric')
        gr = imgRawf[1:hh:2, 0:ww:2]
        gr = np.pad(gr, (kSizeh//2, kSizeh//2), 'symmetric')
        gb = imgRawf[0:hh:2, 1:ww:2]
        gb = np.pad(gb, (kSizeh//2, kSizeh//2), 'symmetric')
        b = imgRawf[0:hh:2, 0:ww:2]
        b = np.pad(b, (kSizeh//2, kSizeh//2), 'symmetric')
        imgRawPad = np.zeros((hhp, wwp))
        imgRawPad[0:hhp:2, 0:wwp:2] = b
        imgRawPad[0:hhp:2, 1:wwp:2] = gb
        imgRawPad[1:hhp:2, 0:wwp:2] = gr
        imgRawPad[1:hhp:2, 1:wwp:2] = r

        # generate mask
        mask_red, mask_green, mask_blue = get_mask_exist(hhp, wwp)

        r_hpf = cv2.filter2D(np.float32(imgRawPad), -1, rb_filter, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)*mask_red
        b_hpf = cv2.filter2D(np.float32(imgRawPad), -1, rb_filter, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)*mask_blue
        g_hpf = cv2.filter2D(np.float32(imgRawPad), -1, g_filter, anchor=(-1, -1), borderType=cv2.BORDER_CONSTANT)*mask_green

        hpf = r_hpf + b_hpf + g_hpf

        # add 3-segment control
        luma = 0.25 * imgRawPad[0:hhp:2, 0:wwp:2] + 0.25 * imgRawPad[1:hhp:2, 0:wwp:2] + \
               0.25 * imgRawPad[0:hhp:2, 1:wwp:2] + 0.25 * imgRawPad[1:hhp:2, 1:wwp:2]
        X_luma = np.array([0.0, 0.05, 0.3, 0.6])
        Y_luma = np.array([0.0, 0.8, 1.5, 1.2])
        luma_gain = interp_gain(luma, X_luma, Y_luma, self.bpp, self.output_folder, 'luma')
        luma_gain = np.repeat(luma_gain, 2, axis=0)
        luma_gain = np.repeat(luma_gain, 2, axis=1)
        hpf = hpf * luma_gain

        if self.save_intermediate_result:
            cv2.imwrite(os.path.join(self.output_folder, 'luma_gain.png'),
                        visualizeRaw(luma_gain * 255, 8, resize=False, color=False))

        blend_mask = cv2.resize(blend_mask, (int(ww//2), int(hh//2)), interpolation=cv2.INTER_LINEAR)
        strength_map = (1-mask_for_dns)*(strength_max - strength_min) + strength_min
        strength_map = strength_map * blend_mask + strength_tree * (1 + np.clip(mask_for_sgn, 0.0, 0.3)*(0.5 - 1.0)/0.3) * (1 - blend_mask)
        strength_map = strength_map * (1 - map_skin) + strength_human * map_skin

        strength_map = np.repeat(strength_map, 2, axis=0)
        strength_map = np.repeat(strength_map, 2, axis=1)
        strength_map = np.pad(strength_map, (kSizeh, kSizeh), 'constant')

        imgRaw_sharpen = imgRawPad + hpf * strength_map

        edge_map2 = ED_metric_2(luma, self.output_folder, self.bpp)
        edge_map2 = np.repeat(edge_map2, 2, axis=0)
        edge_map2 = np.repeat(edge_map2, 2, axis=1)


        if self.save_intermediate_result:
            imgCFA = simpleCFA(imgRawf)
            DRCGain, DRCInvGain, GammaLUT, GammaInvLUT, CCM, invCCM = loadLUT(self.bpp)
            imgY, _, imgRGB = simpleISP(imgCFA, self.bpp, DRCGain, GammaLUT, CCM, False, 0, '')
            cv2.imwrite(os.path.join(self.output_folder, 'input_image.png'),
                        visualizeRaw(imgRGB, 8, resize=False, color=True))

            cv2.imwrite(os.path.join(self.output_folder, 'imgY.png'),
                        visualizeRaw(imgY, 8, resize=False, color=False))

            imgCFA_out = simpleCFA(imgRaw_sharpen[kSizeh:hhp-kSizeh, kSizeh:wwp-kSizeh])
            imgY, _, imgRGB_out = simpleISP(imgCFA_out, self.bpp, DRCGain, GammaLUT, CCM, False, 0, '')
            cv2.imwrite(os.path.join(self.output_folder, 'output_image_sharpen.png'),
                        visualizeRaw(imgRGB_out, 8, resize=False, color=True))

            cv2.imwrite(os.path.join(self.output_folder, 'mask_for_dns.png'),
                        visualizeRaw(mask_for_dns*255, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'mask_for_sgn.png'),
                        visualizeRaw(mask_for_sgn * 255, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'imgY_sharpen.png'),
                        visualizeRaw(imgY, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'map_skin.png'),
                        visualizeRaw(map_skin * 255, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'edge_map2.png'),
                        visualizeRaw(edge_map2 * 255, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'strength_map.png'),
                        visualizeRaw(strength_map/strength_max * 255, 8, resize=False, color=False))


        # halo suppression
        # do we need cross-channel
        cross_channel = False
        if cross_channel:
            stencil = np.ones((5, 5), dtype=np.bool)
            max_img = ndimage.maximum_filter(imgRawPad, footprint=stencil)
            min_img = ndimage.minimum_filter(imgRawPad, footprint=stencil)
        else:
            stencil_rb = np.array([[1, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1]], dtype=np.bool)
            stencil_g = np.array([[1, 0, 1, 0, 1],
                                  [0, 1, 0, 1, 0],
                                  [1, 0, 1, 0, 1],
                                  [0, 1, 0, 1, 0],
                                  [1, 0, 1, 0, 1]], dtype=np.bool)

            max_img = ndimage.maximum_filter(imgRawPad, footprint=stencil_rb)*(mask_red + mask_blue) + \
                      ndimage.maximum_filter(imgRawPad, footprint=stencil_g)*mask_green
            min_img = ndimage.minimum_filter(imgRawPad, footprint=stencil_rb)*(mask_red + mask_blue) + \
                      ndimage.minimum_filter(imgRawPad, footprint=stencil_g)*mask_green

        # the second argument is option
        # option 1: local maximum and minimum
        # option 2: local second maximum and minimum
        # option 3: average of the first 3 and last 3
        # option 4: local mean
        # option 5: local median
        max_img_C, min_img_C = get_local_stats(imgRawPad.astype(np.uint16), 3, 7)
        max_img_C = max_img_C.astype(np.float32)
        min_img_C = min_img_C.astype(np.float32)
        upper_limit = (1 - edge_map2)*(max_img + halo_suppress_pos_noedge * np.maximum(imgRaw_sharpen - max_img, 0.0)) + \
                      edge_map2*(max_img_C + halo_suppress_pos_edge * np.maximum(imgRaw_sharpen - max_img_C, 0.0))

        lower_limit = (1 - edge_map2)*(min_img - halo_suppress_neg_noedge * np.maximum(min_img - imgRaw_sharpen, 0.0)) + \
                      edge_map2*(min_img_C - halo_suppress_neg_edge * np.maximum(min_img_C - imgRaw_sharpen, 0.0))

        imgRaw_sharpen = np.clip(imgRaw_sharpen, a_min=lower_limit, a_max=upper_limit)

        if self.save_intermediate_result:
            imgCFA_out = simpleCFA(imgRaw_sharpen[kSizeh:hhp-kSizeh, kSizeh:wwp-kSizeh])
            imgY, _, imgRGB_out = simpleISP(imgCFA_out, self.bpp, DRCGain, GammaLUT, CCM, False, 0, '')
            cv2.imwrite(os.path.join(self.output_folder, 'output_image_sharpen_halosuppress_2.png'),
                        visualizeRaw(imgRGB_out, 8, resize=False, color=True))

            cv2.imwrite(os.path.join(self.output_folder, 'imgY_sharpen_haloSuppress_2.png'),
                        visualizeRaw(imgY, 8, resize=False, color=False))

        # compute weight map
        lsc = np.amax(lsc_map, axis=2)
        lsc = np.pad(lsc, (kSizeh, kSizeh), 'constant')
        lsc = lsc[::2, ::2]

        map_skin = np.pad(map_skin, (kSizeh//2, kSizeh//2), 'constant')
        weight = weight_map_sr(luma, k=7, eps=census_noise_level, ds=2.0, blur=False, lsc=lsc) * map_strength
        weight = weight + map_skin
        weight = np.clip(weight, 0.0, 1.0)

        imgRaw_sharpen_blend = np.zeros_like(imgRawPad)
        imgRaw_sharpen_blend[0:hhp:2, 0:wwp:2] = weight * imgRaw_sharpen[0:hhp:2, 0:wwp:2] + (1 - weight) * imgRawPad[0:hhp:2, 0:wwp:2]
        imgRaw_sharpen_blend[1:hhp:2, 0:wwp:2] = weight * imgRaw_sharpen[1:hhp:2, 0:wwp:2] + (1 - weight) * imgRawPad[1:hhp:2, 0:wwp:2]
        imgRaw_sharpen_blend[0:hhp:2, 1:wwp:2] = weight * imgRaw_sharpen[0:hhp:2, 1:wwp:2] + (1 - weight) * imgRawPad[0:hhp:2, 1:wwp:2]
        imgRaw_sharpen_blend[1:hhp:2, 1:wwp:2] = weight * imgRaw_sharpen[1:hhp:2, 1:wwp:2] + (1 - weight) * imgRawPad[1:hhp:2, 1:wwp:2]

        if self.save_intermediate_result:
            imgCFA_out = simpleCFA(imgRaw_sharpen_blend[kSizeh:hhp - kSizeh, kSizeh:wwp - kSizeh])
            imgY, _, imgRGB_out = simpleISP(imgCFA_out, self.bpp, DRCGain, GammaLUT, CCM, False, 0, '')
            cv2.imwrite(os.path.join(self.output_folder, 'output_image_sharpen_halosuppress_blend_2.png'),
                        visualizeRaw(imgRGB_out, 8, resize=False, color=True))

            cv2.imwrite(os.path.join(self.output_folder, 'consensus_weight_2.png'),
                        visualizeRaw(weight * 255, 8, resize=False, color=False))

            cv2.imwrite(os.path.join(self.output_folder, 'imgY_sharpen_haloSuppress_blend_2.png'),
                        visualizeRaw(imgY, 8, resize=False, color=False))

        imgRaw_sharpen_blend = imgRaw_sharpen_blend[kSizeh:hhp-kSizeh, kSizeh:wwp-kSizeh]
        imgRaw_sharpen_blend = np.clip(imgRaw_sharpen_blend, 0, self.whitelevel)

        # # add high-frequency noise
        # mean = 0
        # sigma = 2**self.bpp*dither_noise_strength
        # gauss_noise = np.random.normal(mean, sigma, (hh, ww))
        # gauss_noise = gauss_noise.reshape(hh, ww)
        # print('gaussian noise max val is %.5f'%(np.amax(gauss_noise)))
        # imgRaw_sharpen_blend_noise = imgRaw_sharpen_blend + gauss_noise
        # imgRaw_sharpen_blend_noise = np.clip(imgRaw_sharpen_blend_noise, 0, self.whitelevel)
        # if self.save_intermediate_result:
        #     cv2.imwrite(os.path.join(self.output_folder, 'raw_sharpen.png'),
        #                 visualizeRaw(imgRaw_sharpen_blend, self.bpp, resize=False, color=False))
        #     cv2.imwrite(os.path.join(self.output_folder, 'raw_sharpen_noise.png'),
        #                 visualizeRaw(imgRaw_sharpen_blend_noise, self.bpp, resize=False, color=False))

        return imgRaw_sharpen_blend






