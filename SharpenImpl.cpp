#include "SharpenImpl.hpp"

SharpenImpl::SharpenImpl() {}

SharpenImpl::~SharpenImpl() {}

void SharpenImpl::calc_idx_diff(int w, int kernel, std::vector<int> &indexes) {
    int radio = kernel >> 1;
    indexes.resize(kernel * kernel - 1);
    int index = 0;
    for (int y = -radio; y <= radio; y++) {
        int yoff = y * w;
        for (int x = -radio; x <= radio; x++) {
            if (y == 0 && x == 0)
                continue;

            indexes[index++] = yoff + x;
        }
    }
}

void SharpenImpl::CT_descriptor_kxk_sum(const cv::Mat &img, cv::Mat &dst, size_t kernelSize, float eps) {
    CV_Assert(img.type() == CV_32FC1);
    int w = img.cols;
    int h = img.rows;

    if (dst.empty()) {
        dst = cv::Mat::zeros(h, w, CV_8UC1);
    }

    int radio = kernelSize >> 1;
    int xEnd = w - radio;
    int yENd = h - radio;

    std::vector<int> indexes;
    calc_idx_diff(w, kernelSize, indexes);

    for (int y = radio; y < yENd; y++) {
        const float *imgPtr = img.ptr<float>(y);
        uchar *dstPtr = dst.ptr<uchar>(y);

        for (int x = radio; x < xEnd; x++) {
            float curV = imgPtr[x] - eps;
            uchar sum = 0;
            for (auto kIdx: indexes) {
                sum += (imgPtr[kIdx + x] < curV) ? 1 : 0;
            }

            dstPtr[x] = sum;
        }
    }
}

void SharpenImpl::sharpen_gauss_7x7(cv::Mat &mat) {
    //float coefs[] = {0.026267f, 0.100742f, 0.225511f, 0.29496f, 0.225511f, 0.100742f, 0.026267f};
    float gauss_7x7_coefs[] = {
            0.00068996f, 0.00264619f, 0.0059235f, 0.00774771f, 0.0059235f, 0.00264619f, 0.00068996f,
            0.00264619f, 0.01014895f, 0.02271843f, 0.02971486f, 0.02271843f, 0.01014895f, 0.00264619f,
            0.0059235f, 0.02271843f, 0.05085521f, 0.06651672f, 0.05085521f, 0.02271843f, 0.0059235f,
            0.00774771f, 0.02971486f, 0.06651672f, 0.0870014f, 0.06651672f, 0.02971486f, 0.00774771f,
            0.0059235f, 0.02271843f, 0.05085521f, 0.06651672f, 0.05085521f, 0.02271843f, 0.0059235f,
            0.00264619f, 0.01014895f, 0.02271843f, 0.02971486f, 0.02271843f, 0.01014895f, 0.00264619f,
            0.00068996f, 0.00264619f, 0.0059235f, 0.00774771f, 0.0059235f, 0.00264619f, 0.00068996f
    };

    cv::Mat gauss_7x7_kernel(7, 7, CV_32FC1, gauss_7x7_coefs);
    cv::filter2D(mat, mat, -1, gauss_7x7_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

cv::Mat SharpenImpl::weight_map_2(const cv::Mat &bayer, float map_strength, size_t kernel, float eps, float scale, bool blur) {
    size_t h = bayer.rows;
    size_t w = bayer.cols;

    cv::Mat im = cv::Mat::zeros(h / 2, w / 2, bayer.type());
    // convert imgRaw to luma
    {
        // im = 0.25 * imgRaw[0:hh:2, 0:ww:2] + 0.25 * imgRaw[1:hh:2, 0:ww:2] + 
        //  0.25 * imgRaw[0:hh:2, 1:ww:2] + 0.25 * imgRaw[1:hh:2, 1:ww:2]
        for (size_t y = 0; y < h; y += 2) {
            const float *line1 = bayer.ptr<float>(y);
            float *im_ptr = im.ptr<float>(y >> 1);
            for (size_t x = 0; x < w; x += 2) {
                im_ptr[x >> 1] = (line1[x] + line1[x + 1] + line1[x + w] + line1[x + w + 1]) / 4;
            }
        }
    }
//    visualize_bayer_float(im, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/luma.jpg", false);


    // compute census transform (count max value of number 1s and number 0s)
    if (scale > 1.0f) {
        cv::resize(im, im, cv::Size(0, 0), 1.0f / scale, 1.0f / scale);
    }

    cv::Mat census;
    {
        cv::Mat census_ones;
//        float eps = eps / 1000.f;//sx
        CT_descriptor_kxk_sum(im, census_ones, kernel, eps);
//        visualize_bayer_float(census_ones, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/census_ones.jpg", false);

        census = cv::Mat::zeros(census_ones.size(), CV_32FC1);

        size_t total = census_ones.total();
        uchar *census_one_ptr = census_ones.ptr<uchar>();
        float *census_ptr = census.ptr<float>();

        int kernel_pow_2 = kernel * kernel - 1;
        float kernel_pow_2_inv = 1.0f / (float) kernel_pow_2;
        for (size_t i = 0; i < total; i++) {
            int census_zero = kernel_pow_2 - census_one_ptr[i];
            int census_val = kernel_pow_2 - std::max((int) census_one_ptr[i], census_zero);
            census_ptr[i] = (float) census_val * kernel_pow_2_inv;
        }
    }

    if (blur) {
        sharpen_gauss_7x7(census);
    }

    if (scale > 1.0f) {
        cv::resize(census, census, cv::Size(w / 2, h / 2));
    }

    {
        size_t total = census.total();
        float *census_ptr = census.ptr<float>();

        for (size_t i = 0; i < total; i++) {
            float val = census_ptr[i] * map_strength;
            census_ptr[i] = FClamp(val, 0.0f, 1.0f);
        }
    }

    return census;
}

void SharpenImpl::padding_line_with_ksize2(float *dst, const float *src, size_t dst_w, size_t src_w, bool copy_mid) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[dst_w - 1] = src[src_w - 1];
    dst[dst_w - 2] = src[src_w - 2];
    std::memcpy(dst + 2, src, src_w * sizeof(float));
}

cv::Mat SharpenImpl::padding_bayer_for_bggr_with_ksize2(const cv::Mat &bayer, cv::Mat &padMat) {
    CV_Assert(bayer.type() == CV_32FC1);
    size_t hpp = bayer.rows + 4;
    size_t wpp = bayer.cols + 4;

    if (padMat.empty() || padMat.rows != hpp || padMat.cols != wpp || padMat.type() != bayer.type()) {
        padMat = cv::Mat::zeros(hpp, wpp, bayer.type());
    }
    //bayer.copyTo(padMat(cv::Rect(2, 2, bayer.cols, bayer.rows)));
    // 头两行
    {
        float *line1 = padMat.ptr<float>(0);
        float *line2 = padMat.ptr<float>(1);
        const float *bayer_line1 = bayer.ptr<float>(0);
        const float *bayer_line2 = bayer.ptr<float>(1);

        padding_line_with_ksize2(line1, bayer_line1, wpp, bayer.cols);
        padding_line_with_ksize2(line2, bayer_line2, wpp, bayer.cols);
    }


    for (int y = 0; y < bayer.rows; y++) {
        float *line1 = padMat.ptr<float>(y + 2);
        const float *bayer_line1 = bayer.ptr<float>(y);
        padding_line_with_ksize2(line1, bayer_line1, wpp, bayer.cols);
    }

    // 尾两行
    {
        float *line1 = padMat.ptr<float>(hpp - 2);
        float *line2 = padMat.ptr<float>(hpp - 1);
        const float *bayer_line1 = bayer.ptr<float>(bayer.rows - 2);
        const float *bayer_line2 = bayer.ptr<float>(bayer.rows - 1);

        padding_line_with_ksize2(line1, bayer_line1, wpp, bayer.cols);
        padding_line_with_ksize2(line2, bayer_line2, wpp, bayer.cols);
    }

    return padMat;
}

cv::Mat SharpenImpl::repeat_and_padding_with_ksize2(const cv::Mat &mask) {
    CV_Assert(mask.type() == CV_32FC1 || mask.type() == CV_8UC1);
    size_t hpp = mask.rows * 2 + 4;
    size_t wpp = mask.cols * 2 + 4;

    cv::Mat padMat = cv::Mat::zeros(hpp, wpp, mask.type());

    if (mask.type() == CV_32FC1) {
        for (int y = 0; y < mask.rows; y++) {
            float *line1 = padMat.ptr<float>(2 + y * 2) + 2;
            float *line2 = line1 + wpp;
            const float *mask_line1 = mask.ptr<float>(y);

            for (int x = 0; x < mask.cols; x++) {
                *line1++ = mask_line1[x];
                *line1++ = mask_line1[x];
                *line2++ = mask_line1[x];
                *line2++ = mask_line1[x];

                // line1[2*x] = mask_line1[x];
                // line1[2*x+1] = mask_line1[x];
                // line2[2*x] = mask_line1[x];
                // line2[2*x+1] = mask_line1[x];
            }
        }
    } else if (mask.type() == CV_8UC1) {
        for (int y = 0; y < mask.rows; y++) {
            uchar *line1 = padMat.ptr<uchar>(2 + y * 2) + 2;
            uchar *line2 = line1 + wpp;
            const uchar *mask_line1 = mask.ptr<uchar>(y);

            for (int x = 0; x < mask.cols; x++) {
                *line1++ = mask_line1[x];
                *line1++ = mask_line1[x];
                *line2++ = mask_line1[x];
                *line2++ = mask_line1[x];

                // line1[2*x] = mask_line1[x];
                // line1[2*x+1] = mask_line1[x];
                // line2[2*x] = mask_line1[x];
                // line2[2*x+1] = mask_line1[x];
            }
        }
    }

    return padMat;
}

void SharpenImpl::mask_red_blue_for_BGGR(const cv::Mat &bayer, cv::Mat &r_hpf, cv::Mat &b_hpf) {
    int cols = bayer.cols;
    int rows = bayer.rows;
    r_hpf = Mat::zeros(rows, cols, CV_32FC1);
    b_hpf = Mat::zeros(rows, cols, CV_32FC1);
    /*
     * 00 01 02 03 04
     * 10 11 12 13 14
     * 20 21 22 23 24
     * 30 31 32 33 34
     * 40 41 42 43 44
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i % 2 == 0 && j % 2 == 0) {
                r_hpf.at<float>(i, j) = bayer.at<float>(i, j);
            } else if (i % 2 == 1 && j % 2 == 1) {
                b_hpf.at<float>(i, j) = bayer.at<float>(i, j);
            }
        }
    }
}

void SharpenImpl::mask_green_for_BGGR(cv::Mat &g_hpf) {
    int cols = g_hpf.cols;
    int rows = g_hpf.rows;
    /*
     * 00 01 02 03 04
     * 10 11 12 13 14
     * 20 21 22 23 24
     * 30 31 32 33 34
     * 40 41 42 43 44
     */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1))
                g_hpf.at<float>(i, j) = 0;
        }
    }
}


int SharpenImpl::process_cpu(float parameters[], const cv::Mat &input, const cv::Mat &tree_mask,
                                 const cv::Mat &human_mask,
                                 cv::Mat &out_bayer) {
    float halo_suppress = parameters[0];
    float strength_min = parameters[1];
    float strength_max = parameters[2];
    float strength_human = parameters[3];
    float map_strength = parameters[4];
    float white_level = parameters[5];

    Mat bayer = input.clone();

    bayer = bayer * white_level;

    // const size_t kSize = 5;
    const size_t kSizeh = 2; // (kSize//2)

    // need to pad raw image first
    cv::Mat paddedMat;
    padding_bayer_for_bggr_with_ksize2(bayer, paddedMat);
//    visualize_bayer_float(paddedMat, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/paddedMat.jpg");

    // repeat and pad tree_mask 
    cv::Mat pad_tree_mask = repeat_and_padding_with_ksize2(tree_mask);

    // repeat and pad tree_mask 
    cv::Mat pad_human_mask = repeat_and_padding_with_ksize2(human_mask);

    cv::Mat r_hpf, b_hpf, g_hpf;
    {
        float rb_filter_weights[] = {
                -1.0f / 8, 0.0f, -1.0f / 8, 0, -1.0f / 8,
                0, 0, 0, 0, 0,
                -1.0f / 8, 0, 1.0f, 0, -1.0f / 8,
                0, 0, 0, 0, 0,
                -1.0f / 8, 0, -1.0f / 8, 0, -1.0f / 8
        };

        float g_filter_weights[] = {
                -1.0f / 12, 0, -1.0f / 12, 0, -1.0f / 12,
                0, -1.0f / 12, 0, -1.0f / 12, 0,
                -1.0f / 12, 0, 1, 0, -1.0f / 12,
                0, -1.0f / 12, 0, -1.0f / 12, 0,
                -1.0f / 12, 0, -1.0f / 12, 0, -1.0f / 12
        };

        cv::Mat rb_filter_kernel(5, 5, CV_32FC1, (void *) rb_filter_weights);
        cv::Mat g_filter_kernel(5, 5, CV_32FC1, (void *) g_filter_weights);

        cv::Mat rbMat;
        cv::filter2D(paddedMat, rbMat, -1, rb_filter_kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

        mask_red_blue_for_BGGR(rbMat, r_hpf, b_hpf);

        cv::filter2D(paddedMat, g_hpf, -1, g_filter_kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

        mask_green_for_BGGR(g_hpf);

//        visualize_bayer_float(r_hpf * 1000, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/r_hpf.jpg", false);
//        visualize_bayer_float(b_hpf * 1000, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/b_hpf.jpg", false);
//        visualize_bayer_float(g_hpf * 1000, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/g_hpf.jpg", false);
    }

    cv::Mat imgRaw_sharpen = paddedMat.clone();
    {
//         strength_map = tree_mask * strength_min + human_mask * strength_human + (1 - np.clip(tree_mask + human_mask, 0, 1.0)) * strength_max
//         imgRaw_sharpen = imgRawPad + (r_hpf + b_hpf + g_hpf) * strength_map

        size_t hpp = imgRaw_sharpen.rows;
        size_t wpp = imgRaw_sharpen.cols;
        for (size_t y = 0; y < hpp; y++) {
            float *tree_mask_ptr = pad_tree_mask.ptr<float>(y);
            float *human_mask_ptr = pad_human_mask.ptr<float>(y);
            float *r_hpf_ptr = r_hpf.ptr<float>(y);
            float *b_hpf_ptr = b_hpf.ptr<float>(y);
            float *g_hpf_ptr = g_hpf.ptr<float>(y);
            float *sharpen_ptr = imgRaw_sharpen.ptr<float>(y);

            for (size_t x = 0; x < wpp; x++) {
                float tm_val = tree_mask_ptr[x];
                float hm_val = human_mask_ptr[x];
                float thm_val = 1 - (tm_val + hm_val);
                thm_val = FClamp(thm_val, 0.0f, 1.0f);
                float strength_map = tm_val * strength_min + hm_val * strength_human + thm_val * strength_max;
//                cout<<"strength_min:"<<strength_min<<"  strength_human"<<strength_human<<"  strength_max"<<strength_max<<endl;
                sharpen_ptr[x] += (r_hpf_ptr[x] + b_hpf_ptr[x] + g_hpf_ptr[x]) * strength_map;
            }
        }
//        visualize_bayer_float(imgRaw_sharpen, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/imgRaw_sharpen.jpg");
    }

    // todo: save_intermediate_result tree_mask and human_mask
    // halo suppression
    // we do not need cross-channel
    {
        // LOGI("ready to do halo suppression...");
        uchar stencil_rb[] = {
                1, 0, 1, 0, 1,
                0, 0, 0, 0, 0,
                1, 0, 1, 0, 1,
                0, 0, 0, 0, 0,
                1, 0, 1, 0, 1
        };

        uchar stencil_g[] = {
                1, 0, 1, 0, 1,
                0, 1, 0, 1, 0,
                1, 0, 1, 0, 1,
                0, 1, 0, 1, 0,
                1, 0, 1, 0, 1
        };

        cv::Mat stencil_rb_kernel(5, 5, CV_8UC1, (void *) stencil_rb);
        cv::Mat stencil_g_kernel(5, 5, CV_8UC1, (void *) stencil_g);

        // max_img = ndimage.maximum_filter(imgRawPad, footprint=stencil_rb)*(mask_red + mask_blue) + 
        //             ndimage.maximum_filter(imgRawPad, footprint=stencil_g)*mask_green
        // min_img = ndimage.minimum_filter(imgRawPad, footprint=stencil_rb)*(mask_red + mask_blue) + 
        //         ndimage.minimum_filter(imgRawPad, footprint=stencil_g)*mask_green
        // upper_limit = max_img + halo_suppress * np.maximum(imgRaw_sharpen - max_img, 0.0)
        // lower_limit = min_img - halo_suppress * np.maximum(min_img - imgRaw_sharpen, 0.0)
        // imgRaw_sharpen = np.clip(imgRaw_sharpen, a_min=lower_limit, a_max=upper_limit)
        cv::Mat max_img_rb, max_img_g;
        cv::dilate(paddedMat, max_img_rb, stencil_rb_kernel);
        cv::dilate(paddedMat, max_img_g, stencil_g_kernel);

        cv::Mat min_img_rb, min_img_g;
        cv::erode(paddedMat, min_img_rb, stencil_rb_kernel);
        cv::erode(paddedMat, min_img_g, stencil_g_kernel);

        size_t hpp = imgRaw_sharpen.rows;
        size_t wpp = imgRaw_sharpen.cols;

        for (size_t y = 0; y < hpp; y += 2) {
            float *max_img_rb_ptr = max_img_rb.ptr<float>(y);
            float *max_img_rb_ptr2 = max_img_rb_ptr + wpp;

            float *max_img_g_ptr = max_img_g.ptr<float>(y);
            float *max_img_g_ptr2 = max_img_g_ptr + wpp;

            float *min_img_rb_ptr = min_img_rb.ptr<float>(y);
            float *min_img_rb_ptr2 = min_img_rb_ptr + wpp;

            float *min_img_g_ptr = min_img_g.ptr<float>(y);
            float *min_img_g_ptr2 = min_img_g_ptr + wpp;

            float *sharpen_ptr = imgRaw_sharpen.ptr<float>(y);
            float *sharpen_ptr2 = sharpen_ptr + wpp;

            for (size_t x = 0; x < wpp; x += 2) {
                float val_00 = sharpen_ptr[x];
                float val_01 = sharpen_ptr[x + 1];
                float val_10 = sharpen_ptr2[x];
                float val_11 = sharpen_ptr2[x + 1];

                float max_val_00 = halo_suppress * std::max(val_00 - max_img_rb_ptr[x], 0.0f);
                float max_val_01 = halo_suppress * std::max(val_01 - max_img_g_ptr[x + 1], 0.0f);
                float max_val_10 = halo_suppress * std::max(val_10 - max_img_g_ptr2[x], 0.0f);
                float max_val_11 = halo_suppress * std::max(val_11 - max_img_rb_ptr2[x + 1], 0.0f);

                float uplimit_00 = max_img_rb_ptr[x] + max_val_00;
                float uplimit_01 = max_img_g_ptr[x + 1] + max_val_01;
                float uplimit_10 = max_img_g_ptr2[x] + max_val_10;
                float uplimit_11 = max_img_rb_ptr2[x + 1] + max_val_11;

                float min_val_00 = halo_suppress * std::max(min_img_rb_ptr[x] - val_00, 0.0f);
                float min_val_01 = halo_suppress * std::max(min_img_g_ptr[x + 1] - val_01, 0.0f);

                float min_val_10 = halo_suppress * std::max(min_img_g_ptr2[x] - val_10, 0.0f);
                float min_val_11 = halo_suppress * std::max(min_img_rb_ptr2[x + 1] - val_11, 0.0f);

                float lowlimit_00 = min_img_rb_ptr[x] - min_val_00;
                float lowlimit_01 = min_img_g_ptr[x + 1] - min_val_01;
                float lowlimit_10 = min_img_g_ptr2[x] - min_val_10;
                float lowlimit_11 = min_img_rb_ptr2[x + 1] - min_val_11;

                sharpen_ptr[x] = FClamp(val_00, lowlimit_00, uplimit_00);
                sharpen_ptr[x + 1] = FClamp(val_01, lowlimit_01, uplimit_01);

                sharpen_ptr2[x] = FClamp(val_10, lowlimit_10, uplimit_10);
                sharpen_ptr2[x + 1] = FClamp(val_11, lowlimit_11, uplimit_11);
            }
        }
//        visualize_bayer_float(imgRaw_sharpen, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/imgRaw_sharpen_halo.jpg");
    }

    // compute weight map, need FClamp, weightMap only 1/2 sizeof paddedMat
    {
        cv::Mat weightMap = weight_map_2(paddedMat, map_strength, 7, 8.0f, 1.0f, false);
        // imgRaw_sharpen_blend[0:hhp:2, 0:wwp:2] = weight * imgRaw_sharpen[0:hhp:2, 0:wwp:2] + (1 - weight) * imgRawPad[0:hhp:2, 0:wwp:2]
        // imgRaw_sharpen_blend[1:hhp:2, 0:wwp:2] = weight * imgRaw_sharpen[1:hhp:2, 0:wwp:2] + (1 - weight) * imgRawPad[1:hhp:2, 0:wwp:2]
        // imgRaw_sharpen_blend[0:hhp:2, 1:wwp:2] = weight * imgRaw_sharpen[0:hhp:2, 1:wwp:2] + (1 - weight) * imgRawPad[0:hhp:2, 1:wwp:2]
        // imgRaw_sharpen_blend[1:hhp:2, 1:wwp:2] = weight * imgRaw_sharpen[1:hhp:2, 1:wwp:2] + (1 - weight) * imgRawPad[1:hhp:2, 1:wwp:2]
//        visualize_bayer_float(weightMap * 1000.f, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/weightMap.jpg", false);

        size_t hpp = imgRaw_sharpen.rows;
        size_t wpp = imgRaw_sharpen.cols;
        for (size_t y = 0; y < hpp; y += 2) {
            float *sharpen_ptr = imgRaw_sharpen.ptr<float>(y);
            float *sharpen_ptr2 = sharpen_ptr + wpp;

            float *pad_ptr = paddedMat.ptr<float>(y);
            float *pad_ptr2 = pad_ptr + wpp;

            float *weight_ptr = weightMap.ptr<float>(y / 2);

            for (size_t x = 0; x < wpp; x += 2) {
                float weight = weight_ptr[x >> 1];
                sharpen_ptr[x] = weight * sharpen_ptr[x] + (1 - weight) * pad_ptr[x];
                sharpen_ptr[x + 1] = weight * sharpen_ptr[x + 1] + (1 - weight) * pad_ptr[x + 1];
                sharpen_ptr2[x] = weight * sharpen_ptr2[x] + (1 - weight) * pad_ptr2[x];
                sharpen_ptr2[x + 1] = weight * sharpen_ptr2[x + 1] + (1 - weight) * pad_ptr2[x + 1];
            }
        }

//        visualize_bayer_float(imgRaw_sharpen, "/home/SENSETIME/sunxin/2_myrepo/mySharpen/imgRaw_sharpen_weight.jpg");
    }

    // todo: save_intermediate_result imgRaw_sharpen_blend
    imgRaw_sharpen(cv::Rect(kSizeh, kSizeh, bayer.cols, bayer.rows)).copyTo(out_bayer);

    out_bayer = out_bayer / white_level;
    out_bayer = min(max(out_bayer, 0.0f), 1.f);

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// int np_round(float x) {
//     int y = std::floor(x);
//     float c = x - y;

//     if (c > 0.5f) {
//         return y + 1;
//     } else if (c < 0.5f || (y & 1) == 0) {
//         return y;
//     } else {
//         return y + 1;
//     }
// }

void print_mat_info(string mat_name, Mat src) {
    double min = 0.0;
    double max = 0.0;
    minMaxLoc(src, &min, &max);
    Mat mat_mean, mat_stddev;
    meanStdDev(src, mat_mean, mat_stddev);
    double mean, std;
    mean = mat_mean.at<double>(0, 0);
    std = mat_stddev.at<double>(0, 0);

    printf("mat: %s, H: %d, W: %d, Min: %f, Max: %f, Mean: %f, Std: %f, Type: %d\n", mat_name.c_str(), src.rows,
           src.cols, min, max, mean, std, src.type());
}

void color_convert_gbrg_bggr(Mat grbg, Mat &bggr) {
    /*
    grgr
    bgbg
    grgr
    bgbg
    */
    int width = grbg.cols;
    int heigth = grbg.rows;
    copyMakeBorder(grbg, bggr, 0, 1, 0, 0, BORDER_REFLECT);
    bggr = bggr(Rect(0, 1, width, heigth));
}

Mat &MyGammaCorrection(Mat &src, float fGamma) {
    CV_Assert(src.data);  //若括号中的表达式为false，则返回一个错误的信息。

    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));
    // build look up table
    unsigned char lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = pow((float) (i / 255.0), fGamma) * 255.0;
    }
    //先归一化，i/255,然后进行预补偿(i/255)^fGamma,最后进行反归一化(i/255)^fGamma*255

    const int channels = src.channels();
    switch (channels) {
        case 1: {
            //运用迭代器访问矩阵元素
            MatIterator_<uchar> it, end;
            for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++)
                //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
                *it = lut[(*it)];

            break;
        }
        case 3: {

            MatIterator_<Vec3b> it, end;
            for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; it++) {
                //(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;
                //(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;
                //(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }

            break;

        }
    }

    return src;
}

void
split_bayer_bggr_channel_float(const cv::Mat &rawSrc, cv::Mat &rawG1, cv::Mat &rawG2, cv::Mat &rawB, cv::Mat &rawR) {
    int g_row = rawSrc.rows >> 1;
    int g_col = rawSrc.cols >> 1;
    for (int i = 0; i < g_row; i++) {
        const float *ptr_src0 = rawSrc.ptr<float>(i << 1);
        const float *ptr_src1 = rawSrc.ptr<float>((i << 1) + 1);
        float *ptr_B = rawB.ptr<float>(i);
        float *ptr_G1 = rawG1.ptr<float>(i);
        float *ptr_G2 = rawG2.ptr<float>(i);
        float *ptr_R = rawR.ptr<float>(i);

        for (int j = 0; j < g_col; j++) {
            int src_off = j << 1;
            ptr_B[j] = ptr_src0[src_off];
            ptr_G1[j] = ptr_src0[src_off + 1];

            ptr_G2[j] = ptr_src1[src_off];
            ptr_R[j] = ptr_src1[src_off + 1];
        }
    }
}

void
merge_rgb_2_bayer_bggr_float(cv::Mat &srcR, cv::Mat &srcB, cv::Mat &srcG1, cv::Mat &srcG2, cv::Mat &bayer, float black,
                             float white) {
    int row = srcB.rows;
    int col = srcB.cols;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int row_odd = 2 * i + 1;
            int row_even = 2 * i;
            int col_odd = 2 * j + 1;
            int col_even = 2 * j;

            float b_value = srcB.at<float>(i, j);
            float g1_value = srcG1.at<float>(i, j);
            float g2_value = srcG2.at<float>(i, j);
            float r_value = srcR.at<float>(i, j);

            bayer.at<float>(row_even, col_even) = b_value > white ? white : ((b_value < black) ? black : b_value);
            bayer.at<float>(row_even, col_odd) = g1_value > white ? white : ((g1_value < black) ? black : g1_value);
            bayer.at<float>(row_odd, col_even) = g2_value > white ? white : ((g2_value < black) ? black : g2_value);
            bayer.at<float>(row_odd, col_odd) = r_value > white ? white : ((r_value < black) ? black : r_value);
        }
    }
}

void visualize_bayer_float(Mat bayer, string save_path, bool demosaic, float gamma, float gain, float r_gain, float b_gain) {
    Mat input = bayer.clone();
    Mat bayerDe = bayer.clone();

    if (demosaic) {
        // awb
        int bayerRow = bayer.rows / 2;
        int bayerCol = bayer.cols / 2;
        Mat bayerB = cv::Mat(bayerRow, bayerCol, CV_32FC1);
        Mat bayerR = cv::Mat(bayerRow, bayerCol, CV_32FC1);
        Mat bayerGr = cv::Mat(bayerRow, bayerCol, CV_32FC1);
        Mat bayerGb = cv::Mat(bayerRow, bayerCol, CV_32FC1);
        split_bayer_bggr_channel_float(bayer, bayerGr, bayerGb, bayerB, bayerR);
        bayerGb = bayerGb * b_gain;
        bayerGr = bayerGr * r_gain;
        merge_rgb_2_bayer_bggr_float(bayerR, bayerB, bayerGr, bayerGb, bayerDe, 0.0f, 1.0f);

        // gamma
        bayerDe = MyGammaCorrection(bayerDe, gamma);


        // clip
        bayerDe = max(min(bayerDe, 1.f), 0.f);
    }

    bayerDe = bayerDe * gain;
    bayerDe.convertTo(bayerDe, CV_16UC1);

    if (demosaic)
        cvtColor(bayerDe, bayerDe, COLOR_BayerBG2BGR);

    cv::imwrite(save_path, bayerDe);
    print_mat_info(save_path, bayerDe);
}
