#ifndef NIGHT_SIGHT_SharpenImpl_HPP_
#define NIGHT_SIGHT_SharpenImpl_HPP_

#define MAX_STAGE 10
#define FMax(X, Y) ((X)>(Y)?(X):(Y))
#define FMin(X, Y) ((X)<(Y)?(X):(Y))
#define FClamp(X, L, H) FMax((L), FMin((H), (X)))

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "SharpenImpl.hpp"

using namespace std;
using namespace cv;

class SharpenImpl {
public:
    SharpenImpl();

    ~SharpenImpl();

    int process_cpu(float parameters[], const cv::Mat &input, const cv::Mat &tree_mask,
                const cv::Mat &human_mask, cv::Mat &out_bayer);

private:
    void calc_idx_diff(int w, int kernel, std::vector<int> &indexes);

    cv::Mat weight_map_2(const cv::Mat &bayer, float map_strength, size_t kernel = 7, float eps = 2.0f, float scale = 2.0f, bool blur = true);

    void padding_line_with_ksize2(float *dst, const float *src, size_t dst_w, size_t src_w, bool copy_mid = true);

    cv::Mat padding_bayer_for_bggr_with_ksize2(const cv::Mat &bayer, cv::Mat &padMat);

    cv::Mat repeat_and_padding_with_ksize2(const cv::Mat &mask);

    void mask_red_blue_for_BGGR(const cv::Mat &bayer, cv::Mat &r_hpf, cv::Mat &b_hpf);

    void mask_green_for_BGGR(cv::Mat &g_hpf);

    void CT_descriptor_kxk_sum(const cv::Mat &img, cv::Mat &dst, size_t kernelSize, float eps);

    void sharpen_gauss_7x7(cv::Mat &mat);
};

void print_mat_info(string mat_name, Mat src);
void color_convert_gbrg_bggr(Mat grbg, Mat& bggr);
void visualize_bayer_float(Mat bayer, string save_path,  bool demosaic=true, float gamma=0.5, float gain=1000.f, float r_gain=1.f, float b_gain=1.f);
#endif