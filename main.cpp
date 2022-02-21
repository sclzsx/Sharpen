#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#include "SharpenImpl.hpp"


cv::Mat read_raw_u16(std::string path, int h, int w) {
    cv::Mat im(h, w, CV_16UC1);
    FILE *fp = fopen(path.c_str(), "rb");
    fread(im.data, w * h * sizeof(int), 1, fp);
    fclose(fp);
    return im;
}


int main() {
    std::string data_dir = "/home/SENSETIME/sunxin/1_sdk_data/0128_transsion_RGBW_0000/";
    std::string save_dir = "/home/SENSETIME/sunxin/2_myrepo/mySharpen/";

    std::string bayer_path = "0128_transsion_RGBW_0000_out_msdctlite_sdk_gainmap_mu100.raw";
    std::string tree_mask_path = "[NightSight]_tree_mask_to_msdct.png";
    std::string human_mask_path = "[NightSight]_human_mask.png";

    bayer_path = data_dir + bayer_path;
    tree_mask_path = data_dir + tree_mask_path;
    human_mask_path = data_dir + human_mask_path;


    int H = 3472, W = 4640;
    int maskH = H / 2, maskW = W / 2;

    cv::Mat bayer = read_raw_u16(bayer_path, H, W);

    color_convert_gbrg_bggr(bayer, bayer);

    int bl = 256, wl = 4095;
    bayer.convertTo(bayer, CV_32FC1);
    bayer = (bayer - bl) / (wl - bl);
    visualize_bayer_float(bayer, save_dir + "/in_bayer_vis.jpg");

    cv::Mat tree_mask = cv::imread(tree_mask_path, 0);
    tree_mask.convertTo(tree_mask, CV_32FC1);
    tree_mask = tree_mask / 255.f;
    print_mat_info("tree_mask", tree_mask);
    cv::imwrite(save_dir + "/tree_mask.jpg", tree_mask * 255);

    cv::Mat human_mask = cv::imread(human_mask_path, 0);
    human_mask.convertTo(human_mask, CV_32FC1);
    human_mask = human_mask / 255.f;
    print_mat_info("human_mask", human_mask);
    cv::imwrite(save_dir + "/human_mask.jpg", human_mask * 255);

    SharpenImpl sharpener;

    clock_t start, end;
    start = clock();

    float parameters[] = {0.5f, 0.5f, 2.5f, 1.f, 2.5f, 1000.f};//halo, smin, smax, shum, mstr, wl

    cv::Mat out_bayer;

    sharpener.process_cpu(parameters, bayer, tree_mask, human_mask, out_bayer);

    visualize_bayer_float(out_bayer, save_dir + "/out_bayer_vis.jpg");

    end = clock();
    double endtime = (double) (end - start) / CLOCKS_PER_SEC;
    std::cout << "Total Time Cost (s): " << endtime << std::endl;


    return 0;
}

