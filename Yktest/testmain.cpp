#include "global/YkLog.h"
#include <driver_types.h>

// 各模块测试函数声明
int  main_fdk();
int  main_bp_runner();
int  main_bp_verify();
//int main_siddon_bp_runner();
int main_siddon_ray_adjoint_verify();
int main_siddon_voxel_adjoint_verify();
int main_joseph_adjoint_verify();
int main_siddon_bp_verify();
int main_siddon_zslab_verify();
void test_fp_runner(cudaStream_t stream);
void test_fp_runner_siddon_vs_joseph(cudaStream_t stream);
int  main_fp();
int main_fdk_zslab();
int main_fdk_zslab_bigdata();
int main_bp_zslab_verify();
//int main_helical_verify();

int main_helical_from_volume();
int main_helical_online_from_volume();


void test_fp_runner_fixed_offset(cudaStream_t stream);

void test_fp_runner_random_offset(cudaStream_t stream);
void test_stdrecon_with_random_offset(cudaStream_t stream);
void test_recon_with_random_offset(cudaStream_t stream);

void test_fp_runner_periodic_offset(cudaStream_t stream);
void test_stdrecon_with_periodic_offset(cudaStream_t stream);
void test_recon_with_periodic_offset(cudaStream_t stream);

void test_periodic_fp_ideal_recon(cudaStream_t stream);
void test_periodic_fp_corrected_recon(cudaStream_t stream);
void test_random_fp_ideal_recon(cudaStream_t stream);
void test_fixed_fp_ideal_recon(cudaStream_t stream);
void test_fixed_fp_corrected_recon(cudaStream_t stream);
void test_insufficient_angle_fp_recon(cudaStream_t stream);

void test_flat_detector_roty_fp(cudaStream_t stream);
void test_flat_detector_roty_fp_ellipse(cudaStream_t stream);



void test_generate_pcb_phantom();


int main_ossart_test();
int main_iter_sirt_recon_sim();

int main()
{
    YK::Logger::instance().set_level(YK::LogLevel::Debug);
    YK::Logger::instance().add_file_sink("log.txt");

    // 按需开启/注释
    //main_fdk();
    //main_bp_zslab_verify();
    //main_bp_runner();
    //main_bp_verify();
    //main_siddon_bp_runner();
    //main_siddon_ray_adjoint_verify();
    //main_joseph_adjoint_verify();
    //main_siddon_voxel_adjoint_verify();
    //main_siddon_bp_verify();
    //main_siddon_zslab_verify();
    //test_fp_runner(0);
    test_fp_runner_siddon_vs_joseph(0);
    //main_fp();
    //main_fdk_zslab();
    //main_helical_verify();
    //main_fdk_zslab_bigdata();
    //main_helical_from_volume();
    //main_helical_online_from_volume();
    //test_fp_runner_random_offset(0);
    //test_fp_runner_fixed_offset(0);
    //test_recon_with_offset(0);
    //test_recon_with_random_offset(0);

    //test_fp_runner_periodic_offset(0);
    //test_stdrecon_with_periodic_offset(0);
    //test_recon_with_periodic_offset(0);


    //test_periodic_fp_ideal_recon(0);
    //test_periodic_fp_corrected_recon(0);

    //test_fixed_fp_ideal_recon(0);
    //test_fixed_fp_corrected_recon(0);


    //test_random_fp_ideal_recon(0);


    //test_insufficient_angle_fp_recon(0);

    //test_flat_detector_roty_fp(0);

    //test_generate_pcb_phantom();
    //test_flat_detector_roty_fp_ellipse(0);

    //main_ossart_test();
    //main_iter_sirt_recon_sim();
    return 0;
}
