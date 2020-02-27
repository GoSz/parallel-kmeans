/*
 *  Parallel Kmeans
 *
 *  A multi-thread kmeans clustering tool based on OpenMP and Eigen
 *
 */


#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace parallel_kmeans
{

typedef float real;

// Use row-major storage, for better cache locality
typedef Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

/*
 *  @brief  init method of centroid
 *
 */
enum CentInitType
{
    RandInit = 1,
    KPPInit  = 2,
};

/*
 *  @brief  distance metric type
 *
 */
enum DistType
{
    EuclideanDist   = 1,
    CosineDist      = 2,
};


/*
 *  @brief  Parallel Kmeans
 *
 */
class ParallelKmeans
{
public:
    /*
     *  @brief  ctor
     *
     */
    ParallelKmeans(uint32_t cent_num, uint32_t dim, DistType dist_type, int seed = 0);

    /*
     *  @brief  dtor
     *
     */
    ~ParallelKmeans();

    /*
     *  @brief  init cent with kmeans++ method
     *  @note   if iter_max == 0, iteration wont stop unless converged
     *          if thread_num == 0, defalut thread_num will be used
     *
     */
    bool cluster(Matrix& samp_matrix,
                 CentInitType init_type,
                 uint32_t iter_max,
                 real converge = 0.01,
                 uint32_t thread_num = 0);

public:
    Matrix* ptr_cent_matrix;            // centroid of clusters
    std::vector<uint32_t> cent_of_samp; // cluster of each sample
    std::vector<real> samp_cent_dist;   // 

private:
    /*
     *  @brief  init cent with kmeans++ method
     *
     */
    void _calc_cent(const Matrix& samp_matrix);

    /*
     *  @brief  init cent randomly
     *
     */
    void _rand_init(const Matrix& samp_matrix);

    /*
     *  @brief  init cent with kmeans++ method
     *
     */
    void _kmeans_pp_init(const Matrix& samp_matrix);

    bool _cent_init(const Matrix& samp_matrix, CentInitType init_type);

    /*
     *  @brief  normalize each row of the matrix
     *
     */
    void _normalize_mat(Matrix& matrix);

private:
    uint32_t _dim;
    uint32_t _cent_num;
    DistType _dist_type;

    uint32_t* _samp_num_of_cent;
};

} // namespace parallel_kmeans
