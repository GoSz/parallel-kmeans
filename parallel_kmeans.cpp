#include <stdio.h>
#include <omp.h>
#include <iostream>
#include "parallel_kmeans.h"


namespace parallel_kmeans
{

ParallelKmeans::ParallelKmeans(uint32_t cent_num, uint32_t dim, DistType dist_type, int seed):
    _cent_num(cent_num), _dim(dim), _dist_type(dist_type),
    ptr_cent_matrix(NULL), _samp_num_of_cent(NULL)
{
    srand(seed);
    _samp_num_of_cent = new uint32_t[_cent_num];
}

ParallelKmeans::~ParallelKmeans()
{
    delete ptr_cent_matrix;
    delete _samp_num_of_cent;
}

void ParallelKmeans::_normalize_mat(Matrix& matrix)
{
    uint32_t row_num = matrix.rows();
    for (uint32_t row_i = 0; row_i < row_num; row_i++)
    {
        real norm = matrix.row(row_i).norm();
        if (norm > 0)
        {
            matrix.row(row_i) /= norm;
        }
    }
}

void ParallelKmeans::_calc_cent(const Matrix& samp_matrix)
{
    // sum all samp of a cluster
    ptr_cent_matrix->setZero();
    memset(_samp_num_of_cent, 0, _cent_num*sizeof(uint32_t));
    uint32_t samp_num = samp_matrix.rows();
    for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
    {
        ptr_cent_matrix->row(cent_of_samp[samp_i]) += samp_matrix.row(samp_i);
        _samp_num_of_cent[cent_of_samp[samp_i]]++;
    }

    // then get new cent
    #pragma omp parallel for schedule(dynamic)
    for (uint32_t cent_i = 0; cent_i < _cent_num; cent_i++)
    {
        if (_samp_num_of_cent[cent_i] > 0)
        {
            if (_dist_type == DistType::CosineDist)
            {
                real norm = ptr_cent_matrix->row(cent_i).norm();
                if (norm > 0)
                {
                    ptr_cent_matrix->row(cent_i) /= norm;
                }

            }
            else
            {
                ptr_cent_matrix->row(cent_i) /= _samp_num_of_cent[cent_i];
            }
        }
    }
}

void ParallelKmeans::_rand_init(const Matrix& samp_matrix)
{
    // randomly assign samples as centroid
    uint32_t samp_num = samp_matrix.rows();
    for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
    {
        cent_of_samp[samp_i] = rand() % _cent_num;
    }
    _calc_cent(samp_matrix);
    std::cerr << "rand init finished" << std::endl;
}

void ParallelKmeans::_kmeans_pp_init(const Matrix& samp_matrix)
{
    uint32_t samp_num = samp_matrix.rows();
    double sum;
    double *d = new double[samp_num];

    // randomly init the first centroid
    ptr_cent_matrix->row(0) = samp_matrix.row(rand() % samp_num);
    for(uint32_t cent_i = 1; cent_i < _cent_num; cent_i++)
    {
        sum = 0;

        // compare each sample with the topN initialized centroids
        if (_dist_type == DistType::EuclideanDist)
        {
            #pragma omp parallel for reduction(+: sum) schedule(dynamic)
            for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
            {
                Eigen::VectorXf tmp_v = (ptr_cent_matrix->topRows(cent_i).rowwise() - samp_matrix.row(samp_i)).rowwise().norm();
                d[samp_i] = tmp_v.maxCoeff();
                sum += d[samp_i];
            }
        }
        else if (_dist_type == DistType::CosineDist)
        {
            #pragma omp parallel for reduction(+: sum) schedule(dynamic)
            for(uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
            {
                Eigen::VectorXf tmp_v = ptr_cent_matrix->topRows(cent_i) * samp_matrix.row(samp_i).transpose();
                d[samp_i] = 1 - tmp_v.maxCoeff();
                sum += d[samp_i];
            }
        }

        // then selecting a sample with different probability according to their distance
        // to init the next centroid
        sum = sum * rand() / (RAND_MAX - 1.0);
        for(uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
        {
            sum -= d[samp_i];
            if (sum  > 0)
            {
                continue;
            }
            ptr_cent_matrix->row(cent_i) = samp_matrix.row(samp_i);
            break;
        }

        std::cerr << "\r" << "kmeans++ init: " << int( cent_i/float(_cent_num) * 100) << "%" << std::flush;
    }
    std::cerr << "\r" << "kmeans++ init: 100\%, init finished" << std::endl;
    delete [] d;

    // finally calc cluster of each sample
    if (_dist_type == DistType::EuclideanDist)
    {
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
        {
            Eigen::VectorXf tmp_v = (ptr_cent_matrix->rowwise() - samp_matrix.row(samp_i)).rowwise().norm();
            Eigen::VectorXf::Index min_index;
            tmp_v.minCoeff(&min_index);
            cent_of_samp[samp_i] = min_index;
        }
    }
    else if (_dist_type == DistType::CosineDist)
    {
        #pragma omp parallel for schedule(dynamic)
        for(uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
        {
            Eigen::VectorXf tmp_v = (*ptr_cent_matrix) * samp_matrix.row(samp_i).transpose();
            Eigen::VectorXf::Index max_index;
            tmp_v.maxCoeff(&max_index);
            cent_of_samp[samp_i] = max_index;
        }
    }
}

bool ParallelKmeans::_cent_init(const Matrix& samp_matrix, CentInitType init_type)
{
    switch (init_type)
    {
        case CentInitType::RandInit:
            _rand_init(samp_matrix);
            break;
        case CentInitType::KPPInit:
            _kmeans_pp_init(samp_matrix);
            break;
        default:
            return false;
    }
    return true;
}

bool ParallelKmeans::cluster(Matrix& samp_matrix,
                     CentInitType init_type,
                     uint32_t iter_max,
                     real converge,
                     uint32_t thread_num)
{
    if (thread_num != 0)
    {
        omp_set_num_threads(thread_num);
    }

    // for Eigen thread safety
    Eigen::initParallel();

    uint32_t samp_num = samp_matrix.rows();
    cent_of_samp.clear();
    samp_cent_dist.clear();
    cent_of_samp.resize(samp_num);
    samp_cent_dist.resize(samp_num);

    ptr_cent_matrix = new Matrix(_cent_num, _dim);
    Matrix pre_cent_matrix(_cent_num, _dim);    // centroids of last iteration

    // normalize if using cosine dist
    if (_dist_type == DistType::CosineDist)
    {
        _normalize_mat(samp_matrix);
    }

    ptr_cent_matrix->setZero();
    pre_cent_matrix.setZero();

    // init cent first
    if (!_cent_init(samp_matrix, init_type))
    {
        return false;
    }

    // for statistics
    real cent_loss = 0.;    // cent diff
    real avg_dist  = 0.;

    uint32_t iter = 1;
    while (iter <= iter_max || iter_max == 0)
    {
        // update cent
        _calc_cent(samp_matrix);

        cent_loss = (*ptr_cent_matrix - pre_cent_matrix).cwiseAbs().sum();
        printf("iter=%d, cent_loss=%f, avg_dist=%f\n", iter, cent_loss, avg_dist);
        if (cent_loss < converge)
        {
            std::cout << "cent_loss converged." << std::endl;
            break;
        }
    
        // update cent of each samp
        avg_dist = 0;
        if (_dist_type == EuclideanDist)
        {
            #pragma omp parallel for reduction(+: avg_dist) schedule(dynamic)
            for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
            {
                Eigen::VectorXf tmp_v = (ptr_cent_matrix->rowwise() - samp_matrix.row(samp_i)).rowwise().norm();
                Eigen::VectorXf::Index min_index;
                tmp_v.minCoeff(&min_index);

                cent_of_samp[samp_i] = min_index;
                samp_cent_dist[samp_i] = tmp_v(min_index);
                avg_dist += tmp_v(min_index);
            }
        }
        else if (_dist_type == CosineDist)
        {
            #pragma omp parallel for reduction(+: avg_dist) schedule(dynamic)
            for(uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
            {
                Eigen::VectorXf tmp_v(_cent_num);
                tmp_v = (*ptr_cent_matrix) * samp_matrix.row(samp_i).transpose();
                Eigen::VectorXf::Index max_index;
                tmp_v.maxCoeff(&max_index);

                cent_of_samp[samp_i] = max_index;
                samp_cent_dist[samp_i] = tmp_v(max_index);
                avg_dist += tmp_v(max_index);
            }
        }
        avg_dist /= samp_num;

        // save current centroid
        pre_cent_matrix = *ptr_cent_matrix;

        iter++;
    }

    return true;
}

} // namespace parallel_kmeans
