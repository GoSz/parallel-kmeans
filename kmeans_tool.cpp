#include <string.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "parallel_kmeans.h"

using namespace parallel_kmeans;


int ArgPos(char *str, uint32_t argc, char **argv) 
{
    for (int a = 1; a < argc; a++) 
    {
        if (!strcmp(str, argv[a]))
        {
            if (a == argc - 1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

void split_string(const std::string& str, const char delim,
                  std::vector<std::string>& res_vec, bool allow_empty)
{
    res_vec.clear();
    std::stringstream ss;
    ss.str(str);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        if (item.length() == 0 && !allow_empty)
        {
            continue;
        }
        res_vec.push_back(item);
    }
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        printf("Parameters for %s:\n", argv[0]);
        printf("  -f <file>\n");
        printf("      input sample file, one vector per line, default delimiter is whitespace\n");
        printf("  -o <file>\n");
        printf("      use <file> to save the clustering index of each sample\n");
        printf("  -c <file>\n");
        printf("      use <file> to save the clustering centroid\n");
        printf("  -k <int>\n");
        printf("      set the cluster number\n");
        printf("  -d <int>\n");
        printf("      set the dimension of sample vector\n");
        printf("  -t <int>\n");
        printf("      set the distance type of kmeans, 1=euclidean (default), 2=cosine\n");
        printf("  -init <int>\n");
        printf("      set the init type of kmeans, 1=randinit (default), 2=kmeans++init\n");
        printf("  -th <int>\n");
        printf("      set the num of thread\n");
        printf("  -it <int>\n");
        printf("      set max iter num of clustering\n");
        printf("\nExamples:\n");
        printf("%s -f input_file -o output_file -c centroid_file -k 64 -d 100 -it 100 -t 1 -init 1\n", argv[0]);
        return 1;
    }

    uint32_t cent_num = 0;      // 聚类数目
    uint32_t iter_max = 0;      // 最大迭代轮数
    uint32_t dim = 0;           // 样本维度
    uint32_t dist_type = 1;     // 距离类型
    uint32_t init_type = 1;     // 初始化方式
    uint32_t thread_num = 0;    // omp线程数
    char input_file[1024];
    char output_file[1024];
    char centroid_file[1024];

    int i;
    if ((i = ArgPos((char*)"-f", argc, argv))>0) strcpy(input_file, argv[i+1]);
    if ((i = ArgPos((char*)"-o", argc, argv))>0) strcpy(output_file, argv[i+1]);
    if ((i = ArgPos((char*)"-c", argc, argv))>0) strcpy(centroid_file, argv[i+1]);
    if ((i = ArgPos((char*)"-k", argc, argv))>0) cent_num = strtoul(argv[i+1], NULL, 0);
    if ((i = ArgPos((char*)"-d", argc, argv))>0) dim = strtoul(argv[i+1], NULL, 0);
    if ((i = ArgPos((char*)"-t", argc, argv))>0) dist_type = strtoul(argv[i+1], NULL, 0);
    if ((i = ArgPos((char*)"-init", argc, argv))>0) init_type = strtoul(argv[i+1], NULL, 0);
    if ((i = ArgPos((char*)"-it", argc, argv))>0) iter_max = strtoul(argv[i+1], NULL, 0);
    if ((i = ArgPos((char*)"-th", argc, argv))>0) thread_num = strtoul(argv[i+1], NULL, 0);

    if (cent_num == 0 || dim == 0)
    {
        std::cerr << "Error Param, k(cent_num) or d(dim) is zero!" << std::endl;
        return -1;
    }

    // 读取样本文件
    std::ifstream f_samp(input_file);
    if (!f_samp)
    {
        std::cerr << "Open input sample file[" << input_file << "] failed!" << std::endl;
        return -1;
    }

    // 确定样本数量
    std::string line; 
    uint32_t line_count = 0;
    while (getline(f_samp, line))
    {
        line_count++;
    }
    uint32_t samp_num = line_count;
    f_samp.clear();
    f_samp.seekg(0, std::ios_base::beg);

    Matrix samp_matrix(samp_num, dim);
    std::vector<std::string> array_vec;
    line_count = 0;
    std::cerr << "Loading sample array...";
    while (getline(f_samp, line))
    {
        split_string(line, ' ', array_vec, false);
        for(uint32_t i = 0; i < dim; i++) 
        {
            samp_matrix(line_count, i) = atof(array_vec[i].c_str());
        }
        line_count++;
    }
    std::cerr << "\rLoading sample array finished, total sample number: "<< line_count << std::endl;
    f_samp.close();

    // 输出文件路径确认
    std::ofstream f_out(output_file);
    if (!f_out.is_open())
    {
        std::cerr << "Open output index file[" << output_file << "] failed!" << std::endl;
        return -1;
    }

    std::ofstream f_cent(centroid_file);
    if (!f_cent.is_open())
    {
        std::cerr << "Open output centroid file[" << centroid_file << "] failed!" << std::endl;
        return -1;
    }

    // 进行K-means聚类
    ParallelKmeans kmeans(cent_num, dim, static_cast<DistType>(dist_type), (unsigned)time(NULL));
    if (!kmeans.cluster(samp_matrix, static_cast<CentInitType>(init_type), iter_max, 0.01, thread_num))
    {
        std::cerr << "ParallelKmeans clustering failed!" << std::endl;
        return -1;
    }

    // 输出每个样本对应的类簇
    for (uint32_t samp_i = 0; samp_i < samp_num; samp_i++)
    {
        f_out << kmeans.cent_of_samp[samp_i] << "\t" << kmeans.samp_cent_dist[samp_i] << std::endl;
    }
    f_out.close();

    // 输出每个类簇的聚类中心
    for (uint32_t cent_i = 0; cent_i < cent_num; cent_i++)
    {
        f_cent << cent_i;
        for (uint32_t d = 0; d < dim; d++)
        {
            f_cent << '\t' << kmeans.ptr_cent_matrix->coeff(cent_i, d);
        }
        f_cent << std::endl;
    }
    f_cent.close();

    return 0;
}
