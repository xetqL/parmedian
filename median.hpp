//
// Created by xetql on 8/31/20.
//

#ifndef PARALLEL_MEDIAN_MEDIAN_HPP
#define PARALLEL_MEDIAN_MEDIAN_HPP
#include <vector>
#include <mpi.h>
#include <random>

template<class T> void split_vec(const std::vector<T>& vec, T pivot, std::vector<T>& le, std::vector<T>& gr, std::vector<T>& pi){
    le.clear();
    gr.clear();
    pi.clear();
    le.reserve(vec.size()/3);
    gr.reserve(vec.size()/3);
    pi.reserve(vec.size()/3);
    for(auto v : vec) {
        if(v < pivot)
            le.push_back(v);
        else if (v > pivot)
            gr.push_back(v);
        else
            pi.push_back(v);
    }
}
template<class T> double nlogn_median(std::vector<T> v){
    sort(v.begin(), v.end());
    if(v.size() % 2)
        return v.at((v.size() / 2.0));
    else
        return 0.5 * (v.at(v.size() / 2 - 1) + v.at(v.size() / 2));
}
template<class T> double median(std::vector<T> x, size_t look_for) {
    std::random_device rd;
    std::vector<T> le, gr, pi;
    do {
        T pivot = x.at(rd() % x.size());
        split_vec(x, pivot, le, gr, pi);
        if(look_for < le.size()) {
            x = le;
        } else if (look_for < le.size() + pi.size()) {
            return pivot;
        } else {
            x = gr;
            look_for = look_for - le.size() - pi.size();
        }
    } while(true);
}
template<class T> double median(std::vector<T> x){
    if(x.size() % 2){
        return median(x, x.size() / 2);
    } else {
        return 0.5 * (median(x, x.size() / 2 - 1) + median(x, x.size() / 2));
    }
}
namespace par {
    int ws,rk;
    template<class T>
    MPI_Datatype get_mpi_type(){
        if constexpr (std::is_same<T, float>::value)                    return MPI_FLOAT;
        if constexpr (std::is_same<T, double>::value)                   return MPI_DOUBLE;
        if constexpr (std::is_same<T, int>::value)                      return MPI_INT;
        if constexpr (std::is_same<T, unsigned int>::value)             return MPI_UNSIGNED;
        if constexpr (std::is_same<T, long>::value)                     return MPI_LONG;
        if constexpr (std::is_same<T, long int>::value)                 return MPI_LONG_INT;
        if constexpr (std::is_same<T, long double>::value)              return MPI_LONG_DOUBLE;
        if constexpr (std::is_same<T, long long>::value)                return MPI_LONG_LONG;
        if constexpr (std::is_same<T, long long int>::value)            return MPI_LONG_LONG_INT;
        if constexpr (std::is_same<T, unsigned long>::value)            return MPI_UNSIGNED_LONG;
        if constexpr (std::is_same<T, unsigned long long>::value)       return MPI_UNSIGNED_LONG_LONG;
        if constexpr (std::is_same<T, short>::value)                    return MPI_SHORT;
        if constexpr (std::is_same<T, short int>::value)                return MPI_SHORT_INT;
        if constexpr (std::is_same<T, char>::value)                     return MPI_CHAR;
        return MPI_DATATYPE_NULL;
    }
    namespace {
    template<class T>
    double median(std::vector<T> x, size_t look_for) {
        std::random_device rd;
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        /**
         * Until vec.size() == 1
         * 1. Someone selects pivot and broadcast
         * 2. All: split into le and gr
         * 3. All: keeps le if SUM(le_p) > SUM(gr_p) for all p in procs
         * 4. Go back to 1.
         */
        std::vector<T> le, gr, pi;
        std::array<size_t, 3> split_sizes{};
        do {
            size_t size = x.size(), total_size, lb, ub, ipivot;
            T pivot;
            MPI_Reduce(&size, &total_size, 1, get_mpi_type<size_t>(), MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Scan(&size, &ub, 1, get_mpi_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
            lb = ub - size;
            if (!rk) {
                ipivot = (rd() % total_size);
            }
            MPI_Bcast(&ipivot, 1, get_mpi_type<size_t>(), 0, MPI_COMM_WORLD);
            if(lb <= ipivot && ipivot < ub) {
                pivot = x.at(ipivot - lb);
                for(auto pe = 0; pe < ws; ++pe)
                    if(pe != rk) MPI_Send(&pivot, 1, get_mpi_type<T>(), pe, 999, MPI_COMM_WORLD);
            } else {
                MPI_Recv(&pivot, 1, get_mpi_type<T>(), MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }

            auto li = std::partition(x.begin(), x.end(), [&pivot](auto v){return v < pivot;});
            auto pi = std::partition(li, x.end(), [&pivot](auto v){return v == pivot;});

            split_sizes = {std::distance(x.begin(), li), std::distance(li, pi)};

            MPI_Allreduce(MPI_IN_PLACE, &split_sizes, 2, get_mpi_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);

            if(look_for < split_sizes[0]) {
                x = std::vector<T>(x.begin(), li);
            } else if (look_for < split_sizes[0] + split_sizes[1]) {
                return pivot;
            } else {
                x = std::vector<T>(pi, x.end());
                look_for = look_for - split_sizes[0] - split_sizes[1];
            }
        } while(true);
    }
    }
    template<class T>
    double median(const std::vector<T>& x) {
        size_t total_size, size = x.size();
        MPI_Allreduce(&size, &total_size, 1, get_mpi_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);
        if(total_size % 2) {
            return median<T>(x, total_size / 2);
        } else {
            return 0.5 * (median<T>(x, total_size / 2 - 1) + median<T>(x, total_size / 2));
        }
    }
    template<class InputIterator>
    double median(InputIterator beg, InputIterator end) {
        std::vector<typename std::iterator_traits<InputIterator>::value_type> x(beg, end);
        return median(x);
    }
}

#endif //PARALLEL_MEDIAN_MEDIAN_HPP
