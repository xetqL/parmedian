//
// Created by xetql on 8/31/20.
//

#ifndef PARALLEL_MEDIAN_MEDIAN_HPP
#define PARALLEL_MEDIAN_MEDIAN_HPP
#include <vector>
#include <mpi.h>
#include <random>
#include <iterator>
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
        return  (double) (v.at(v.size() / 2 - 1) + v.at(v.size() / 2)) / 2.0;
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

template<class InputIt>
void partial_sort(InputIt begin, InputIt end, int stride) {
    auto itp = begin, itn = itp;
    while(itn != end) {
        while(itn != end && std::distance(itp, itn) < stride){
            itn++;
        }
        std::sort(itp, itn);
        itp = itn;
    }
}
template<class InputIt>
std::vector<typename std::iterator_traits<InputIt>::value_type> partial_medians(InputIt begin, InputIt end, int stride)  {
    std::vector<typename std::iterator_traits<InputIt>::value_type> medians; medians.reserve(std::distance(begin, end) / stride);
    auto itp = begin, itn = itp;
    while(itn != end) {
        int _stride = stride;
        while(itn != end && _stride > 0){
            itn++;
            _stride--;
        }
        std::sort(itp, itn);
        medians.push_back(*(itp + (stride-_stride)/2));
        itp = itn;
    }
    return medians;
}
namespace par {
    int ws,rk;
    template<class T>
    constexpr MPI_Datatype get_mpi_type(){
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
        else return MPI_DATATYPE_NULL;
    }

    namespace {
    template<class T>
    double median(std::vector<T> x, size_t look_for) {
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        /**
         * Until vec.size() == 1
         * 1. Someone selects pivot and broadcast
         * 2. All: split into le and gr
         * 3. All: keeps le if SUM(le_p) > SUM(gr_p) for all p in procs
         * 4. Go back to 1.
         */
        size_t cnt;
        std::array<size_t, 2> split_sizes {};
        std::array<T, 2> pivot_msg{};
        std::vector<T> unfiltered_medians(2*ws);
        std::vector<T> all_medians(ws);
        constexpr T zero = (T) 0, one = (T) 1;
        auto itp = x.begin(), itn = x.end();
        T pivot;
        do {
            //
            T median_of_medians;
            auto medians = partial_medians(itp, itn, 5);
            std::nth_element(medians.begin(), medians.end(), medians.begin() + medians.size() / 2);
            if(!medians.empty()) {
                pivot_msg.at(0) = one;
                pivot_msg.at(1) = *(medians.begin() + medians.size() / 2);
            } else {
                pivot_msg.at(0) = zero;
            }
            MPI_Gather(pivot_msg.data(), 2, get_mpi_type<T>(), unfiltered_medians.data(), 2, get_mpi_type<T>(), 0, MPI_COMM_WORLD);

            cnt = 0;
            for(size_t i = 0; i < ws; ++i) {
                all_medians.at(cnt) = unfiltered_medians.at(2*i+1);
                cnt += unfiltered_medians.at(2*i);
            }
            if(!rk) {
                std::nth_element(all_medians.begin(), all_medians.begin()+cnt, all_medians.begin() + (cnt / 2));
                pivot = *(all_medians.begin() + (cnt / 2));
            }
            MPI_Bcast(&pivot, 1, get_mpi_type<T>(), 0, MPI_COMM_WORLD);

            auto li = std::partition(itp, itn, [pivot](const auto& v){return v < pivot;});
            auto pi = std::partition(li,  itn, [pivot](const auto& v){return v == pivot;});

            split_sizes = {std::distance(itp, li), std::distance(li, pi)};

            MPI_Allreduce(MPI_IN_PLACE, &split_sizes, 2, get_mpi_type<size_t>(), MPI_SUM, MPI_COMM_WORLD);

            if(look_for < split_sizes[0]) {
                itn = li;
            } else if(look_for >= split_sizes[0] + split_sizes[1]){
                itp = pi;
                look_for = look_for - split_sizes[0] - split_sizes[1];
            }

        } while (!(look_for >= split_sizes[0] && look_for < split_sizes[0] + split_sizes[1]));

        return pivot;
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
