#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <chrono>
#include <iomanip>
using namespace std::chrono;
#include "median.hpp"
template<class T>
struct format {
    T v;
    format(T v) : v(v) {}
    friend std::ostream &operator<<(std::ostream &os, const format &format) {
        os << std::setw(10) << format.v << std::right;
        return os;
    }
};
using namespace std;
int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int ws,rk;
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
	const int S = 100000;
	vector<double> v(S), r(S * ws);
    srand(rk + time(NULL));
	int MAX_TRIAL = 10;
	for(int trial = 0; trial < MAX_TRIAL; ++trial){
	    if(!rk) cout << "trial - " << trial << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        generate(v.begin(), v.end(), []() { return (double) rand() / (double) RAND_MAX ; });

        MPI_Barrier(MPI_COMM_WORLD);
        auto part1  = steady_clock::now();
        auto parmed = par::median(begin(v), end(v));
        auto part2  = steady_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);

        auto seqt1 = steady_clock::now();
        MPI_Gather(v.data(), v.size(), par::get_mpi_type<decltype(v)::value_type>(), r.data(), v.size(), par::get_mpi_type<decltype(v)::value_type>(), 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        auto seqt2 = steady_clock::now();
        auto tupload = duration_cast<milliseconds>(seqt2-seqt1).count();
        std::cout << std::setw(10) << "t upload : " << format(tupload) << std::endl;

        if(!rk) {
            std::cout << "par median: "  << format(parmed) << format(duration_cast<milliseconds>(part2-part1).count()) << std::endl;
            auto seqt1 = steady_clock::now();
            //std::nth_element(r.begin(),  (r.begin() + (r.size() / 2)), r.end());
            float med   = *(r.begin() + r.size() / 2);
            auto seqt2 = steady_clock::now();
            std::cout << std::setw(10) << "qs  median: " << format(med) << format((duration_cast<milliseconds>(seqt2-seqt1).count()  + tupload)) << std::endl;
            seqt1 = steady_clock::now();
            //med   = nlogn_median(r);

            seqt2 = steady_clock::now();
            std::cout << std::setw(10) << "sortmedian: " << format(med) << format(duration_cast<milliseconds>(seqt2-seqt1).count()) << std::endl;
            //assert(med == parmed);

        }
    }
	MPI_Finalize();
 	return 0;

}
