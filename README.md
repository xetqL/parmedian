# parmedian
Generic header only C++ MPI Implementation of the parallel quickselect algorithm for median finding.
# Requirements
- MPI
- C++14
# To use
Copy the file "median.hpp" into your include directory.
## Example
```c++
#include "median.hpp" // requires MPI
//...
using namespace std;
int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  //...
  vector<int> some_numbers(10000);
  generate(begin(some_numbers), end(some_numbers), [](){return rand() % 1000;});
  cout << par::median(begin(some_numbers), end(some_numbers)) << endl;
  
  MPI_Finalize();
  return EXIT_SUCCESS;
}
```
