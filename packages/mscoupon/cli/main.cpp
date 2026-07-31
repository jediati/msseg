#include <exception>
#include <iostream>

namespace mscoupon {
int run_cli(int argc, char** argv);
}

int main(int argc, char** argv) {
  try {
    return mscoupon::run_cli(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
