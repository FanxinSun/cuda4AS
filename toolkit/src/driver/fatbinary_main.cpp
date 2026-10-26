#include <iostream>
#include <string>
#include <vector>
#include <fstream>

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-V" || arg == "--version") {
            std::cout << "fatbinary: cuda4AS fatbinary generator\n"
                      << "release 12.0, V12.0.0\n";
            return 0;
        }
        if (arg == "-o" && i + 1 < argc) {
            std::ofstream out(argv[i + 1]);
            out.close();
        }
    }
    return 0;
}
