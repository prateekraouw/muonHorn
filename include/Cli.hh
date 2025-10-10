// include/Cli.hh
#pragma once
#include "Params.hh"
#include <string>
struct Args { HornParams hp{}; std::string out_dir = "out"; };
Args parse_args(int argc, char** argv);

