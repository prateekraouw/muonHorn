// src/Cli.cc
#include "Cli.hh"
#include <cstring>
#include <stdexcept>
static double getd(char**& a){ if(*++a) return std::stod(*a); throw std::runtime_error("bad arg"); }
static int    geti(char**& a){ if(*++a) return std::stoi(*a);  throw std::runtime_error("bad arg"); }
static std::string gets(char**& a){ if(*++a) return std::string(*a); throw std::runtime_error("bad arg"); }

Args parse_args(int argc, char** argv){
  Args x{};
  for(char** a=argv+1; *a; ++a){
    if(!std::strcmp(*a,"--a_mm"))         x.hp.a_mm = getd(a);
    else if(!std::strcmp(*a,"--r_neck_mm")) x.hp.r_neck_mm = getd(a);
    else if(!std::strcmp(*a,"--Rout_mm"))   x.hp.Rout_mm = getd(a);
    else if(!std::strcmp(*a,"--zMin_mm"))   x.hp.zMin_mm = getd(a);
    else if(!std::strcmp(*a,"--zMax_mm"))   x.hp.zMax_mm = getd(a);
    else if(!std::strcmp(*a,"--r_max_mm"))   x.hp.r_max_mm = getd(a);
    else if(!std::strcmp(*a,"--I_A"))       x.hp.I_A = getd(a);
    else if(!std::strcmp(*a,"--spacing_mm"))x.hp.spacing_mm = getd(a);
    else if(!std::strcmp(*a,"--n_events"))  x.hp.n_events = geti(a);
    else if(!std::strcmp(*a,"--out_dir"))   x.out_dir = gets(a);
  }
  return x;
}

