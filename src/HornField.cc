#include "HornField.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

HornField::HornField(G4double Iamp_A,
                     G4double zt_mm,
                     G4double a_mm,
                     G4double rconst_mm,
                     G4double Rout_mm,
                     G4double zGeomMin_mm, G4double zGeomMax_mm,
                     G4double zPowMin_mm , G4double zPowMax_mm,
                     int    sign_dir)
: fI_A(Iamp_A),
  fZt_mm(zt_mm),
  fA_mm(a_mm),
  fRconst_mm(rconst_mm),
  fRout_mm(Rout_mm),
  fZGeomMin(zGeomMin_mm), fZGeomMax(zGeomMax_mm),
  fZPowMin(zPowMin_mm) ,  fZPowMax(zPowMax_mm),
  fSign( (sign_dir>=0) ? +1 : -1 )
{}

G4double HornField::Rin2_mm2(G4double z_mm) const {
    const G4double z_rel = (fSign > 0) ? (z_mm - fZt_mm) : (fZt_mm - z_mm);
    if (z_rel <= 0.0) {
        // upstream (for expanding) or downstream (for contracting) of the vertex → at r = rconst
        return fRconst_mm * fRconst_mm;
    }
    const G4double dr2 = z_rel / fA_mm;    // *** divide by a (mm) ***
    return fRconst_mm * fRconst_mm + dr2;
}

void HornField::GetFieldValue(const G4double Point[4], G4double* B) const {
    const G4double x = Point[0], y = Point[1], z = Point[2];
    B[0]=B[1]=B[2]=0.0;
    if (z < fZPowMin || z > fZPowMax) return;  // geometry window
    if (z < fZPowMin  || z > fZPowMax) return;  // powered window
    
    const G4double r2 = x*x + y*y;
    if (r2 <= 0.0) return;

    const G4double rin2 = Rin2_mm2(z);
    const G4double r_in = std::sqrt(rin2);
    if (r2 <= rin2) return;              // inside inner conductor → no field
    const G4double r = std::sqrt(r2);
    if (r >= fRout_mm) return;           // outside outer conductor → no field
    
    // Bphi = (μ0/2π) I / r_m
    const G4double r_m  = r * 1.0e-3;
    const G4double Bphi = (kMu0Over2Pi * fI_A) / r_m;
    
    const G4double invr = 1.0/r;
    const G4double cosf = x * invr;
    const G4double sinf = y * invr;
    
    B[0] = -Bphi * sinf;
    B[1] =  Bphi * cosf;
    B[2] =  0.0;
}
