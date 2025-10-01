#ifndef HORNFIELD_HH
#define HORNFIELD_HH

#include "G4MagneticField.hh"
#include "globals.hh"

/**
 * HornField
 * ---------
 * Analytic toroidal horn field with optional flat "nose" section then a parabola:
 *
 *   r_in^2(z) = r_const^2                      ,  z <= zt_mm
 *            = r_const^2 + a_mm * (z - zt_mm)  ,  z >  zt_mm
 *
 * Field exists only if:
 *   - z in [zGeomMin_mm, zGeomMax_mm]  (geometry window)
 *   - z in [zPowMin_mm , zPowMax_mm ]  (powered window)
 *   - r_in(z) < r < Rout_mm
 *
 * Units:
 *   - Inputs & coordinates: mm (Geant4 default length)
 *   - Current I: Ampere
 *   - Output field: Tesla
 */
class HornField : public G4MagneticField {
public:
  HornField(G4double Iamp_A,
            G4double zt_mm,
            G4double a_mm,
            G4double rconst_mm,
            G4double Rout_mm,
            G4double zGeomMin_mm, G4double zGeomMax_mm,
            G4double zPowMin_mm , G4double zPowMax_mm, int sign_dir = +1);

  ~HornField() override = default;

  void GetFieldValue(const G4double Point[4], G4double* Bfield) const override;

  // Optional accessors
  G4double GetCurrentA()        const { return fI_A; }
  G4double GetZTransition()     const { return fZt_mm; }
  G4double GetParabolaAmm()     const { return fA_mm; }
  G4double GetRconst()          const { return fRconst_mm; }
  G4double GetRout()            const { return fRout_mm; }
  G4double GetZGeomMin()        const { return fZGeomMin; }
  G4double GetZGeomMax()        const { return fZGeomMax; }
  G4double GetZPowMin()         const { return fZPowMin; }
  G4double GetZPowMax()         const { return fZPowMax; }

private:
  // Helpers
  G4double Rin2_mm2(G4double z_mm) const;

private:
  // Inputs
  G4double fI_A;           // Current [A]
  G4double fZt_mm;         // Nose→parabola transition [mm]
  G4double fA_mm;          // Parabola coefficient "a" in mm (r_in^2 = r_const^2 + a*(z-zt))
  G4double fRconst_mm;     // Nose (flat) inner radius [mm]; set 0 for pure parabola
  G4double fRout_mm;       // Outer radius (annulus outer wall) [mm]

  // Axial gates
  G4double fZGeomMin, fZGeomMax; // geometry window [mm]
  G4double fZPowMin , fZPowMax ; // powered window  [mm]
  int fSign; // converging or diverging geometry

  // μ0/(2π) in SI (H/m)
  static constexpr G4double kMu0Over2Pi = 2.0e-7;
};

#endif // HORNFIELD_HH
