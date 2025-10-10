#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4MagneticField.hh"
#include "G4FieldManager.hh"
#include "Params.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    explicit DetectorConstruction(const HornParams& hp);
    ~DetectorConstruction() override;
    G4VPhysicalVolume* Construct() override;

    // Original getters
    G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }
    G4LogicalVolume* GetDetector1Volume() const { return fDetector1Volume; }
    G4LogicalVolume* GetDetector2Volume() const { return fDetector2Volume; }
    G4LogicalVolume* GetDetector3Volume() const { return fDetector3Volume; }
    G4LogicalVolume* GetDetector4Volume() const { return fDetector4Volume; }

    // Original position getters
    G4ThreeVector GetDetector1Position() const { return fDetector1Position; }
    G4ThreeVector GetDetector2Position() const { return fDetector2Position; }
    G4ThreeVector GetDetector3Position() const { return fDetector3Position; }
    G4ThreeVector GetDetector4Position() const { return fDetector4Position; }


private:

    G4LogicalVolume* fWorldLogical;
    G4LogicalVolume* fScoringVolume;
    G4LogicalVolume* fDetector1Volume;
    G4LogicalVolume* fDetector2Volume;
    G4LogicalVolume* fDetector3Volume;
    G4LogicalVolume* fDetector4Volume;

    G4ThreeVector fDetector1Position;
    G4ThreeVector fDetector2Position;
    G4ThreeVector fDetector3Position;
    G4ThreeVector fDetector4Position;
    
    HornParams hp_;
    G4LogicalVolume* BuildHornLV(const HornParams& hp); // builds one horn (local z: 0..L)

};


#endif
