#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include <map>
#include <fstream>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4FieldManager.hh"
#include "G4MagneticField.hh"
#include "G4TransportationManager.hh"
#include <unordered_map>

class EventAction;
class G4LogicalVolume;

// Struct to store particle information
struct ParticleInfo {
  G4String type;
  G4double energy;
  G4ThreeVector position;
  G4ThreeVector direction;
  G4double kineticEnergy;
};

struct TrackAccum {
  G4ThreeVector lastPos;
  G4double      s = 0.0;  // mm
  bool          inited = false;
};

class SteppingAction : public G4UserSteppingAction
{
public:
  SteppingAction(EventAction* eventAction);
  virtual ~SteppingAction();
  
  // Method called for each step
  virtual void UserSteppingAction(const G4Step*);
  
private:
  EventAction* fEventAction;
  G4LogicalVolume* fScoringVolume;
  G4LogicalVolume* fDetector1Volume;
  G4LogicalVolume* fDetector2Volume;
  G4LogicalVolume* fDetector3Volume;
  G4LogicalVolume* fDetector4Volume;
  
  
  G4LogicalVolume* fCounterVolume;  // Add this for the 10m detector
  
   std::unordered_map<G4int, TrackAccum> fAcc; // by TrackID
  
  // Maps to count particles at each detector
  std::map<G4String, G4int> fParticleCounter;
  std::map<G4String, G4int> fDetector1Particles;
  std::map<G4String, G4int> fDetector2Particles;
  std::map<G4String, G4int> fDetector3Particles;
  std::map<G4String, G4int> fDetector4Particles;
  
  // Store detected particle information
  std::vector<ParticleInfo> fDetectedParticles;
};

#endif