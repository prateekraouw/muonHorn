#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

class G4ParticleGun;
class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();    
    virtual ~PrimaryGeneratorAction();

    // Method from G4VUserPrimaryGeneratorAction
    virtual void GeneratePrimaries(G4Event* event);
    
  private:
    G4ParticleGun* fParticleGun;
    bool fGunVisCreated;
    
    // Method to create a visible representation of the particle gun
    void CreateGunVisualization(const G4ThreeVector& position, 
                               const G4ThreeVector& direction,
                               G4double angleRad);
};

#endif