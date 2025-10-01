#ifndef RunAction_h
#define RunAction_h 1
#include "G4UserRunAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <map>
#include <string>
#include <fstream>
class G4Run;
class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    virtual ~RunAction();
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    
    void AddSecondaryParticle(const G4String& name) { fSecondaryParticles[name]++; }
                              
    // Count particle for summary
    void CountParticle(const G4String& name) { fParticleCounts[name]++; }
    void RecordPionDecay(const G4String& pionType, const G4String& muonType, 
                    G4double pionEnergy, G4double muonEnergy,
                    const G4ThreeVector& position);
                    
    // New functions for 6D vector file
    void Record6DVector(G4int detectorID, const G4String& particleName,
                                 const G4ThreeVector& pos, const G4ThreeVector& mom,
                                 G4double kineticEnergy);
                    
  private:
    std::map<G4String, int> fSecondaryParticles;
    std::map<G4String, int> fParticleCounts;  // For tracking all particles
    std::ofstream fOutputFile;
    G4bool isWorker = true; 
    // File stream for 6D vector output
    std::ofstream file6DVector;

    };
#endif