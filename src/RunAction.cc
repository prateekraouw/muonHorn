#include "RunAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4UnitsTable.hh"
#include "G4AnalysisManager.hh"
#include "G4AutoLock.hh"
#include "G4Threading.hh"    // defines G4Mutex, G4MUTEX_INITIALIZER
#include <fstream>
#include <string>
#include <sstream>

// Thread-local guard: ensure we book exactly once per thread
static G4ThreadLocal bool t_ntupleBooked = false;

RunAction::RunAction() : G4UserRunAction() {
}

RunAction::~RunAction() {
  // Nothing here; master closes in EndOfRunAction, workers close their CSV there too.
}

void RunAction::BeginOfRunAction(const G4Run* run) {
    const G4int runID = run->GetRunID();

    // Only workers open CSV files (master thread doesn't need file I/O)
    if (!G4Threading::IsMasterThread()) {
        const G4int tid = G4Threading::G4GetThreadId();
        std::ostringstream vfn; 
        vfn << "6D_vector_run" << runID << "_t" << tid << ".csv";
        file6DVector.open(vfn.str(), std::ios::out);
        if (file6DVector) {
            file6DVector << "Det,P_Type,"
                        << "x[mm],y[mm],z[mm],"
                        << "px[MeV/c],py[MeV/c],pz[MeV/c],"
                        << "E[GeV]\n";
            G4cout << "[T" << tid << "] Opened 6D vector file: " << vfn.str() << G4endl;
        }
    } else {
        G4cout << "[MASTER] Begin run " << runID << " (no file output)" << G4endl;
    }
}

void RunAction::EndOfRunAction(const G4Run*) {
    if (G4Threading::IsMasterThread()) {
        G4cout << "[MASTER] End of run completed." << G4endl;
    } else {
        if (file6DVector.is_open()) {
            file6DVector.close();
            const G4int tid = G4Threading::G4GetThreadId();
            G4cout << "[T" << tid << "] Closed 6D vector file." << G4endl;
        }
    }
}

void RunAction::Record6DVector(G4int detectorID, const G4String& particleName,
                               const G4ThreeVector& pos, const G4ThreeVector& mom,
                               G4double kineticEnergy)
{
  if (!file6DVector.is_open()) return;
  auto finite3 = [](const G4ThreeVector& v){
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  };
  if (!finite3(pos) || !finite3(mom)) return;

  file6DVector
    << detectorID << ','
    << particleName << ','
    << pos.x()/mm << ','
    << pos.y()/mm << ','
    << pos.z()/mm << ','
    << mom.x()/MeV << ','
    << mom.y()/MeV << ','
    << mom.z()/MeV << ','
    << kineticEnergy/GeV << '\n';
}
