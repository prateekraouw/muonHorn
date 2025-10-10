#include "SteppingAction.hh"
#include "EventAction.hh"
#include "DetectorConstruction.hh"
#include "RunAction.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4LogicalVolume.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4FieldManager.hh"
#include "G4MagneticField.hh"
#include "G4ElectroMagneticField.hh"
#include "G4TransportationManager.hh"
#include "G4AutoLock.hh"
#include "G4Threading.hh" 
#include "G4AnalysisManager.hh"
#include <fstream>
#include <iomanip>
#include <memory>


SteppingAction::SteppingAction(EventAction* eventAction)
: G4UserSteppingAction(),
  fEventAction(eventAction),
  fScoringVolume(nullptr),
  fDetector1Volume(nullptr),
  fDetector2Volume(nullptr),
  fDetector3Volume(nullptr),
  fDetector4Volume(nullptr)
{}

SteppingAction::~SteppingAction()
{
    // Only print from master thread to avoid multiple outputs
    if (G4Threading::IsMasterThread()) {
        G4cout << "=== Final Particle Summary (Master Thread Only) ===" << G4endl;
        for (auto const& pair : fParticleCounter) {
            if (pair.first == "mu+" || pair.first == "mu-" ||
                pair.first == "pi+" || pair.first == "pi-") {
                G4cout << pair.first << ": " << pair.second << G4endl;
            }
        }
        G4cout << "=============================================" << G4endl;

        // Detector-specific summaries
        G4cout << "\n=== Detector 1 Particles ===" << G4endl;
        for (auto const& pair : fDetector1Particles) {
            G4cout << pair.first << ": " << pair.second << G4endl;
        }
        
        G4cout << "\n=== Detector 2 Particles ===" << G4endl;
        for (auto const& pair : fDetector2Particles) {
            G4cout << pair.first << ": " << pair.second << G4endl;
        }
        
        G4cout << "\n=== Detector 3 Particles ===" << G4endl;
        for (auto const& pair : fDetector3Particles) {
            G4cout << pair.first << ": " << pair.second << G4endl;
        }
        
        G4cout << "\n=== Detector 4 Particles ===" << G4endl;
        for (auto const& pair : fDetector4Particles) {
            G4cout << pair.first << ": " << pair.second << G4endl;
        }
    }
}


void SteppingAction::UserSteppingAction(const G4Step* step)
{
    constexpr G4double kPmin = 100.*MeV;
    constexpr G4double kPmax = 400.*MeV;
  

    // Lazily cache detector logical volumes
    if (!fDetector1Volume) {
        const auto* detCon =
            static_cast<const DetectorConstruction*>(
                G4RunManager::GetRunManager()->GetUserDetectorConstruction());
        if (detCon) {
            fDetector1Volume = detCon->GetDetector1Volume();
            fDetector2Volume = detCon->GetDetector2Volume();
            fDetector3Volume = detCon->GetDetector3Volume();
            fDetector4Volume = detCon->GetDetector4Volume();
        }
    }
  
    // Only care about first step after entering any volume
    if (!step->IsFirstStepInVolume()) return;

    // Safely get pre-step logical volume
    const auto preHandle = step->GetPreStepPoint()->GetTouchableHandle();
    if (!preHandle) return;
    const auto* preVol = preHandle->GetVolume();
    if (!preVol) return;
    auto* preLV = preVol->GetLogicalVolume();
    if (!preLV) return;

    // Identify which detector (if any)
    int detID = 0;
    if(preLV == fDetector2Volume) detID = 2;
    /*else if (preLV == fDetector2Volume) detID = 1;
    else if (preLV == fDetector3Volume) detID = 3;
    else if (preLV == fDetector4Volume) detID = 4;*/

    if (detID == 0) return; // not a detector crossing we track

    // Track & particle
    auto* trk = step->GetTrack();
    if (!trk) return;
    const G4String pname = trk->GetDefinition()->GetParticleName();

    // Only muons and pions
    if (!(/*pname=="mu+" || pname=="mu-" || pname=="pi-" ||*/ pname=="pi+")) return;
   
    // Pre-step kinematics (always valid here)
    const auto* pre = step->GetPreStepPoint();
    const G4double p_entry = pre->GetMomentum().mag(); // MeV/c
    if (!(p_entry >= kPmin && p_entry < kPmax)) return; // momentum cut: 100 MeV < p < 400 MeV
    
    const G4ThreeVector pos = pre->GetPosition();
    const G4ThreeVector preMom = pre->GetMomentum();
    const G4double Ekin = trk->GetKineticEnergy();
    const G4double t = pre->GetGlobalTime(); // ns
    
    // Post-step info (guarded)
    const auto* post = step->GetPostStepPoint();
    const G4VProcess* proc = (post ? post->GetProcessDefinedStep() : nullptr);
    const G4String procName = (proc ? proc->GetProcessName() : "none");
    
    // Accumulate path length for this track (at this detector boundary step)
    const G4int tid = trk->GetTrackID();

    // Count particles per detector (thread-local counting)
    fParticleCounter[pname]++;
    if (detID == 1) fDetector1Particles[pname]++;
    else if (detID == 2) fDetector2Particles[pname]++;
    else if (detID == 3) fDetector3Particles[pname]++;
    else if (detID == 4) fDetector4Particles[pname]++;



    // ---- Log to RunAction (MT-safe inside Record6DVector) ----
    if (auto* runAct = dynamic_cast<RunAction*>(
            const_cast<G4UserRunAction*>(G4RunManager::GetRunManager()->GetUserRunAction())))
    {
        runAct->Record6DVector(
            detID, pname,
            pos, preMom,               // using pre-step state at entry
            trk->GetKineticEnergy()
        );
    }
}
