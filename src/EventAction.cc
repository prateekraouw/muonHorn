#include "EventAction.hh"
#include "RunAction.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

EventAction::EventAction()
: G4UserEventAction(),
  fEdep(0.),
  fMuonsAtDetector1(0),
  fPionsAtDetector1(0),
  fMuonsAtDetector2(0),
  fPionsAtDetector2(0),
  fMuonsAtDetector3(0),
  fPionsAtDetector3(0),
  fMuonsAtDetector4(0),
  fPionsAtDetector4(0)
{
  // Constructor implementation (if needed)
}

EventAction::~EventAction()
{
  // Destructor implementation (if needed)
}

void EventAction::BeginOfEventAction(const G4Event*)
{
  // Reset all accumulated values at the beginning of each event
  fEdep = 0.;
  fMuonsAtDetector1 = 0;
  fPionsAtDetector1 = 0;
  fMuonsAtDetector2 = 0;
  fPionsAtDetector2 = 0;
  fMuonsAtDetector3 = 0;
  fPionsAtDetector3 = 0;
  fMuonsAtDetector4 = 0;
  fPionsAtDetector4 = 0;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  // Print event information
  G4int eventID = event->GetEventID();
  G4cout << "\n--------------------" << G4endl;
  G4cout << "Event " << eventID << " completed." << G4endl;
  G4cout << "Energy deposit: " << fEdep/MeV << " MeV" << G4endl;
  
  // Print detector counts
  G4cout << "Detector 1 - Muons: " << fMuonsAtDetector1 
         << ", Pions: " << fPionsAtDetector1 << G4endl;
  G4cout << "Detector 2 (10m) - Muons: " << fMuonsAtDetector2 
         << ", Pions: " << fPionsAtDetector2 << G4endl;
  G4cout << "Detector3 - Muons: " << fMuonsAtDetector3
         << ", Pions: " << fPionsAtDetector3<<G4endl;
   G4cout << "Detector4 - Muons: " << fMuonsAtDetector4
         << ", Pions: " << fPionsAtDetector4<<G4endl;
  G4cout << "--------------------" << G4endl;
}
