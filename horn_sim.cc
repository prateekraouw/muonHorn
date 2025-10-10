#include "G4MTRunManager.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "Randomize.hh"
#include "G4GeometryTolerance.hh"
#include "G4TransportationManager.hh"
#include "G4PropagatorInField.hh"
#include "G4ChordFinder.hh"
#include "G4FieldManager.hh"
#include "G4SystemOfUnits.hh"

#include "Cli.hh"

int main(int argc, char** argv){
  
  // Parse CLI
  Args args = parse_args(argc, argv);
  
  // Deterministic seed for repeatability
  G4Random::setTheSeed(123456789);
  G4Random::setTheEngine(new CLHEP::RanecuEngine);
  
  // Construct the default run manager
  auto* runManager = new G4MTRunManager;
  runManager->SetNumberOfThreads(7);
  
  // Variable to store number of events
  int numEvents;
  
  // Set mandatory initialization classes
  if (argc > 1) {
      runManager->SetUserInitialization(new DetectorConstruction(args.hp));
      numEvents = args.hp.n_events;
  } else {
      // Create default HornParams
      HornParams defaultParams;
      defaultParams.a_mm = 0.00873;
      defaultParams.Rout_mm = 840.0;
      defaultParams.r_neck_mm = 250.0;
      defaultParams.r_max_mm = 540.0;
      defaultParams.spacing_mm = 1500.0;
      defaultParams.zMin_mm = 0.0;        // ← ADD THIS
      defaultParams.zMax_mm = 2000.0;     // ← ADD THIS
      defaultParams.I_A = 60000;
      defaultParams.n_events = 10000;
      
      runManager->SetUserInitialization(new DetectorConstruction(defaultParams));
      numEvents = defaultParams.n_events;
  }
  
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization());
  
  // Initialize G4 kernel
  runManager->Initialize();
  
  // Run simulation without visualization
  G4cout << "========================================" << G4endl;
  G4cout << "Running simulation (no visualization)..." << G4endl;
  G4cout << "Horn Parameters:" << G4endl;
  G4cout << "  a_mm: 0.00873" << G4endl;
  G4cout << "  Rout_mm: 840.0" << G4endl;
  G4cout << "  r_neck_mm: 250.0" << G4endl;
  G4cout << "  spacing_mm: 2000.0" << G4endl;
  G4cout << "  Current (I_A): 60000 A" << G4endl;
  G4cout << "  Number of events: " << numEvents << G4endl;
  G4cout << "  Number of threads: 7" << G4endl;
  G4cout << "========================================" << G4endl;
  
  // Execute the simulation with parameters from else block
  runManager->BeamOn(numEvents);
  
  G4cout << "========================================" << G4endl;
  G4cout << "Simulation complete!" << G4endl;
  G4cout << "Check output CSV files for results." << G4endl;
  G4cout << "========================================" << G4endl;
  
  // Job termination
  delete runManager;
  
  return 0;
}
