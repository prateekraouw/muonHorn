#include "PhysicsList.hh"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4SystemOfUnits.hh"
#include "G4ProductionCutsTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"
#include "G4UnitsTable.hh"

PhysicsList::PhysicsList() 
: G4VModularPhysicsList()
{
  // Default cut value
  SetDefaultCutValue(1.0*mm);

  // EM Physics
  RegisterPhysics(new G4EmStandardPhysics());
  RegisterPhysics(new G4EmExtraPhysics());
  
  // Decay Physics
  RegisterPhysics(new G4DecayPhysics());
  
  // Radioactive decay
  RegisterPhysics(new G4RadioactiveDecayPhysics());
  
  // Hadron Elastic Physics
  RegisterPhysics(new G4HadronElasticPhysics());
  
  // Hadron Physics - handles inelastic interactions
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT());
  
  // Stopping Physics
  RegisterPhysics(new G4StoppingPhysics());
  
  // Ion Physics
  RegisterPhysics(new G4IonPhysics());
}

PhysicsList::~PhysicsList()
{
}

void PhysicsList::SetCuts()
{
  // Call the base class method first
  G4VModularPhysicsList::SetCuts();

  // Set energy threshold to 8 GeV for neutrons, electrons, and photons
  const G4double energyThreshold = 8.0 * GeV;
  
  // Set energy thresholds using particle names
  G4cout << "Setting energy threshold for neutrons: " << energyThreshold/GeV << " GeV" << G4endl;
  SetCutValue(energyThreshold, "neutron");
  
  G4cout << "Setting energy threshold for electrons: " << energyThreshold/GeV << " GeV" << G4endl;
  SetCutValue(energyThreshold, "e-");
  
  G4cout << "Setting energy threshold for photons: " << energyThreshold/GeV << " GeV" << G4endl;
  SetCutValue(energyThreshold, "gamma");
  
  // Print summary
  G4cout << "\n----------------------------------------------------------" << G4endl;
  G4cout << "Physics List: Energy threshold for neutrons, electrons," << G4endl;
  G4cout << "and photons set to " << energyThreshold/GeV << " GeV" << G4endl;
  G4cout << "Only particles with energy > 8 GeV will be tracked" << G4endl;
  G4cout << "----------------------------------------------------------\n" << G4endl;
  
  // Dump the cuts table for reference
  DumpCutValuesTable();
}