#include "PrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(nullptr)
{
  G4int nofParticles = 1;
  fParticleGun = new G4ParticleGun(nofParticles);

  // Define the beam as protons
  G4ParticleDefinition* particleDefinition
    = G4ParticleTable::GetParticleTable()->FindParticle("proton");
  fParticleGun->SetParticleDefinition(particleDefinition);

  // Set initial energy of proton beam (8 GeV)
  fParticleGun->SetParticleEnergy(8.0*GeV);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
      const G4int    nPerEvent    = 1;      // particles per event (the bunch size)
      const G4double sigmaX       = 5.0*mm;   // transverse spot size (rms) in x
      const G4double sigmaY       = 5.0*mm;   // transverse spot size (rms) in y
      const G4double sigmaThetaX  = 0.0*mrad; // angular divergence (rms) about x
      const G4double sigmaThetaY  = 0.0*mrad; // angular divergence (rms) about y
      const G4double z            = -0.5*m;   // source plane
    
      // Ensure we shoot ONE particle per GeneratePrimaryVertex call.
      fParticleGun->SetNumberOfParticles(1);
    
      for (G4int i = 0; i < nPerEvent; ++i) {
        // Gaussian transverse position
        const G4double x = CLHEP::RandGauss::shoot(0.0, sigmaX);
        const G4double y = CLHEP::RandGauss::shoot(0.0, sigmaY);
    
        // Gaussian small-angle slopes (theta_x, theta_y) relative to +z
        const G4double tx = CLHEP::RandGauss::shoot(0.0, sigmaThetaX);
        const G4double ty = CLHEP::RandGauss::shoot(0.0, sigmaThetaY);
    
        // Build direction vector; for small angles, (tx, ty, 1) is a good approximation
        G4ThreeVector dir(tx, ty, 1.0);
        dir = dir.unit();
    
        fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));
        fParticleGun->SetParticleMomentumDirection(dir);
    
        // Energy was set in the constructor (8 GeV)
        fParticleGun->GeneratePrimaryVertex(anEvent);
      }
}
