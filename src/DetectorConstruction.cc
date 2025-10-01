#include "DetectorConstruction.hh"
#include "HornField.hh"
#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Paraboloid.hh"
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VisAttributes.hh"
#include "G4VPhysicalVolume.hh"
#include "G4TransportationManager.hh"
#include "G4MagIntegratorDriver.hh"
#include "G4IntegrationDriver.hh" // new driver
#include "G4ChordFinder.hh"
#include "G4ClassicalRK4.hh"
#include "G4MagneticField.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4UserLimits.hh"

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction(),
  fScoringVolume(nullptr),
  fWorldLogical(nullptr),
  fDetector1Volume(nullptr),
  fDetector2Volume(nullptr),
  fDetector3Volume(nullptr),
  fDetector4Volume(nullptr)
{
    
}

DetectorConstruction::~DetectorConstruction()
{
    
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    // ========== MATERIALS ==========
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* world_mat = nist->FindOrBuildMaterial("G4_Ar");
    G4Material* graphite_mat = nist->FindOrBuildMaterial("G4_GRAPHITE");
    G4Material* tungsten_mat = nist->FindOrBuildMaterial("G4_W");
    G4Material* beryllium = nist->FindOrBuildMaterial("G4_Be");
    G4Material* copper = nist->FindOrBuildMaterial("G4_Cu");
    G4Material* scintillator_mat = nist->FindOrBuildMaterial("G4_Ar");
    G4Material* vacuum_mat = nist->FindOrBuildMaterial("G4_Galactic");

    // ========== WORLD VOLUME ==========
    G4double world_size = 10000*cm;
    G4double world_radius = 0.5*world_size;
    G4double world_length = 20*world_size;

    G4Tubs* solidWorld = new G4Tubs("World", 0, world_radius, 0.5*world_length, 0*deg, 360*deg);
    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, world_mat, "World");
    fWorldLogical = logicWorld;

    G4VPhysicalVolume* physWorld = new G4PVPlacement(nullptr, G4ThreeVector(), logicWorld,
                                                    "World", nullptr, false, 0, true);

    // ========== GRAPHITE TARGET ==========
    G4double block_x = 10*cm;
    G4double block_y = 10*cm;
    G4double block_z = 40*cm;

    G4Box* solidGraphite = new G4Box("Graphite", 0.5*block_x, 0.5*block_y, 0.5*block_z);
    G4LogicalVolume* logicGraphite = new G4LogicalVolume(solidGraphite, graphite_mat, "Graphite");

    G4RotationMatrix* rotation = new G4RotationMatrix();
    rotation->rotateX(0*rad);
    
    
    // ========= HORN ===============
    
    // --- horn profile & sizes ---
    const G4double a_mm    = 0.00873;   // mm^-1  (parabola hits r_max at 2000 mm)
    const G4double Rout    = 840.0*mm;  // outer conductor radius
    const G4double r_neck  = 250.0*mm;  // neck radius
    const G4double r_max   = 540.0*mm;  // parabola end radius
    const G4double tInner  = 1.0*mm;    // inner shell thickness (ghost)
    const G4double tOuter  = 1.0*mm;    // outer shell thickness

    
    /// --- horn spans ---
    const G4double z1Beg = 0.0*mm, z1End = 2000.0*mm;  // Horn 1: 0 → 2000
    const G4double z2Beg = 3000.0*mm, z2End = 5000.0*mm;  // Horn 2: 3000 → 5000
    
    // ===================== VIS (inner scintillator_mat, outer light blue, gas light) =====================
    auto visscintillator_mat = new G4VisAttributes(G4Colour(0.80, 0.35, 0.15, 0.50)); // scintillator_mat translucent
    visscintillator_mat->SetForceSolid(true);
    auto visGas    = new G4VisAttributes(G4Colour(0.20, 0.60, 1.00, 0.08)); // faint blue gas
    visGas->SetForceSolid(true);
    auto visOuter  = new G4VisAttributes(G4Colour(0.60, 0.80, 1.00, 0.15)); // light blue shell
    visOuter->SetForceSolid(true);
    
    // ===================== FIELDS & FIELD MANAGERS =====================
    const G4double I1 = +70000.0;  // A (focus)
    const G4double I2 = -70000.0;  // A (defocus)
    
    // HornField(Iamp_A, zt_mm, a_mm, rconst_mm, Rout_mm, zGeomMin_mm, zGeomMax_mm, zPowMin_mm, zPowMax_mm, signDr)
    HornField* hf1 = new HornField(I1, 0.0, a_mm, r_neck, Rout, z1Beg, z1End, z1Beg, z1End, +1);
    HornField* hf2 = new HornField(I2, 0.0, a_mm, r_neck, Rout, z2Beg, z2End, z2Beg, z2End, -1);
    
    auto* eq1      = new G4Mag_UsualEqRhs(hf1);
    auto* stepper1 = new G4ClassicalRK4(eq1, 8);
    // Use G4IntegrationDriver instead of deprecated G4MagInt_Driver
    auto* driver1 = new G4MagInt_Driver(0.5*mm, stepper1, stepper1->GetNumberOfVariables());
    auto* chord1   = new G4ChordFinder(driver1);
    
    
    
    G4FieldManager* fmHorn1 = new G4FieldManager(hf1, chord1);
    
    // CRITICAL: Relaxed tolerances for 200 kA horn field
    fmHorn1->SetDeltaOneStep(0.01*mm);           // Increased from 0.05*mm
    fmHorn1->SetMinimumEpsilonStep(1e-5);       // Relaxed from 1e-4
    fmHorn1->SetMaximumEpsilonStep(1e-3);       // Increased from 1e-3
    fmHorn1->GetChordFinder()->SetDeltaChord(0.1*mm);  // Increased from 0.5*mm
    
    // Additional safety parameters
    fmHorn1->SetDeltaIntersection(0.1*mm);      // Intersection accuracy
    fmHorn1->SetAccuraciesWithDeltaOneStep(0.5*mm); // Consistent accuracy
    
    
    //========Horn 2 Field Manager=============
    
    auto* eq2      = new G4Mag_UsualEqRhs(hf2);
    auto* stepper2 = new G4ClassicalRK4(eq2, 8);
    auto* driver2  = new G4MagInt_Driver(0.5*mm, stepper2, stepper2->GetNumberOfVariables());
    auto* chord2   = new G4ChordFinder(driver2);
    
    G4FieldManager* fmHorn2 = new G4FieldManager(hf2, chord2);
    fmHorn2->SetDeltaOneStep(0.01*mm);
    fmHorn2->SetMinimumEpsilonStep(1e-5);
    fmHorn2->SetMaximumEpsilonStep(1e-3);
    fmHorn2->GetChordFinder()->SetDeltaChord(0.1*mm);
    
    // ===================== HORN 1 GEOMETRY =====================
    {
        const G4double hHalfZ = 0.5*(z1End - z1Beg);  // 1000mm
        const G4double hZpos  = 0.5*(z1End + z1Beg);  // 1000mm
        const G4double tolerance = 0.01*mm;
        const G4double safeRout = Rout - tolerance;
    
        // === OUTER CONDUCTOR SHELL ===
        auto* sH1OuterShell = new G4Tubs("H1_OuterShell", safeRout, Rout + tOuter, 
                                         hHalfZ, 0*deg, 360*deg);
        auto* lvH1OuterShell = new G4LogicalVolume(sH1OuterShell, copper, "H1_OuterShellLV");
        lvH1OuterShell->SetVisAttributes(visOuter);
    
        // === PARABOLIC INNER CONDUCTOR (SOLID) ===
        auto* sH1ParaConductor = new G4Paraboloid("H1_ParaConductor", hHalfZ, r_neck, r_max);
        auto* lvH1ParaConductor = new G4LogicalVolume(sH1ParaConductor, copper, "H1_ParaConductorLV");
        lvH1ParaConductor->SetVisAttributes(visOuter);
    
        // === VACUUM CAVITY INSIDE PARABOLIC CONDUCTOR ===
        const G4double rNeckInner = r_neck - tInner;
        const G4double rMaxInner = r_max - tInner;
        const G4double cavityHalfZ = hHalfZ - tolerance;
    
        if (rNeckInner <= 0.1*mm || rMaxInner <= rNeckInner + 0.1*mm) {
            G4Exception("DetectorConstruction", "InvalidGeometry", 
                        FatalException, "Horn inner dimensions too small");
        }
    
        auto* sH1Cavity = new G4Paraboloid("H1_Cavity", cavityHalfZ, rNeckInner, rMaxInner);
        auto* lvH1Cavity = new G4LogicalVolume(sH1Cavity, scintillator_mat, "H1_CavityLV");
        lvH1Cavity->SetVisAttributes(visGas);
    
        // === GAS REGION (BETWEEN PARABOLIC AND CYLINDRICAL CONDUCTORS) ===
        // This is where the magnetic field should exist
        auto* sH1GasRegion = new G4SubtractionSolid("H1_GasRegion", 
                                                    new G4Tubs("H1_GasCyl", 0, safeRout, hHalfZ, 0*deg, 360*deg),
                                                    sH1ParaConductor);
        auto* lvH1GasRegion = new G4LogicalVolume(sH1GasRegion, scintillator_mat, "H1_GasRegionLV");
        auto* limGap = new G4UserLimits(50*mm);
        //lvH1GasRegion->SetUserLimits(limGap);
        lvH1GasRegion->SetVisAttributes(visGas);
        lvH1GasRegion->SetFieldManager(fmHorn1, true); // Field only in gas region
    
        // === DIRECT PLACEMENT IN WORLD VOLUME ===
        // Place outer shell directly in world
        auto* pvH1OuterShell = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                 lvH1OuterShell, "H1_OuterShellPV", logicWorld, false, 0, true);
    
        // Place gas region directly in world  
        auto* pvH1GasRegion = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                lvH1GasRegion, "H1_GasRegionPV", logicWorld, false, 0, true);
    
        // Place parabolic conductor directly in world
        auto* pvH1ParaConductor = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                    lvH1ParaConductor, "H1_ParaConductorPV", logicWorld, false, 0, true);
    
        // Place cavity inside the parabolic conductor (relative positioning)
        auto* pvH1Cavity = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), 
                                             lvH1Cavity, "H1_CavityPV", lvH1ParaConductor, false, 0, true);
        
        new G4PVPlacement(rotation, G4ThreeVector(0,0,25.0*cm),
                                             logicGraphite, "Graphite",
                                             lvH1Cavity, false, 0, true);
    }



     
    // ===================== HORN 2 (defocusing: r_max -> r_neck) =====================
    {
        const G4double hHalfZ = 0.5*(z2End - z2Beg);  // 1000mm
        const G4double hZpos  = 0.5*(z2End + z2Beg);  // 4000mm
        const G4double tolerance = 0.01*mm;
        const G4double safeRout = Rout - tolerance;
    
        // === OUTER CONDUCTOR SHELL ===
        auto* sH2OuterShell = new G4Tubs("H2_OuterShell", safeRout, Rout + tOuter, 
                                         hHalfZ, 0*deg, 360*deg);
        auto* lvH2OuterShell = new G4LogicalVolume(sH2OuterShell, copper, "H2_OuterShellLV");
        lvH2OuterShell->SetVisAttributes(visOuter);
    
        // === PARABOLIC INNER CONDUCTOR (FLIPPED: r_max -> r_neck) ===
        // Create parabolic conductor with flipped orientation
        auto* sH2ParaBase = new G4Paraboloid("H2_ParaBase", hHalfZ, r_neck, r_max);
        
        // Flip the paraboloid to make it go from r_max to r_neck along +z direction
        auto* rotFlipX = new G4RotationMatrix();
        rotFlipX->rotateX(180.*deg);
        
        auto* sH2ParaConductor = new G4DisplacedSolid("H2_ParaConductor", sH2ParaBase, 
                                                      rotFlipX, G4ThreeVector(0,0,0));
        auto* lvH2ParaConductor = new G4LogicalVolume(sH2ParaConductor, copper, "H2_ParaConductorLV");
        lvH2ParaConductor->SetVisAttributes(visOuter);
    
        // === VACUUM CAVITY INSIDE PARABOLIC CONDUCTOR ===
        const G4double rNeckInner = r_neck - tInner;
        const G4double rMaxInner = r_max - tInner;
        const G4double cavityHalfZ = hHalfZ - tolerance;
    
        if (rNeckInner <= 0.1*mm || rMaxInner <= rNeckInner + 0.1*mm) {
            G4Exception("DetectorConstruction", "InvalidGeometry", 
                        FatalException, "Horn 2 inner dimensions too small");
        }
    
        // Create flipped cavity to match the conductor
        auto* sH2CavityBase = new G4Paraboloid("H2_CavityBase", cavityHalfZ, rNeckInner, rMaxInner);
        auto* sH2Cavity = new G4DisplacedSolid("H2_Cavity", sH2CavityBase, 
                                               rotFlipX, G4ThreeVector(0,0,0));
        auto* lvH2Cavity = new G4LogicalVolume(sH2Cavity, scintillator_mat, "H2_CavityLV");
        lvH2Cavity->SetVisAttributes(visGas);
    
        // === GAS REGION (BETWEEN PARABOLIC AND CYLINDRICAL CONDUCTORS) ===
        auto* sH2GasRegion = new G4SubtractionSolid("H2_GasRegion", 
                                                    new G4Tubs("H2_GasCyl", 0, safeRout, hHalfZ, 0*deg, 360*deg),
                                                    sH2ParaConductor);
        auto* lvH2GasRegion = new G4LogicalVolume(sH2GasRegion, scintillator_mat, "H2_GasRegionLV");
        
        // Enhanced user limits for Horn 2 magnetic field
        //auto* limField2 = new G4UserLimits();
        //limField2->SetMaxAllowedStep(5.0*mm);
        //limField2->SetUserMaxTrackLength(10*m);
        //limField2->SetUserMaxTime(10*microsecond);
        
        //lvH2GasRegion->SetUserLimits(limField2);
        lvH2GasRegion->SetVisAttributes(visGas);
        lvH2GasRegion->SetFieldManager(fmHorn2, true); // Use Horn 2 field manager
    
        // === DIRECT PLACEMENT IN WORLD VOLUME ===
        // Place outer shell directly in world
        auto* pvH2OuterShell = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                 lvH2OuterShell, "H2_OuterShellPV", logicWorld, false, 0, true);
    
        // Place gas region directly in world
        auto* pvH2GasRegion = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                lvH2GasRegion, "H2_GasRegionPV", logicWorld, false, 0, true);
    
        // Place parabolic conductor directly in world
        auto* pvH2ParaConductor = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, hZpos), 
                                                    lvH2ParaConductor, "H2_ParaConductorPV", logicWorld, false, 0, true);
    
        // Place cavity inside the parabolic conductor (relative positioning)
        auto* pvH2Cavity = new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), 
                                             lvH2Cavity, "H2_CavityPV", lvH2ParaConductor, false, 0, true);
    
        G4cout << "=== Horn 2 Geometry Summary ===" << G4endl;
        G4cout << "Parabolic conductor (flipped): " << r_max/mm << " mm → " << r_neck/mm << " mm" << G4endl;
        G4cout << "Position: z = " << z2Beg/mm << " to " << z2End/mm << " mm" << G4endl;
        G4cout << "Center at: z = " << hZpos/mm << " mm" << G4endl;
        G4cout << "===============================" << G4endl;
    }



    
    
    // ========== DETECTORS ==========
    G4double detector_thickness = 0.1*cm;

    // Detector 1
    G4double detector1_position = 2.1*m;
    G4Tubs* solidDetector1 = new G4Tubs("Detector1", 0*cm, 70*cm, 0.5*detector_thickness, 0*deg, 360*deg);
    G4LogicalVolume* logicDetector1 = new G4LogicalVolume(solidDetector1, scintillator_mat, "Detector1");
    fDetector1Volume = logicDetector1;
    fDetector1Position = G4ThreeVector(0, 0, detector1_position);
    new G4PVPlacement(nullptr, fDetector1Position, logicDetector1, "Detector1", logicWorld, false, 0, false);

    // Detector 2
    G4double detector2_position = 5.1*m;
    G4Tubs* solidDetector2 = new G4Tubs("Detector2", 0*cm, 70*cm, 0.5*detector_thickness, 0*deg, 360*deg);
    G4LogicalVolume* logicDetector2 = new G4LogicalVolume(solidDetector2, scintillator_mat, "Detector2");
    fDetector2Volume = logicDetector2;
    fDetector2Position = G4ThreeVector(0, 0, detector2_position);
    new G4PVPlacement(nullptr, fDetector2Position, logicDetector2, "Detector2", logicWorld, false, 0, false);
    
    /* 
    // Detector 3
    G4double detector3_position = 17.7*meter;  //  923cm
    G4Tubs* solidDetector3 = new G4Tubs("Detector3", 0*cm, 70*cm, 0.5*detector_thickness, 0*deg, 360*deg);
    G4LogicalVolume* logicDetector3 = new G4LogicalVolume(solidDetector3, scintillator_mat, "Detector3");
    fDetector3Volume = logicDetector3;
    fDetector3Position = G4ThreeVector(0, 0, detector3_position);
    new G4PVPlacement(nullptr, fDetector3Position, logicDetector3, "Detector3", logicWorld, false, 0, false);

    // Detector 4
    G4double detector4_position = 11.2*meter;  //  1035cm
    G4Tubs* solidDetector4 = new G4Tubs("Detector4", 0*cm, 70*cm, 0.5*detector_thickness, 0*deg, 360*deg);
    G4LogicalVolume* logicDetector4 = new G4LogicalVolume(solidDetector4, scintillator_mat, "Detector4");
    fDetector4Volume = logicDetector4;
    fDetector4Position = G4ThreeVector(0, 0, detector4_position);
    new G4PVPlacement(nullptr, fDetector4Position, logicDetector4, "Detector4", logicWorld, false, 0, false);
    */
    // ========== VISUALIZATION ==========
    G4VisAttributes* Graphite_vis_att = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5));
    logicGraphite->SetVisAttributes(Graphite_vis_att);

    G4VisAttributes* detector1_vis_att = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0));
    detector1_vis_att->SetVisibility(true);
    logicDetector1->SetVisAttributes(detector1_vis_att);

    G4VisAttributes* detector2_vis_att = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0));
    detector2_vis_att->SetVisibility(true);
    logicDetector2->SetVisAttributes(detector2_vis_att);

    /* 
    G4VisAttributes* detector3_vis_att = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0));
    detector3_vis_att->SetVisibility(true);
    logicDetector3->SetVisAttributes(detector3_vis_att);

    G4VisAttributes* detector4_vis_att = new G4VisAttributes(G4Colour::Blue());
    detector4_vis_att->SetVisibility(true);
    logicDetector4->SetVisAttributes(detector4_vis_att);
    */

    G4VisAttributes* world_vis_att = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0, 0.1));
    world_vis_att->SetVisibility(true);
    world_vis_att->SetForceWireframe(true);
    logicWorld->SetVisAttributes(world_vis_att);

    // Set scoring volumes
    fScoringVolume = logicGraphite;

    return physWorld;
}
