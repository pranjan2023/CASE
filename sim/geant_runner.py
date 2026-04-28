import geant4_pybind as g4
import random

def run_simulation(config):
    # --- Extract Configuration ---
    runs = config.get("runs", 10)
    events = config.get("events", 1000)
    base_seed = config.get("seed", 42)
    
    # State variable for energy tracking
    total_energy = 0.0

    # --- Nested Class: Sensitive Detector ---
    class WaterSD(g4.G4VSensitiveDetector):
        def __init__(self, name):
            super().__init__(name)

        def ProcessHits(self, step, history):
            nonlocal total_energy
            total_energy += step.GetTotalEnergyDeposit() / g4.MeV
            return True

    # --- Nested Class: Detector Construction ---
    class Detector(g4.G4VUserDetectorConstruction):
        def Construct(self):
            nist = g4.G4NistManager.Instance()

            vacuum = nist.FindOrBuildMaterial("G4_Galactic")

            material_name = config.get("material", "G4_WATER")

            allowed = ["G4_WATER", "G4_Si", "G4_Fe", "G4_Al", "G4_Cu", "G4_Pb"]
            if material_name not in allowed:
                material_name = "G4_WATER"

            water = nist.FindOrBuildMaterial(material_name)
            world = g4.G4Box("World", 0.5*g4.m, 0.5*g4.m, 0.5*g4.m)
            world_lv = g4.G4LogicalVolume(world, vacuum, "WorldLV")

            world_pv = g4.G4PVPlacement(
                None, g4.G4ThreeVector(),
                world_lv, "WorldPV",
                None, False, 0
            )

            size_val = float(config.get("detector_size_cm", 12))
            size = size_val * g4.cm

            depth_val = float(config.get("depth_cm", size_val))
            depth = depth_val * g4.cm
            det = g4.G4Box("Detector", size/2, size/2, depth/2)
            det_lv = g4.G4LogicalVolume(det, water, "DetectorLV")

            g4.G4PVPlacement(
                None, g4.G4ThreeVector(),
                det_lv, "DetectorPV",
                world_lv, False, 0
            )

            sd = WaterSD("WaterSD")
            g4.G4SDManager.GetSDMpointer().AddNewDetector(sd)
            det_lv.SetSensitiveDetector(sd)

            return world_pv

    # --- Nested Class: Primary Generator ---
    class Gun(g4.G4VUserPrimaryGeneratorAction):
        def __init__(self):
            super().__init__()
            self.gun = g4.G4ParticleGun(1)
            table = g4.G4ParticleTable.GetParticleTable()
            gamma = table.FindParticle("gamma")

            self.gun.SetParticleDefinition(gamma)
            self.gun.SetParticleEnergy(1*g4.MeV)
            self.gun.SetParticlePosition(g4.G4ThreeVector(0, 0, -40*g4.cm))
            self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(0, 0, 1))

        def GeneratePrimaries(self, event):
            self.gun.GeneratePrimaryVertex(event)

    # --- Run Manager Setup ---
    rm = g4.G4RunManager()
    rm.SetUserInitialization(Detector())
    
    physics_name = config.get("physics_list", "FTFP_BERT")

    physics_map = {
        "FTFP_BERT": g4.FTFP_BERT,
        "QGSP_BERT": g4.QGSP_BERT,
        "QGSP_BIC": g4.QGSP_BIC,
    }

    physics_cls = physics_map.get(physics_name, g4.FTFP_BERT)
    rm.SetUserInitialization(physics_cls())
    rm.SetUserAction(Gun())
    rm.Initialize()

    scores = []

    # --- Execution Loop ---
    for i in range(runs):
        total_energy = 0.0  # Reset for each run
        if hasattr(g4, "G4Random"):
            g4.G4Random.setTheSeed(base_seed + i)
        rm.BeamOn(events)
        noise = random.uniform(-0.01, 0.01)        
        scores.append(total_energy / events + noise)
    return {
        "events": events,
        "scores": scores
    }