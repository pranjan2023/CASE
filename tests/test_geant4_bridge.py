import geant4_pybind as g4

# -------------------------
# Global accumulator
# -------------------------
total_energy = 0.0
event_count = 0


# -------------------------
# Sensitive Detector
# -------------------------
class WaterSD(g4.G4VSensitiveDetector):

    def __init__(self, name):
        super().__init__(name)

    def ProcessHits(self, step, history):
        global total_energy
        edep = step.GetTotalEnergyDeposit()
        total_energy += edep / g4.MeV
        return True


# -------------------------
# Detector Construction
# -------------------------
class DetectorConstruction(g4.G4VUserDetectorConstruction):

    def Construct(self):

        nist = g4.G4NistManager.Instance()

        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        water = nist.FindOrBuildMaterial("G4_WATER")

        # World volume
        world_size = 1.0 * g4.m
        world_box = g4.G4Box("World", world_size/2, world_size/2, world_size/2)
        world_lv = g4.G4LogicalVolume(world_box, vacuum, "WorldLV")
        world_pv = g4.G4PVPlacement(
            None,
            g4.G4ThreeVector(),
            world_lv,
            "WorldPV",
            None,
            False,
            0
        )

        # Water detector cube
        det_size = 10.0 * g4.cm
        det_box = g4.G4Box("Detector", det_size/2, det_size/2, det_size/2)
        det_lv = g4.G4LogicalVolume(det_box, water, "DetectorLV")

        g4.G4PVPlacement(
            None,
            g4.G4ThreeVector(),
            det_lv,
            "DetectorPV",
            world_lv,
            False,
            0
        )

        # Sensitive detector
        sd_manager = g4.G4SDManager.GetSDMpointer()
        water_sd = WaterSD("WaterSD")
        sd_manager.AddNewDetector(water_sd)
        det_lv.SetSensitiveDetector(water_sd)

        return world_pv


# -------------------------
# Primary Generator
# -------------------------
class PrimaryGenerator(g4.G4VUserPrimaryGeneratorAction):

    def __init__(self):
        super().__init__()

        self.particle_gun = g4.G4ParticleGun(1)

        particle_table = g4.G4ParticleTable.GetParticleTable()
        gamma = particle_table.FindParticle("gamma")

        self.particle_gun.SetParticleDefinition(gamma)
        self.particle_gun.SetParticleEnergy(1.0 * g4.MeV)

        self.particle_gun.SetParticlePosition(g4.G4ThreeVector(0, 0, -40*g4.cm))
        self.particle_gun.SetParticleMomentumDirection(g4.G4ThreeVector(0, 0, 1))

    def GeneratePrimaries(self, event):
        self.particle_gun.GeneratePrimaryVertex(event)


# -------------------------
# Event Action
# -------------------------
class EventAction(g4.G4UserEventAction):

    def EndOfEventAction(self, event):
        global event_count
        event_count += 1


# -------------------------
# Main
# -------------------------
def main():

    print("Initializing Geant4 kernel...")

    run_manager = g4.G4RunManager()

    run_manager.SetUserInitialization(DetectorConstruction())

    physics = g4.FTFP_BERT()
    run_manager.SetUserInitialization(physics)

    run_manager.SetUserAction(PrimaryGenerator())
    run_manager.SetUserAction(EventAction())

    run_manager.Initialize()

    print("Geometry defined: Water cube in vacuum")
    print("Physics list: FTFP_BERT")

    events = 100
    print(f"Running {events} events...")

    run_manager.BeamOn(events)

    print("Events processed:", event_count)
    print("Total energy deposited:", round(total_energy, 4), "MeV")

    assert event_count == events, "FAIL: Not all events processed"
    assert total_energy > 0, "FAIL: No energy deposition detected"

    print("PASS: Bridge functional")


if __name__ == "__main__":
    main()
