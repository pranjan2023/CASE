import geant4_pybind as g4
import sys

seed = int(sys.argv[1])

g4.G4Random.setTheSeed(seed)

total_energy = 0.0


class WaterSD(g4.G4VSensitiveDetector):

    def __init__(self, name):
        super().__init__(name)

    def ProcessHits(self, step, history):
        global total_energy
        total_energy += step.GetTotalEnergyDeposit() / g4.MeV
        return True


class Detector(g4.G4VUserDetectorConstruction):

    def Construct(self):

        nist = g4.G4NistManager.Instance()

        vacuum = nist.FindOrBuildMaterial("G4_Galactic")
        water = nist.FindOrBuildMaterial("G4_WATER")

        world = g4.G4Box("World",0.5*g4.m,0.5*g4.m,0.5*g4.m)
        world_lv = g4.G4LogicalVolume(world,vacuum,"WorldLV")

        world_pv = g4.G4PVPlacement(None,g4.G4ThreeVector(),world_lv,"WorldPV",None,False,0)

        det = g4.G4Box("Detector",5*g4.cm,5*g4.cm,5*g4.cm)
        det_lv = g4.G4LogicalVolume(det,water,"DetectorLV")

        g4.G4PVPlacement(None,g4.G4ThreeVector(),det_lv,"DetectorPV",world_lv,False,0)

        sd = WaterSD("WaterSD")
        g4.G4SDManager.GetSDMpointer().AddNewDetector(sd)
        det_lv.SetSensitiveDetector(sd)

        return world_pv


class Gun(g4.G4VUserPrimaryGeneratorAction):

    def __init__(self):
        super().__init__()

        self.gun = g4.G4ParticleGun(1)

        table = g4.G4ParticleTable.GetParticleTable()
        gamma = table.FindParticle("gamma")

        self.gun.SetParticleDefinition(gamma)
        self.gun.SetParticleEnergy(1*g4.MeV)

        self.gun.SetParticlePosition(g4.G4ThreeVector(0,0,-40*g4.cm))
        self.gun.SetParticleMomentumDirection(g4.G4ThreeVector(0,0,1))

    def GeneratePrimaries(self,event):
        self.gun.GeneratePrimaryVertex(event)


rm = g4.G4RunManager()

rm.SetUserInitialization(Detector())
rm.SetUserInitialization(g4.FTFP_BERT())
rm.SetUserAction(Gun())

rm.Initialize()

rm.BeamOn(100)

print(total_energy)
