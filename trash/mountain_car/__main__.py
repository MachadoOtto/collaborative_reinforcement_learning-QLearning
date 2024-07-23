# Main MountainCar

## Imports
from instance import MountainCar

## Main
if __name__ == "__main__":
    mountainCar = MountainCar()
    mountainCar.train()
    mountainCar.save()
    mountainCar.load()
    mountainCar.play()