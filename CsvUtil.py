import csv
import os

### Create CSV for loss functions, including 4 columns:
### step, cuda_device_index initial_loss, aesthetic_loss and total_loss.
def createCsvFile(filePath):
    with open(filePath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "cuda_device_index", "initial_loss", "aesthetic_loss", "total_loss"])

### write one row of loss info.
def writeLossRow(filePath, step, cuda_device_index, initial_loss, aesthetic_loss, total_loss):
    with open(filePath, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([step, cuda_device_index, initial_loss, aesthetic_loss, total_loss])
