import os
RootPath = os.path.dirname(os.path.realpath(__file__))

DataPath = os.path.join(RootPath, "Data")

ImgPath = os.path.join(RootPath, 'IMG')
if not os.path.exists(ImgPath):
    os.mkdir(ImgPath)

ModelPath = os.path.join(RootPath, 'Model')
if not os.path.exists(ModelPath):
    os.mkdir(ModelPath)
