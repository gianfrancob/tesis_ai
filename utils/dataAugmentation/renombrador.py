import os

# folder = "Images"
folder = "Labels"

#stage = "train"
stage = "val"

claseName = "Pivot"
# claseName = "Silobolsa"

# LABEL RENOMBRATOR
# ===============================
path = f'/home/gianfranco/workspace/shared_volume/images_bolsasPivotAug/{stage}/Images/' + claseName

files = os.listdir(path)

for fil in sorted(files):
    if (fil != 'clf'):
        pathForTFrecord = path + "/" + fil
        pathForTFrecord = pathForTFrecord.replace("/home/gianfranco", "")
        print(pathForTFrecord)
        # print(path + "/" + fil)
        # with open(path + "/" + fil, "r") as f:
        #     original_lines = f.readlines()
        #     print(original_lines)
        #     print(len(original_lines))
        #     f.close()
        # with open(path + "/clf/" + fil, "w") as f:
        #     new_lines = []
        #     for l in original_lines[1:]:
        #         print(len(l))
        #         print(l)
        #         new_lines += [claseName + " " + l.replace("\r", "")]
        #         print(l)
        #     f.writelines(new_lines)
        #     f.close()


        #     print(new_lines)



# IMAGE and LABEL NUMERATOR
# ===============================
'''
path = f'/home/gianfranco/workspace/shared_volume/images_bolsasPivot+/{stage}/{folder}/' + claseName

files = os.listdir(path)

extension = ".JPG" if folder == "Images" else ".txt"

startingIndex = 31
i = 0
for fil in sorted(files):
    if (fil != 'clf'):
        srcPath = path + "/" + fil
        print(srcPath)
        dstPath = path + "/" + claseName + str(startingIndex + i) + extension
        os.rename(srcPath, dstPath)
        i += 1
'''
