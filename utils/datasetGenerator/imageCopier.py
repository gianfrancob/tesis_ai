import os

path = "/home/gianfranco/workspace/shared_volume/images_bolsasPivot+gen/"
kinds = ["train"]#, "val"]

for kind in kinds:
    with open(path + f"pivot_silobolsa_{kind}_balanced+_corregido.csv", "r") as f:
        original_lines = f.readlines()
        # print(original_lines)

        for line in original_lines[1:]:
            filename,width,height,clas,xmin,ymin,xmax,ymax = line.split(",")

            os.system(f'cp ../../..{filename} ./images/{kind}/')
