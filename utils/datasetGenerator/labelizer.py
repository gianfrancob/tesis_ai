path = "/home/gianfranco/workspace/shared_volume/images_bolsasPivot+gen/"
kinds = ["train", "val"]

for kind in kinds:
    with open(path + f"pivot_silobolsa_{kind}.csv", "r") as f:
        original_lines = f.readlines()
        # print(original_lines)

        for line in original_lines[1:]:
            filename,width,height,clas,xmin,ymin,xmax,ymax = line.split(",")

            width = int(width)
            height = int(height)
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            name = filename.split("/")[-1].replace(".JPG", ".txt").replace(".jpg", ".txt")
            
            if (xmin > width) or (xmax > width) or (ymin > height) or (ymax > height):
            	width = int(height)
            	height = int(width)
            
            with open(path + f"/labels/{kind}/" + name, "a+") as label:
                bbwidth = xmax-xmin
                bbheigth = ymax-ymin
                xcenter = xmin + bbwidth/2
                ycenter = ymin + bbheigth/2
                clas = 0 if clas == "Pivot" else 1

                # Move read cursor to the start of file.
                label.seek(0)
                # If file is not empty then append '\n'
                data = label.read(100)
                if len(data) > 0:
                    label.write("\n")
                # Append text at the end of file
                label.write(f"{clas} {xcenter/width} {ycenter/height} {bbwidth/width} {bbheigth/height}")


