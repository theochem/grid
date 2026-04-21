import numpy as np
import glob
#Get all files in the current directory
files_in_current_dir = glob.glob('*')

print(files_in_current_dir)

#Loop through each file
for filename in files_in_current_dir:
    print(filename)
    #Process only files that start with "md"
    if filename.startswith("md"):
        print(True)
        #Open file in read mode
        f1 = open(filename, "r") 
        file_content=f1.read()

        #Split the file into values
        values=file_content.split()    

        #Each data entry has 4 values: x, y, z, weight
        size=int(len(values)/4)
        degree=int(np.sqrt(size)-1)
        # print(degree,size)

        weights=[]
        points=[]

        #Extract weights (every 4th value starting from index 3)
        for i in range(3,len(values),4):
            weights+=[(float(values[i]))]
        # print(weights)

        #Extract all the points (first 3 values of every group of 4)
        for j in range(0,len(values)-2,4):
            points+=[[float(values[j]),float(values[j+1]),float(values[j+2])]]
        # print(points)

        #Ensure all weights are positive
        assert np.all(np.array(weights)>0)
        #Ensure weights sum to 4*pi
        assert np.isclose(np.sum(weights),4 * np.pi)
        for k in points:
            #Ensure all points lie on unit sphere (x^2 + y^2 + z^2 = 1)
            assert np.isclose((k[0]**2)+(k[1]**2)+(k[2]**2),1)

        #Save processed data in compressed NumPy format
        np.savez(f"data/md_{degree}_{size}.npz",degree=[degree],size=[size],weights=weights,points=points)
        

    else:
        print("false")

