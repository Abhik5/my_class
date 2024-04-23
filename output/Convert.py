# Open the .dat file for reading
with open("nonthermal00_cl_lensed.dat", "rb") as dat_file:
    # Read the contents of the .dat file
    dat_content = dat_file.read()

# Open a text file for writing
with open("output_Cl_lensed00.txt", "w") as txt_file:
    # Write the contents of the .dat file to the text file
    txt_file.write(dat_content.decode('utf-8'))  # Assuming the content is UTF-8 encoded
